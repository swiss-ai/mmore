import gc
import re
from typing import Optional

import numpy as np
import torch
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from transformers import AutoModelForImageTextToText, AutoProcessor

from ....type import MultimodalSample
from ..vision.image_utils import load_images_from_paths

_DEFAULT_MAX_IMAGES = 20
_DEFAULT_MAX_IMAGE_SIDE = 768


def _release_device_memory() -> None:
    """Free accelerator cache after unloading a large model (indexer calls this too)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _release_after_chunk() -> bool:
    """Per-chunk cache flush only on MPS; CUDA batch indexing stays fast."""
    mps = getattr(torch.backends, "mps", None)
    return mps is not None and mps.is_available()


def _sequence_hidden(outputs) -> torch.Tensor:
    """Last-layer sequence hidden states (Qwen2.5-VL may omit ``last_hidden_state``)."""
    last = getattr(outputs, "last_hidden_state", None)
    if last is not None:
        return last
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states:
        return hidden_states[-1]
    raise AttributeError(
        "Model output has no last_hidden_state or hidden_states; "
        "forward must use output_hidden_states=True."
    )


def _mean_pool(
    hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]
) -> torch.Tensor:
    if attention_mask is None:
        return hidden.mean(dim=1)
    mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype, device=hidden.device)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class MultimodalEmbeddings(Embeddings):
    """Dense embedding model that can consume text and optional images."""

    def __init__(
        self,
        model_name: str,
        max_images: int = _DEFAULT_MAX_IMAGES,
        max_image_side: int = _DEFAULT_MAX_IMAGE_SIDE,
    ):
        super().__init__()
        self.max_images = max_images
        self.max_image_side = max_image_side
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # type: ignore[reportPrivateImportUsage]
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.device = self.model.device

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search docs.

        Args:
            texts: One text string per chunk. Image paths may appear inside
                ``<|image|>…<|image|>`` pairs; those are stripped to the model's
                image token and the files are loaded. Plain text chunks work too.

        Returns:
            List of embeddings.
        """
        embeddings: list[list[float]] = []

        proc_token = "<|image|>"
        vision_placeholder = getattr(self.processor, "image_token", None)
        if not isinstance(vision_placeholder, str) or not vision_placeholder:
            vision_placeholder = proc_token

        for text in texts:
            text = text.replace("<attachment>", "")
            paths, prompt = MultimodalEmbeddings._cap_image_tags_in_text(
                text,
                proc_token=proc_token,
                vision_placeholder=vision_placeholder,
                max_images=self.max_images,
            )
            images = load_images_from_paths(
                paths,
                max_images=self.max_images,
                max_side=self.max_image_side,
            )

            inputs = None
            outputs = None
            try:
                if images:
                    inputs = self.processor(
                        text=prompt, images=images, return_tensors="pt"
                    ).to(self.device)
                else:
                    inputs = self.processor(text=prompt, return_tensors="pt").to(
                        self.device
                    )

                with torch.inference_mode():
                    outputs = self.model(**inputs, output_hidden_states=True)
                    hidden = _sequence_hidden(outputs)
                    pooled = _mean_pool(hidden, inputs.get("attention_mask"))
                    embedding = pooled.cpu().numpy().squeeze(0).astype(np.float32)
                embeddings.append(embedding.tolist())
            finally:
                del inputs, outputs, images
                if _release_after_chunk():
                    _release_device_memory()

        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed query text."""
        return self.embed_documents([text])[0]

    @staticmethod
    def _cap_image_tags_in_text(
        text: str,
        proc_token: str,
        vision_placeholder: str,
        max_images: int,
    ) -> tuple[list[str], str]:
        """Keep at most ``max_images`` placeholders; count must match images passed to Qwen."""
        pattern = rf"{re.escape(proc_token)}(.*?){re.escape(proc_token)}"
        paths: list[str] = []
        seen: set[str] = set()

        def replacer(match: re.Match) -> str:
            path = match.group(1).strip()
            if not path or path in seen or len(paths) >= max_images:
                return ""
            seen.add(path)
            paths.append(path)
            return vision_placeholder

        prompt = re.sub(pattern, replacer, text)
        return paths, prompt

    @staticmethod
    def _multimodal_to_text(sample: MultimodalSample):
        """Build one string for embedding: modality lines then chunk text."""
        s = "\n".join(
            [
                f"<|{modality.type}|>{modality.value}<|{modality.type}|>"
                for modality in sample.modalities
            ]
        )
        s += sample.text
        return s

    @staticmethod
    def _multimodal_to_doc(sample: MultimodalSample) -> Document:
        meta = sample.metadata.to_dict()
        return Document(MultimodalEmbeddings._multimodal_to_text(sample), metadata=meta)
