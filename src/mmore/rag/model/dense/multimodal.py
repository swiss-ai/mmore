"""Unified multimodal utilities for RAG.

This module contains:
- multimodal dense embeddings for retrieval/indexing
- image path/image loading helpers used at inference time
- multimodal generation adapters (OpenAI-compatible and HF vision models)
"""

import base64
import io
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import torch
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from ....type import MultimodalSample
from ...llm import LLM, LLMConfig

logger = logging.getLogger(__name__)
DEFAULT_HF_VISION_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
class MultimodalEmbeddings(Embeddings):
    """Dense embedding model that can consume text and optional images."""

    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.device = self.model.device

        """Interface for embedding models.

        This implementation follows the standard embedding abstraction, while extending
        it with multimodal inputs (text plus optional images).

        Embedding models map content to a vector (a point in n-dimensional space).

        Similar inputs are usually mapped to nearby points in that space. The exact
        notion of "similarity" and distance depends on the specific model.

        The abstraction provides a method for embedding a list of documents and a method
        for embedding a query text. The query embedding is expected to be a single vector,
        while document embeddings are expected to be a list of vectors.

        Usually the query embedding is identical to the document embedding, but the
        abstraction allows treating them independently.

        In addition to the synchronous methods, this interface also provides asynchronous
        versions of the methods.

        By default, the asynchronous methods are implemented using the synchronous methods;
        however, implementations may choose to override the asynchronous methods with
        an async native implementation for performance reasons.
        """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents for dense retrieval.

        Args:
            texts: Serialized document payloads. They can include multimodal tags.

        Returns:
            List of embeddings.
        """
        embeddings = []

        for text in texts:
            # Legacy placeholder used upstream in processed text chunks.
            text = text.replace("<attachment>", "")
            prompt, extracted_paths = MultimodalEmbeddings._extract_multimodal_inputs(
                text, proc_token="<|image|>"
            )
            # Missing/invalid images are skipped by helper; text-only embedding still works.
            images = load_images_from_paths(extracted_paths)

            if images:
                inputs = self.processor(
                    text=prompt, images=images, return_tensors="pt"
                ).to(self.device)
            else:
                inputs = self.processor(text=prompt, return_tensors="pt").to(
                    self.device
                )

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                last_hidden_state = outputs.hidden_states[0]
                embedding = last_hidden_state.mean(dim=1)  # Shape: (1, hidden_size)
                embedding = embedding.cpu().numpy().squeeze(0).astype(np.float32)
                embeddings.append(embedding)

        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed query text.

        Args:
            text: Text to embed.

        Returns:
            Embedding.
        """
        return self.embed_documents([text])[0]

    @staticmethod
    def _multimodal_to_text(sample: MultimodalSample):
        """Serialize modalities and text into the retrieval-time text format."""
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
        return Document(
            MultimodalEmbeddings._multimodal_to_text(sample), metadata=sample.metadata
        )

    @staticmethod
    def _extract_multimodal_inputs(
        text, proc_token: str, pattern: Optional[str] = None
    ) -> tuple[str, list[str]]:
        pattern = pattern or rf"{re.escape(proc_token)}(.*?){re.escape(proc_token)}"

        # Find all matches in the input string
        extracted_strings = re.findall(pattern, text)

        # Replace all matches with an empty string
        cleaned_string = re.sub(pattern, proc_token, text)

        # Return a tuple with the cleaned string and the list of extracted strings
        return (cleaned_string, extracted_strings)


def aggregate_image_paths(docs: List[Document]) -> List[str]:
    """Collect unique, non-empty image paths from `doc.metadata["image_paths"]`."""
    image_paths: List[str] = []
    seen: set[str] = set()
    for doc in docs:
        for path in doc.metadata.get("image_paths") or []:
            path_str = str(path).strip()
            if path_str and path_str not in seen:
                seen.add(path_str)
                image_paths.append(path_str)
    return image_paths


def load_images_from_paths(paths: List[str], max_images: int = 20) -> List[Any]:
    """Load images from paths, limit count, and return RGB copies."""
    loaded: List[Any] = []
    for path in paths[:max_images]:
        try:
            p = Path(path)
            if not p.exists():
                logger.debug("Image path does not exist: %s", path)
                continue
            with Image.open(path) as img:
                loaded.append(img.convert("RGB").copy())
        except Exception as e:
            logger.debug("Failed to load image %s: %s", path, e)
    return loaded


def _images_to_base64_data_urls(images: List[Any]) -> List[str]:
    """Convert PIL Images (or objects with save()) to data URLs."""
    urls = []
    for img in images:
        try:
            buf = io.BytesIO()
            if hasattr(img, "save"):
                img.save(buf, format="PNG")
            else:
                raise TypeError("Image object must have save() (e.g. PIL.Image)")
            b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
            urls.append(f"data:image/png;base64,{b64}")
        except Exception as e:
            logger.warning("Skip image in multimodal message: %s", e)
    return urls


def _build_vision_content(text: str, image_data_urls: List[str]) -> List[dict]:
    """Build content list for HumanMessage."""
    content: List[dict] = [{"type": "text", "text": text}]
    for url in image_data_urls:
        content.append({"type": "image_url", "image_url": {"url": url, "detail": "auto"}})
    return content


class BaseMultimodalLLM(ABC):
    """Interface for vision-capable LLMs: (text, images) -> answer text."""

    @abstractmethod
    def invoke_with_images(
        self,
        text: str,
        images: Optional[List[Any]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        ...


class OpenAIMultimodalAdapter(BaseMultimodalLLM):
    """Wrap a LangChain chat model to accept text + images."""

    def __init__(self, model: BaseChatModel):
        self._model = model

    def invoke_with_images(
        self,
        text: str,
        images: Optional[List[Any]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        image_urls = _images_to_base64_data_urls(images or [])
        content = _build_vision_content(text, image_urls)
        messages: List[Union[SystemMessage, HumanMessage]] = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=content))
        response = self._model.invoke(messages)
        return getattr(response, "content", str(response)) or ""


class HuggingFaceVisionAdapter(BaseMultimodalLLM):
    """Vision-language adapter for Hugging Face models such as Qwen2.5-VL."""

    def __init__(
        self,
        model_id: str = DEFAULT_HF_VISION_MODEL,
        max_new_tokens: int = 512,
        device_map: str = "auto",
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.device_map = device_map
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import transformers

            model_cls = getattr(
                transformers, "Qwen2_5_VLForConditionalGeneration", None
            ) or getattr(transformers, "Qwen2VLForConditionalGeneration", None)
            if model_cls is None:
                model_cls = AutoModelForImageTextToText

            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = model_cls.from_pretrained(
                self.model_id,
                torch_dtype="auto",
                device_map=self.device_map,
            )
            logger.info("Loaded Hugging Face vision model: %s", self.model_id)
        except Exception as e:
            logger.exception("Failed to load HF vision model %s: %s", self.model_id, e)
            raise

    def invoke_with_images(
        self,
        text: str,
        images: Optional[List[Any]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        self._load()
        images = images or []
        # Qwen-style multimodal messages: image blocks followed by text.
        content: List[dict] = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": text})
        messages = [{"role": "user", "content": content}]
        if system_prompt:
            messages.insert(
                0, {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            )

        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            raise ImportError(
                "For Hugging Face vision models install: pip install qwen-vl-utils"
            ) from None

        text_prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device)
        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return (output_text[0] if output_text else "").strip()


def get_multimodal_llm(config: LLMConfig) -> BaseMultimodalLLM:
    """Build a multimodal LLM adapter from config."""
    provider = (getattr(config, "organization", None) or config.provider or "").upper()
    if provider == "HF":
        model_id = config.llm_name
        if not model_id or model_id == "gpt2":
            model_id = DEFAULT_HF_VISION_MODEL
        return HuggingFaceVisionAdapter(
            model_id=model_id,
            max_new_tokens=config.max_new_tokens or 512,
        )
    base = LLM.from_config(config)
    return OpenAIMultimodalAdapter(base)
