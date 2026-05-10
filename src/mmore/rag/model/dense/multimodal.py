import re
from typing import Optional

import numpy as np
import torch
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from transformers import AutoModelForImageTextToText, AutoProcessor

from ....type import MultimodalSample
from ..vision.image_utils import load_images_from_paths


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

        This is an interface meant for implementing multimodal embedding models.

        Text embedding models are used to map text to a vector (a point in n-dimensional
        space).

        Texts that are similar will usually be mapped to points that are close to each
        other in this space. The exact details of what's considered "similar" and how
        "distance" is measured in this space are dependent on the specific embedding model.

        This abstraction contains a method for embedding a list of documents and a method
        for embedding a query text. The embedding of a query text is expected to be a single
        vector, while the embedding of a list of documents is expected to be a list of
        vectors.

        Usually the query embedding is identical to the document embedding, but the
        abstraction allows treating them independently.

        In addition to the synchronous methods, this interface also provides asynchronous
        versions of the methods.

        By default, the asynchronous methods are implemented using the synchronous methods;
        however, implementations may choose to override the asynchronous methods with
        an async native implementation for performance reasons.
        """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search docs.

        Args:
            texts: One text string per chunk. Image paths may appear inside
                ``<|image|>…<|image|>`` pairs; those are stripped to the model’s
                image token and the files are loaded. Plain text chunks work too.

        Returns:
            List of embeddings.
        """
        embeddings = []

        for text in texts:
            # Parsers mark image positions with "<attachment>" in the text, but that string has no file path—only <|image|>path<|image|> does.
            # Remove it so it does not end up as junk tokens in the model prompt.
            text = text.replace("<attachment>", "")
            prompt, extracted_paths = MultimodalEmbeddings._extract_multimodal_inputs(
                text, proc_token="<|image|>"
            )
            # Missing/invalid images are skipped by helper; text-only embedding still works.
            images = load_images_from_paths(extracted_paths)
            # Qwen2-VL processors expect <|image_pad|> (processor.image_token), not <|image|>.
            # Otherwise vision features are built but input_ids get no image placeholders → mismatch.
            vision_placeholder = getattr(self.processor, "image_token", None)
            if vision_placeholder:
                prompt = prompt.replace("<|image|>", vision_placeholder)

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
        """Build one string for embedding: modality lines then chunk text.

        Each modality becomes ``<|type|>value<|type|>`` on its own line (paths for
        images). ``sample.text`` is appended as-is.
        """
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
        meta = (
            sample.metadata.to_dict()
            if hasattr(sample.metadata, "to_dict")
            else dict(sample.metadata or {})
        )
        return Document(MultimodalEmbeddings._multimodal_to_text(sample), metadata=meta)

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
