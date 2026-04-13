"""
Multimodal LLM: generate from text + images (vision). Extensible per provider (OpenAI, HF, etc.).
Hugging Face: supports Qwen2.5-VL and similar vision-language models (e.g. Qwen/Qwen2.5-VL-3B-Instruct).
"""

import base64
import io
import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .llm import LLM, LLMConfig

logger = logging.getLogger(__name__)

# Default Hugging Face vision model: good quality / size tradeoff, ~3B params
DEFAULT_HF_VISION_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


def _images_to_base64_data_urls(images: List[Any]) -> List[str]:
    """Convert PIL Images (or objects with save()) to data URLs for message content."""
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
    """Build content list for HumanMessage: one text block then one image_url per image."""
    content: List[dict] = [{"type": "text", "text": text}]
    for url in image_data_urls:
        content.append({
            "type": "image_url",
            "image_url": {"url": url, "detail": "auto"},
        })
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
        """Generate a text response from prompt text and optional images."""
        ...


class OpenAIMultimodalAdapter(BaseMultimodalLLM):
    """Wrap a LangChain OpenAI (or compatible) chat model to accept text + images."""

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
    """
    Vision-language model on Hugging Face (e.g. Qwen2.5-VL-3B-Instruct).
    Lazy-loads model and processor on first use. Uses PIL images directly.
    """

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

            from transformers import AutoProcessor

            # Qwen2.5-VL / Qwen2-VL: model class name depends on transformers version
            model_cls = (
                getattr(transformers, "Qwen2_5_VLForConditionalGeneration", None)
                or getattr(transformers, "Qwen2VLForConditionalGeneration", None)
            )
            if model_cls is None:
                from transformers import AutoModelForImageTextToText

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
        import torch

        self._load()
        images = images or []
        # Build Qwen-style messages: images first (PIL or path), then text
        content: List[dict] = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": text})
        messages = [{"role": "user", "content": content}]

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
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return (output_text[0] if output_text else "").strip()


def get_multimodal_llm(config: LLMConfig) -> BaseMultimodalLLM:
    """Build a multimodal LLM adapter from config. OPENAI -> OpenAI adapter; HF -> Hugging Face vision (e.g. Qwen2.5-VL)."""
    org = (getattr(config, "organization", None) or config.provider or "").upper()
    if org == "HF":
        model_id = config.llm_name
        if not model_id or model_id == "gpt2":
            model_id = DEFAULT_HF_VISION_MODEL
        return HuggingFaceVisionAdapter(
            model_id=model_id,
            max_new_tokens=config.max_new_tokens or 512,
        )
    if org == "OPENAI":
        base = LLM.from_config(config)
        return OpenAIMultimodalAdapter(base)
    # Fallback: wrap text LLM as OpenAI-style; may work for other providers
    base = LLM.from_config(config)
    logger.info("Using generic multimodal adapter for organization=%s", org)
    return OpenAIMultimodalAdapter(base)
