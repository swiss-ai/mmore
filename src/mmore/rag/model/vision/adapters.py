from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

import torch
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from transformers import AutoModelForImageTextToText, AutoProcessor

from ...llm import LLM, LLMConfig
from .image_utils import build_vision_content, images_to_base64_data_urls

DEFAULT_HF_VISION_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"


class BaseMultimodalLLM(ABC):
    """Interface for vision-capable LLMs: (text, images) -> answer text."""

    @abstractmethod
    def invoke_with_images(
        self,
        text: str,
        images: Optional[List[Any]] = None,
        system_prompt: Optional[str] = None,
    ) -> str: ...


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
        image_urls = images_to_base64_data_urls(images or [])
        content = build_vision_content(text, image_urls)
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
        import transformers

        model_cls = getattr(
            transformers, "Qwen2_5_VLForConditionalGeneration", None
        ) or getattr(transformers, "Qwen2VLForConditionalGeneration", None)
        if model_cls is None:
            model_cls = AutoModelForImageTextToText

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        try:
            self._model = model_cls.from_pretrained(
                self.model_id,
                torch_dtype="auto",
                device_map=self.device_map,
            )
        except NotImplementedError as exc:
            # Some torch/accelerate/transformers combinations can fail when
            # dispatching from meta tensors with device_map="auto".
            if "meta tensor" not in str(exc):
                raise
            self._model = model_cls.from_pretrained(
                self.model_id,
                torch_dtype="auto",
                device_map=None,
                low_cpu_mem_usage=False,
            )
            if torch.cuda.is_available():
                self._model = self._model.to("cuda")

    def _get_model_device(self) -> torch.device:
        if self._model is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_device = getattr(self._model, "device", None)
        if model_device is not None and str(model_device) != "meta":
            return model_device

        device_map = getattr(self._model, "hf_device_map", None)
        if isinstance(device_map, dict):
            for mapped_device in device_map.values():
                if isinstance(mapped_device, str) and mapped_device not in {
                    "disk",
                    "meta",
                }:
                    return torch.device(mapped_device)

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def invoke_with_images(
        self,
        text: str,
        images: Optional[List[Any]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        self._load()
        images = images or []
        content: List[dict] = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": text})
        messages = [{"role": "user", "content": content}]
        if system_prompt:
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                },
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
        inputs = inputs.to(self._get_model_device())
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
