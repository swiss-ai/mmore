import logging
import threading
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union

import torch
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from ...llm import LLM, LLMConfig
from .image_utils import build_vision_content, images_to_base64_data_urls

DEFAULT_HF_VISION_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
logger = logging.getLogger(__name__)


class BaseMultimodalLLM(ABC):
    """Interface for vision-capable LLMs: (text, images) -> answer text."""

    def _load(self) -> None:
        """Eager-load local weights when supported (no-op by default)."""

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
        messages.append(HumanMessage(content=content))  # type: ignore[arg-type]
        response = self._model.invoke(messages)
        return getattr(response, "content", str(response)) or ""


class HuggingFaceVisionAdapter(BaseMultimodalLLM):
    """Vision-language adapter for Hugging Face models such as Qwen2.5-VL."""

    _load_lock = threading.Lock()
    _model_cache: dict[str, Tuple[Any, Any]] = {}

    def __init__(
        self,
        model_id: str = DEFAULT_HF_VISION_MODEL,
        max_new_tokens: int = 512,
        device_map: Optional[Union[str, dict]] = None,
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.device_map = device_map
        self._model = None
        self._processor = None
        self._inference_device: Any = None

    def _resolve_model_cls(self):
        """Pick the HF class that matches the checkpoint (Qwen2-VL ≠ Qwen2.5-VL)."""
        import transformers

        config = AutoConfig.from_pretrained(self.model_id)
        model_type = (getattr(config, "model_type", None) or "").lower()
        type_to_cls = {
            "qwen2_5_vl": "Qwen2_5_VLForConditionalGeneration",
            "qwen2_vl": "Qwen2VLForConditionalGeneration",
            "qwen3_vl": "Qwen3VLForConditionalGeneration",
        }
        cls_name = type_to_cls.get(model_type)
        if cls_name is not None:
            model_cls = getattr(transformers, cls_name, None)
            if model_cls is not None:
                return model_cls
        return AutoModelForImageTextToText

    def _build_from_pretrained_kwargs(self, dtype: Any) -> dict:
        load_kwargs: dict = {"low_cpu_mem_usage": False, "torch_dtype": dtype}
        if self.device_map is not None:
            load_kwargs["device_map"] = self.device_map
        return load_kwargs

    def _load(self) -> None:
        if self._model is not None:
            return

        with HuggingFaceVisionAdapter._load_lock:
            if self._model is not None:
                return

            cached = HuggingFaceVisionAdapter._model_cache.get(self.model_id)
            if cached is not None:
                self._model, self._processor = cached
                self._inference_device = self._get_model_device()
                return

            model_cls = self._resolve_model_cls()
            self._processor = AutoProcessor.from_pretrained(self.model_id)

            dtype = (
                torch.bfloat16  # type: ignore[reportPrivateImportUsage]
                if torch.cuda.is_available()
                else torch.float32  # type: ignore[reportPrivateImportUsage]
            )
            load_kwargs = self._build_from_pretrained_kwargs(dtype)
            # Default path: no device_map -> avoid accelerate meta-tensor dispatch.
            self._model = model_cls.from_pretrained(self.model_id, **load_kwargs)

            self._inference_device = torch.device("cpu")  # type: ignore[reportPrivateImportUsage]
            if torch.cuda.is_available() and self.device_map is None:
                try:
                    self._model = self._model.to("cuda")  # type: ignore[call-arg]
                    self._inference_device = torch.device("cuda")  # type: ignore[reportPrivateImportUsage]
                except NotImplementedError as exc:
                    if "meta tensor" not in str(exc):
                        raise
                    logger.warning(
                        "Could not move vision model to CUDA (%s); keeping CPU.",
                        exc,
                    )

            HuggingFaceVisionAdapter._model_cache[self.model_id] = (
                self._model,
                self._processor,
            )

    def _get_model_device(self) -> Any:
        if self._inference_device is not None:
            return self._inference_device

        if self._model is None:
            return torch.device(  # type: ignore[reportPrivateImportUsage]
                "cuda" if torch.cuda.is_available() else "cpu"
            )

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
                    return torch.device(mapped_device)  # type: ignore[reportPrivateImportUsage]

        return torch.device(  # type: ignore[reportPrivateImportUsage]
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def invoke_with_images(
        self,
        text: str,
        images: Optional[List[Any]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        self._load()
        if self._processor is None or self._model is None:
            raise RuntimeError("Vision model failed to load")
        processor = self._processor
        model = self._model
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

        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        vision_info = process_vision_info(messages)
        image_inputs, video_inputs, *_ = vision_info
        inputs = processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._get_model_device())
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
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
