"""Judge-specific LLM loading (does not alter the global RAG LLM path)."""

try:
    import torch
except ImportError:
    torch = None

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

from ...utils import load_config
from ..llm import LLM, LLMConfig


def judge_llm_from_config(config: str | LLMConfig) -> BaseChatModel:
    """Load an LLM for judge evaluation.

    HuggingFace pipelines return generation-only output here so JSON parsing
    stays reliable. All other providers reuse the standard ``LLM.from_config``.
    """
    if isinstance(config, str):
        config = load_config(config, LLMConfig)
    if config.provider != "HF":
        return LLM.from_config(config)

    if torch is None:
        raise ImportError(
            "torch is required for HuggingFace models. "
            "Install it with: uv pip install 'mmore[cpu]' or uv pip install 'mmore[cu126]'"
        )

    pipeline_kwargs = {**config.generation_kwargs, "return_full_text": False}
    if torch.backends.mps.is_available():
        return ChatHuggingFace(
            llm=HuggingFacePipeline.from_model_id(
                model_id=config.llm_name,
                task="text-generation",
                device_map="mps",
                pipeline_kwargs=pipeline_kwargs,
            )
        )
    if torch.cuda.is_available():
        current_device = LLM.device_count
        LLM.device_count = (LLM.device_count + 1) % LLM._get_nb_devices()
    else:
        current_device = -1

    return ChatHuggingFace(
        llm=HuggingFacePipeline.from_model_id(
            config.llm_name,
            task="text-generation",
            device=current_device,
            pipeline_kwargs=pipeline_kwargs,
        )
    )
