import os
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any, ClassVar, Optional

import torch
from langchain_anthropic import ChatAnthropic
from langchain_cohere import ChatCohere
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI

from ..utils import load_config

logger = getLogger(__name__)

_OPENAI_MODELS = [
    "gpt-4",
    "gpt-4-turbo",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-3.5-turbo",
    "davinci",
    "curie",
    "babbage",
    "ada",
]
_ANTHROPIC_MODELS = [
    "claude-1",
    "claude-1.3",
    "claude-2",
    "claude-instant-1",
    "claude-instant-1.1",
    "claude-instant-1.2",
]
_MISTRAL_MODELS = ["mistral-7b", "mistral-7b-instruct", "mistral-7b-chat"]
_COHERE_MODELS = [
    "command",
    "command-light",
    "command-nightly",
    "summarize",
    "embed-english-v2.0",
]

loaders: dict[str, Any] = {
    "OPENAI": ChatOpenAI,
    "ANTHROPIC": ChatAnthropic,
    "MISTRAL": ChatMistralAI,
    "COHERE": ChatCohere,
    "HF": ChatHuggingFace,
}


def _infer_provider_from_legacy_model_name(llm_name: str) -> Optional[str]:
    if llm_name in _OPENAI_MODELS:
        return "OPENAI"
    if llm_name in _ANTHROPIC_MODELS:
        return "ANTHROPIC"
    if llm_name in _MISTRAL_MODELS:
        return "MISTRAL"
    if llm_name in _COHERE_MODELS:
        return "COHERE"
    return None


@dataclass
class LLMConfig:
    llm_name: str
    provider: Optional[str] = None
    base_url: Optional[str] = None
    # Deprecated alias of "provider". Kept for backward compatibility.
    organization: Optional[str] = None
    # Optional override (e.g., "SWISSAI_API_KEY") for API-key lookup.
    api_key_env_var: Optional[str] = None
    max_new_tokens: Optional[int] = None
    temperature: float = 0.7
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    client_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.provider is not None:
            self.provider = self.provider.upper()

        if self.organization is not None:
            self.organization = self.organization.upper()

        if self.provider and self.organization and self.provider != self.organization:
            if self.organization in loaders:
                raise ValueError(
                    "Both 'provider' and deprecated 'organization' are set with different values. "
                    "Set only 'provider'."
                )
            if self.base_url is None:
                raise ValueError(
                    f"Unknown organization '{self.organization}' without base_url. "
                    "Set 'provider' explicitly."
                )
            self.api_key_env_var = (
                self.api_key_env_var or f"{self.organization}_API_KEY"
            )
            logger.warning(
                "Using provider='%s' for routing and '%s' for API key env var. "
                "Set provider and api_key_env_var explicitly.",
                self.provider,
                self.api_key_env_var,
            )

        if self.provider is None and self.organization is not None:
            if self.organization in loaders:
                self.provider = self.organization
                logger.warning(
                    "LLMConfig.organization is deprecated. Use provider='%s' instead.",
                    self.provider,
                )
            elif self.base_url is not None:
                # Backward compatibility for openai-compatible endpoints
                # with custom key env var naming (e.g., SWISSAI_API_KEY).
                self.provider = "OPENAI"
                self.api_key_env_var = (
                    self.api_key_env_var or f"{self.organization}_API_KEY"
                )
                logger.warning(
                    "LLMConfig.organization='%s' is deprecated. "
                    "Set provider='OPENAI' and api_key_env_var='%s'.",
                    self.organization,
                    self.api_key_env_var,
                )
            else:
                raise ValueError(
                    f"Unknown organization '{self.organization}'. Set provider explicitly."
                )

        if self.provider is None:
            inferred_provider = _infer_provider_from_legacy_model_name(self.llm_name)
            if inferred_provider is not None:
                self.provider = inferred_provider
                logger.warning(
                    "Inferring provider from llm_name is deprecated. "
                    "Set provider='%s' explicitly.",
                    self.provider,
                )
            elif self.base_url is not None:
                self.provider = "OPENAI"
                logger.warning(
                    "No provider configured. Defaulting to provider='OPENAI' because base_url is set."
                )
            else:
                self.provider = "HF"
                logger.warning("No provider configured. Defaulting to provider='HF'.")

        if self.provider not in loaders:
            supported = ", ".join(sorted(loaders))
            raise ValueError(
                f"Unsupported provider '{self.provider}'. Supported providers: {supported}."
            )

    @property
    def generation_kwargs(self) -> dict[str, Any]:
        provider = self.resolved_provider
        max_token_key = (
            "max_new_tokens"
            if provider in {"ANTHROPIC", "MISTRAL", "COHERE", "HF"}
            else "max_completion_tokens"
        )
        return {"temperature": self.temperature, max_token_key: self.max_new_tokens}

    @property
    def _resolved_api_key_env_var(self) -> str:
        return self.api_key_env_var or f"{self.resolved_provider}_API_KEY"

    @property
    def resolved_provider(self) -> str:
        if self.provider is None:
            raise ValueError(
                "Provider resolution failed; provider should never be None."
            )
        return self.provider

    @property
    def api_key(self) -> Optional[str]:
        provider = self.resolved_provider
        if provider == "HF":
            return None

        key_env_var = self._resolved_api_key_env_var
        if key_env_var in os.environ:
            return os.environ[key_env_var]

        if provider == "OPENAI" and self.base_url:
            # Keep compatibility with keyless openai-compatible local servers.
            return "EMPTY"

        LLM._check_key(key_env_var)
        return os.environ[key_env_var]

    @property
    def is_huggingface(self) -> bool:
        return self.resolved_provider == "HF"

    @property
    def inference_kwargs(self) -> dict[str, Any]:
        return {**self.generation_kwargs, **self.model_kwargs}

    @property
    def loader_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.llm_name,
            **self.inference_kwargs,
            **self.client_kwargs,
        }
        if self.base_url is not None and self.resolved_provider == "OPENAI":
            kwargs["base_url"] = self.base_url

        api_key = self.api_key
        if api_key is not None:
            kwargs["api_key"] = api_key
        return kwargs

    @property
    def hf_kwargs(self) -> dict[str, Any]:
        return {
            "model_id": self.llm_name,
            "task": "text-generation",
            "model_kwargs": self.model_kwargs,
            "pipeline_kwargs": self.generation_kwargs,
        }


class LLM(BaseChatModel):
    """Class parsing the model name and arguments to load the correct LangChain model"""

    device_count: ClassVar[int] = 0
    nb_devices: ClassVar[int] = (
        torch.cuda.device_count() if torch.cuda.is_available() else 1
    )

    @staticmethod
    def _check_key(key_env_var: str) -> None:
        if key_env_var not in os.environ:
            raise ValueError(
                "Unable to find the API key. "
                f"Please restart after setting the '{key_env_var}' environment variable."
            )

    @classmethod
    def from_config(cls, config: str | LLMConfig) -> BaseChatModel:
        if isinstance(config, str):
            config = load_config(config, LLMConfig)

        if config.is_huggingface:
            cls.device_count = (cls.device_count + 1) % (
                cls.nb_devices + 1
            )  # rotate devices, +1 for accounting the -1 below
            return ChatHuggingFace(
                llm=HuggingFacePipeline.from_model_id(
                    config.hf_kwargs["model_id"],
                    task=config.hf_kwargs["task"],
                    device=cls.device_count - 1,
                    model_kwargs=config.hf_kwargs["model_kwargs"],
                    pipeline_kwargs=config.hf_kwargs["pipeline_kwargs"],
                )
            )

        loader = loaders[config.resolved_provider]
        return loader(**config.loader_kwargs)
