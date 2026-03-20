import os
import re
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any, ClassVar, Optional
from urllib.parse import urlparse

import torch
from langchain_anthropic import ChatAnthropic
from langchain_cohere import ChatCohere
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI

from ..utils import load_config

logger = getLogger(__name__)

loaders: dict[str, Any] = {
    "OPENAI": ChatOpenAI,
    "ANTHROPIC": ChatAnthropic,
    "MISTRAL": ChatMistralAI,
    "COHERE": ChatCohere,
    "HF": ChatHuggingFace,
}

_PROVIDER_MODEL_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "OPENAI": (
        re.compile(r"(^|[/:_-])(gpt|chatgpt)([/:_.-]|$)", re.IGNORECASE),
        re.compile(r"^o[134]\b", re.IGNORECASE),  # o1/o3/o4 family
        re.compile(r"(^|[/:_-])openai([/:_.-]|$)", re.IGNORECASE),
    ),
    "ANTHROPIC": (
        re.compile(r"(^|[/:_-])claude([/:_.-]|$)", re.IGNORECASE),
        re.compile(r"(^|[/:_-])anthropic([/:_.-]|$)", re.IGNORECASE),
    ),
    "MISTRAL": (
        re.compile(
            r"(^|[/:_-])(mistral|mixtral|ministral|pixtral|codestral)([/:_.-]|$)",
            re.IGNORECASE,
        ),
    ),
    "COHERE": (
        re.compile(r"(^|[/:_-])(command|cohere|c4ai)([/:_.-]|$)", re.IGNORECASE),
    ),
}

_PROVIDER_BASE_URL_HINTS: dict[str, tuple[str, ...]] = {
    "OPENAI": ("openai",),
    "ANTHROPIC": ("anthropic",),
    "MISTRAL": ("mistral",),
    "COHERE": ("cohere",),
}


def _normalize_value(value: Optional[str]) -> Optional[str]:
    return value.upper() if value is not None else None


def _infer_organization_from_hints(
    llm_name: str, base_url: Optional[str]
) -> Optional[str]:
    for organization, patterns in _PROVIDER_MODEL_PATTERNS.items():
        if any(pattern.search(llm_name) for pattern in patterns):
            return organization

    if base_url:
        hostname = (urlparse(base_url).hostname or "").lower()
        for organization, hints in _PROVIDER_BASE_URL_HINTS.items():
            if any(hint in hostname for hint in hints):
                return organization

    return None


@dataclass
class LLMConfig:
    llm_name: str
    organization: Optional[str] = None
    base_url: Optional[str] = None
    # Optional override (e.g., "SWISSAI_API_KEY") for API-key lookup.
    api_key_env_var: Optional[str] = None
    max_new_tokens: Optional[int] = None
    temperature: float = 0.7
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    client_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.organization = _normalize_value(self.organization)

        if self.organization is None:
            self.organization = _infer_organization_from_hints(
                self.llm_name, self.base_url
            )

        if self.organization is None:
            self.organization = "OPENAI" if self.base_url is not None else "HF"
            logger.warning(
                "No organization configured. Defaulting to organization='%s'.",
                self.organization,
            )

        self._validate_organization()

    def _validate_organization(self) -> None:
        if self.organization in loaders:
            return

        supported = ", ".join(sorted(loaders))
        raise ValueError(
            f"Unsupported organization '{self.organization}'. Supported organizations: {supported}."
        )

    @property
    def generation_kwargs(self) -> dict[str, Any]:
        max_token_key = (
            "max_new_tokens"
            if self.resolved_organization in {"ANTHROPIC", "MISTRAL", "COHERE", "HF"}
            else "max_completion_tokens"
        )
        return {"temperature": self.temperature, max_token_key: self.max_new_tokens}

    @property
    def resolved_organization(self) -> str:
        if self.organization is None:
            raise ValueError(
                "Organization resolution failed; organization should never be None."
            )
        return self.organization

    @property
    def api_key(self) -> Optional[str]:
        organization = self.resolved_organization
        if organization == "HF":
            return None

        key_env_var = self.api_key_env_var or f"{self.resolved_organization}_API_KEY"
        if key_env_var in os.environ:
            return os.environ[key_env_var]

        if organization == "OPENAI" and self.base_url:
            # Keep compatibility with keyless openai-compatible local servers.
            return "EMPTY"

        LLM._check_key(key_env_var)
        return os.environ[key_env_var]

    @property
    def is_huggingface(self) -> bool:
        return self.resolved_organization == "HF"

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
        if self.base_url is not None and self.resolved_organization == "OPENAI":
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

        loader = loaders[config.resolved_organization]
        return loader(**config.loader_kwargs)
