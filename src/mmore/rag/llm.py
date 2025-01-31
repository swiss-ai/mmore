from langchain_core.language_models.chat_models import BaseChatModel

# HF Models
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

# Proprietary Models
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from langchain_cohere import ChatCohere

from mmore.index.implementations.graphrag.vllm_model import vLLMWrapper

from dataclasses import dataclass, field

import os
from getpass import getpass

from ..utils import load_config

_OPENAI_MODELS = ['gpt-4', 'gpt-4-turbo', 'gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo', 'davinci', 'curie', 'babbage', 'ada']
_ANTHROPIC_MODELS = ['claude-1', 'claude-1.3', 'claude-2', 'claude-instant-1', 'claude-instant-1.1', 'claude-instant-1.2']
_MISTRAL_MODELS = ['mistral-7b', 'mistral-7b-instruct', 'mistral-7b-chat']
_COHERE_MODELS = ['command', 'command-light', 'command-nightly', 'summarize', 'embed-english-v2.0']

loaders = {
    "OPENAI": ChatOpenAI,
    "ANTHROPIC": ChatAnthropic,
    "MISTRAL": ChatMistralAI,
    "COHERE": ChatCohere,
    "HF": ChatHuggingFace,
    "VLLM": vLLMWrapper
}

# TODO (@paultltc): Add generation kwargs
@dataclass
class LLMConfig:
    llm_name: str
    base_url: str = None
    organization: str = None
    max_new_tokens: int = None
    temperature: float = 0.7

    def __post_init__(self):
        self.organization = self.organization or (
            "OPENAI" if self.llm_name in _OPENAI_MODELS else
            "ANTHROPIC" if self.llm_name in _ANTHROPIC_MODELS else
            "MISTRAL" if self.llm_name in _MISTRAL_MODELS else
            "COHERE" if self.llm_name in _COHERE_MODELS else
            "HF" if self.base_url is None else
            None
        )

        if not self.organization is None:
            self.organization = self.organization.upper()

    @property
    def generation_kwargs(self):
        max_token_key = "max_new_tokens" if (self.organization in ["ANTHROPIC", "MISTRAL", "COHERE", "HF", "VLLM"]) \
                         else "max_completion_tokens"
        return {
            "temperature": self.temperature,
            max_token_key: self.max_new_tokens
        }

    @property
    def api_key(self):
        if self.organization:
            LLM._check_key(self.organization)
            return os.environ[f"{self.organization}_API_KEY"]
        else:
            return "EMPTY"
    
class LLM():
    """Class parsing the model name and arguments to load the correct LangChain model"""
    @staticmethod
    def _check_key(org):
        if f"{org}_API_KEY" not in os.environ:
            print(f"Enter your {org} API key:")
            os.environ[f"{org}_API_KEY"] = getpass()

    @classmethod
    def from_config(cls, config: str | LLMConfig, **kwargs):
        if isinstance(config, str):
            config = load_config(config, LLMConfig)

        if config.organization == "HF":
            chat_models = ChatHuggingFace(llm=HuggingFacePipeline.from_model_id(
                config.llm_name,
                task="text-generation",
                device_map="auto",
                pipeline_kwargs=config.generation_kwargs
            ))
            chat_models.llm.pipeline.tokenizer.pad_token_id = chat_models.llm.pipeline.tokenizer.eos_token_id
            return chat_models
        elif config.organization == "VLLM":
            return vLLMWrapper(
                model_name=config.llm_name
            )
        else:
            loader = loaders.get(config.organization, ChatOpenAI)
            return loader(
                model=config.llm_name, 
                base_url=config.base_url,
                api_key=config.api_key,
                **config.generation_kwargs
            )