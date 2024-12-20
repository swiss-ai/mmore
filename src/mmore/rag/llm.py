from langchain_core.language_models.chat_models import BaseChatModel

# HF Models
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

# Proprietary Models
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from langchain_cohere import ChatCohere

from dataclasses import dataclass, field

import os
from getpass import getpass

from ..utils import load_config

_OPENAI_MODELS = ['gpt-4', 'gpt-4-turbo', 'gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo', 'davinci', 'curie', 'babbage', 'ada']
_ANTHROPIC_MODELS = ['claude-1', 'claude-1.3', 'claude-2', 'claude-instant-1', 'claude-instant-1.1', 'claude-instant-1.2']
_MISTRAL_MODELS = ['mistral-7b', 'mistral-7b-instruct', 'mistral-7b-chat']
_COHERE_MODELS = ['command', 'command-light', 'command-nightly', 'summarize', 'embed-english-v2.0']


# TODO (@paultltc): Add generation kwargs
@dataclass
class LLMConfig:
    llm_name: str
    max_new_tokens: int = 100
    temperature: float = 0.7

    @property
    def generation_kwargs(self):
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }


class LLM(BaseChatModel):
    """Class parsing the model name and arguments to load the correct LangChain model"""

    @staticmethod
    def _check_key(org):
        if f"{org}_API_KEY" not in os.environ:
            print(f"Enter your {org} API key:")
            os.environ[f"{org}_API_KEY"] = getpass()

    @classmethod
    def from_config(cls, config: str | LLMConfig):
        if isinstance(config, str):
            config = load_config(config, LLMConfig)

        # TODO (@paultltc): [FEATURE] Handle all API based models (e.g ChatGPT) -> https://python.langchain.com/docs/how_to/qa_citations/
        if config.llm_name in _OPENAI_MODELS:
            LLM._check_key('OPENAI')
            return ChatOpenAI(model=config.llm_name, temperature=config.generation_kwargs["temperature"])
        elif config.llm_name in _ANTHROPIC_MODELS:
            LLM._check_key('ANTHROPIC')
            return ChatAnthropic(model=config.llm_name, **config.generation_kwargs)
        elif config.llm_name in _MISTRAL_MODELS:
            LLM._check_key('MISTRAL')
            return ChatMistralAI(model=config.llm_name, **config.generation_kwargs)
        elif config.llm_name in _COHERE_MODELS:
            LLM._check_key('COHERE')
            return ChatCohere(model=config.llm_name, **config.generation_kwargs)
        else:
            return ChatHuggingFace(llm=HuggingFacePipeline.from_model_id(
                config.llm_name,
                task="text-generation",
                device_map="auto",
                pipeline_kwargs=config.generation_kwargs
            ))
