from langchain_core.embeddings import Embeddings

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_cohere import CohereEmbeddings
from langchain_mistralai import MistralAIEmbeddings
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_aws import BedrockEmbeddings
from langchain_community.embeddings import FakeEmbeddings

from .multimodal import MultimodalEmbeddings

from dataclasses import dataclass

_OPENAI_MODELS = [
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002"
]

_GOOGLE_MODELS = [
    "textembedding-gecko@001"
]

_COHERE_MODELS = [
    "embed-english-light-v2.0",
    "embed-english-v2.0",
    "embed-multilingual-v2.0"
]

_MISTRAL_MODELS = [
    "mistral-textembedding-7B-v1",
    "mistral-textembedding-13B-v1"
]

_NVIDIA_MODELS = [
    "nvidia-clarity-text-embedding-v1",
    "nvidia-megatron-embedding-530B"
]

_AWS_MODELS = [
    "amazon-titan-embedding-xlarge",
    "amazon-titan-embedding-light"
]


loaders = {
    'OPENAI': OpenAIEmbeddings,
    #'GOOGLE': VertexAIEmbeddings,
    'COHERE': CohereEmbeddings,
    'MISTRAL': MistralAIEmbeddings,
    'NVIDIA': NVIDIAEmbeddings,
    'AWS': BedrockEmbeddings,
    'HF': lambda model, **kwargs: HuggingFaceEmbeddings(model_name=model, model_kwargs={'trust_remote_code': True}, **kwargs),
    'FAKE': lambda **kwargs: FakeEmbeddings(size=2048), # For testing purposes, don't use in production
}

@dataclass
class DenseModelConfig:
    model_name: str
    is_multimodal: bool = False

    @property
    def organization(self) -> str:
        if self.model_name in _OPENAI_MODELS:
            return 'OPENAI'
        elif self.model_name in _GOOGLE_MODELS:
            return 'GOOGLE'
        elif self.model_name in _COHERE_MODELS:
            return 'COHERE'
        elif self.model_name in _MISTRAL_MODELS:
            return 'MISTRAL'
        elif self.model_name in _NVIDIA_MODELS:
            return 'NVIDIA'
        elif self.model_name in _AWS_MODELS:
            return 'AWS'
        elif self.model_name == 'debug':
            return 'FAKE'   # For testing purposes
        else:
            return 'HF'

class DenseModel(Embeddings):
    @classmethod
    def from_config(cls, config: DenseModelConfig) -> Embeddings:
        if config.organization == 'HF' and config.is_multimodal:
            return MultimodalEmbeddings(model_name=config.model_name)
        else:
            return loaders[config.organization](model=config.model_name)