from langchain_core.embeddings import Embeddings

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_cohere import CohereEmbeddings
from langchain_mistralai import MistralAIEmbeddings
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_aws import BedrockEmbeddings
from langchain_community.embeddings import FakeEmbeddings

from .multimodal import MultimodalEmbeddings

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
    'GOOGLE': VertexAIEmbeddings,
    'COHERE': CohereEmbeddings,
    'MISTRAL': MistralAIEmbeddings,
    'NVIDIA': NVIDIAEmbeddings,
    'AWS': BedrockEmbeddings,
    'HF': lambda model, **kwargs: HuggingFaceEmbeddings(model_name=model, **kwargs),
    'FAKE': lambda **kwargs: FakeEmbeddings(size=2048), # For testing purposes, don't use in production
}

def organization(dense_model_name: str) -> str:
    if dense_model_name in _OPENAI_MODELS:
        return 'OPENAI'
    elif dense_model_name in _GOOGLE_MODELS:
        return 'GOOGLE'
    elif dense_model_name in _COHERE_MODELS:
        return 'COHERE'
    elif dense_model_name in _MISTRAL_MODELS:
        return 'MISTRAL'
    elif dense_model_name in _NVIDIA_MODELS:
        return 'NVIDIA'
    elif dense_model_name in _AWS_MODELS:
        return 'AWS'
    elif dense_model_name == 'debug':
        return 'FAKE'   # For testing purposes
    else:
        return 'HF'

def load_dense_model(dense_model_name: str) -> Embeddings:
    if dense_model_name == 'meta-llama/Llama-3.2-11B-Vision':
        return MultimodalEmbeddings(model_name=dense_model_name)
    return loaders[organization(dense_model_name)](model=dense_model_name)