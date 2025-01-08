from .base import BaseQueryExpansion, QueryExpansionConfig

from .PRF import PRF
from .LLMs import LLMs

from mmore.utils import load_config
from src.mmore.rag.retriever import Retriever, RetrieverConfig


__all__ = ['PRF', 'LLMs']

def load_query_expansion(config: RetrieverConfig) -> BaseQueryExpansion:
    if config.query_expansion.query_expansion_type == 'PRF':
        retrieverConfig = config.deepcopy()
        retrieverConfig.query_expansion = None
        return PRF(config.query_expansion, retrieverConfig)
    elif config.query_expansion.query_expansion_type == 'LLMs':
        retrieverConfig = config.deepcopy()
        retrieverConfig.query_expansion = None
        return LLMs(config.query_expansion, retrieverConfig)
    else:
        raise ValueError(f"Unrecognized query expansion type: {config.query_expansion_type}")
  