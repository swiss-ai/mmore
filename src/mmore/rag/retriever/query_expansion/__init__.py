from .base import BaseQueryExpansion, BaseQueryExpansionConfig

# from .PRF import PRF
from .LLMs import LLMsExpansion

from mmore.utils import load_config
# from src.mmore.rag.retriever.retriever import Retriever, RetrieverConfig


__all__ = ['PRF', 'LLMsExpansion']

def load_query_expansion(config: BaseQueryExpansionConfig) -> BaseQueryExpansion:
    # if config.query_expansion.query_expansion_type == 'PRF':
        # retrieverConfig = config.deepcopy()
        # retrieverConfig.query_expansion = None
        # return PRF(config.query_expansion, Retriever.from_config(retrieverConfig))
    if config.query_expansion_type == 'LLMsExpansion':
        return LLMsExpansion(config)
    else:
        raise ValueError(f"Unrecognized query expansion type: {config.query_expansion_type}")
  