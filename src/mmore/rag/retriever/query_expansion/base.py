"""
Query Expansion Module for RAG Retriever. 
This module is used to expand the query with similar terms before retrieving documents.
"""

from typing import List, Dict, Any, Tuple, Literal, get_args
from dataclasses import dataclass, field
from src.mmore.utils import load_config
from src.mmore.rag.llm import LLM, LLMConfig

from langchain_core.embeddings import Embeddings

from langchain_milvus.utils.sparse import BaseSparseEmbedding
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from abc import ABC, abstractmethod

import logging
logger = logging.getLogger(__name__)

@dataclass
class BaseQueryExpansionConfig:
    query_expansion_type: Literal["PRF", "LLMsExpansion"]
    llm: LLMConfig = field(default_factory=lambda: LLMConfig(llm_name='gpt-4o'))


class BaseQueryExpansion(ABC):
    """Handles query expansion for RAG retriever."""

    def __init__(self, config: BaseQueryExpansionConfig):
        self.config = config

    @abstractmethod
    def expand_query(self, query: str, collection_name: str, partition_name: str) -> str:
        """
        Expand the query with similar terms.
        """
        pass