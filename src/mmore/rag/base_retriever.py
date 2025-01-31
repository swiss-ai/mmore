from abc import ABC
from dataclasses import dataclass

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

@dataclass
class RetrieverConfig(ABC):
    """Configuration for the retriever."""
    pass
