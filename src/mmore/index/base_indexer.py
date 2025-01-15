from abc import ABC, abstractmethod
from typing import List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

from mmore.types.type import MultimodalSample

@dataclass
class BaseIndexerConfig(ABC):
    """Base configuration class that all indexer configs should inherit from"""
    pass

class BaseIndexer(ABC):
    """Abstract base class for all indexers in the system"""
    
    @abstractmethod
    def index_documents(
        self,
        documents: List[MultimodalSample],
        collection_name: Optional[str] = None,
        partition_name: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Index a list of documents
        
        Args:
            documents: List of documents to index
            collection_name: Optional name for the collection/index
            partition_name: Optional partition name within the collection
            **kwargs: Additional implementation-specific parameters
            
        Returns:
            Implementation specific return value (e.g. number of documents indexed)
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_config(cls, config: str | BaseIndexerConfig, **kwargs) -> "BaseIndexer":
        """Create an indexer instance from a config file
        
        Args:
            config_path: Path to the configuration file
            **kwargs: Additional implementation-specific parameters
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_documents(
        cls,
        config: str | BaseIndexerConfig,
        documents: List[MultimodalSample],
        collection_name: Optional[str] = None,
        partition_name: Optional[str] = None,
        **kwargs
    ):
        """Create and run indexer directly from documents
        
        Args:
            config_path: Path to the configuration file
            documents: List of documents to index
            collection_name: Optional name for the collection/index
            partition_name: Optional partition name within the collection
            **kwargs: Additional implementation-specific parameters
        """
        indexer = cls.from_config(config, **kwargs)
        indexer.index_documents(
            documents=documents,
            collection_name=collection_name,
            partition_name=partition_name,
            **kwargs
        )
        return indexer