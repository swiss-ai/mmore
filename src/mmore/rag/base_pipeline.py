from abc import ABC, abstractmethod
from typing import Dict, List, Any, Union, Optional
from dataclasses import dataclass
from pathlib import Path

from langchain_core.language_models import LanguageModelLike

@dataclass
class BaseRAGConfig(ABC):
    """Base configuration for RAG pipelines"""
    pass

class BaseRAGPipeline(ABC):
    """Abstract base class for RAG pipelines"""
    
    llm: LanguageModelLike
    config: BaseRAGConfig
    
    
    def __call__(self, 
                 queries: Union[Dict[str, Any], List[Dict[str, Any]]], 
                 return_dict: bool = False) -> List[Dict[str, Any]]:
        """Process one or multiple queries
        
        Args:
            queries: Single query dict or list of query dicts
            return_dict: Whether to return full result dictionaries or just answers
            
        Returns:
            List of results, either as full dicts or just answer strings
        """
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, config: str | BaseRAGConfig, **kwargs) -> "BaseRAGPipeline":
        """Create pipeline instance from config file
        
        Args:
            config_path: Path to configuration file
            llm: Optional pre-configured language model
            kwargs: Additional keyword arguments
        """
        pass
