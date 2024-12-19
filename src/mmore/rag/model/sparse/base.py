from typing import List
from langchain_milvus.utils.sparse import BaseSparseEmbedding
from .splade import SpladeSparseEmbedding
from langchain_core.embeddings import Embeddings

from dataclasses import dataclass

_SPLADE_MODELS = [
    "naver/splade-cocondenser-selfdistil"
]

loaders = {
    'SPLADE': SpladeSparseEmbedding
}

_names = {
    'splade': "naver/splade-cocondenser-selfdistil"
}

@dataclass
class SparseModelConfig:
    model_name: str
    is_multimodal: bool = False

    def __post_init__(self):
        if self.model_name.lower() in _names:
            self.model_name = _names.get(self.model_name, self.model_name)

    @property
    def model_type(self) -> str:
        if self.model_name in _SPLADE_MODELS:
            return 'SPLADE'
        else:
            raise NotImplementedError()

class SparseModel(Embeddings):
    @classmethod
    def from_config(cls, config: SparseModelConfig) -> 'SparseModel':
        return loaders.get(config.model_type, SpladeSparseEmbedding)(model_name=config.model_name)