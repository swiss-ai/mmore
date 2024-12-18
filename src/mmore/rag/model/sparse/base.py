from typing import List
from langchain_milvus.utils.sparse import BaseSparseEmbedding
from .splade import SpladeSparseEmbedding

_sparse_models = {
    'splade': "naver/splade-cocondenser-selfdistil"
}

# TODO: How to we handle corpus based embeddings?
def load_sparse_model(sparse_model_name: str, corpus: List[str] = None) -> BaseSparseEmbedding:
    if sparse_model_name.lower() == 'bm25':
        return NotImplementedError()
        # return BM25SparseEmbedding(corpus)
    else:
        sparse_model_name = _sparse_models.get(sparse_model_name, sparse_model_name)
        return SpladeSparseEmbedding(sparse_model_name)

