from typing import Dict, List, Any, Optional, Literal

# ----------------------------- EMBEDDING MODELS ----------------------------- #

from langchain_core.embeddings import Embeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from .multimodal_model import MultimodalEmbeddings
from .splade import SpladeSparseEmbedding

dense_models_ = {
    'llama32vision': 'meta-llama/Llama-3.2-11B-Vision'
}

sparse_models_ = {
    'splade': "naver/splade-cocondenser-selfdistil"
}


def load_dense_model(dense_model_name: str) -> Embeddings:
    if dense_model_name == 'meta-llama/Llama-3.2-11B-Vision':
        return MultimodalEmbeddings(model_name=dense_model_name)
    else:
        return HuggingFaceEmbeddings(model_name=dense_model_name)


# TODO: How to we handle corpus based embeddings?
def load_sparse_model(sparse_model_name: str, corpus: List[str] = None):
    if sparse_model_name.lower() == 'bm25':
        return NotImplementedError()
        # return BM25SparseEmbedding(corpus)
    else:
        sparse_model_name = sparse_models_.get(sparse_model_name, sparse_model_name)
        return SpladeSparseEmbedding(sparse_model_name)
