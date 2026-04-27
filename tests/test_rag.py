from typing import Dict, List
from unittest.mock import patch

import pytest
from langchain_community.embeddings import FakeEmbeddings
from langchain_core.documents import Document
from langchain_milvus.utils.sparse import BaseSparseEmbedding
from pymilvus import MilvusClient

from mmore.index.indexer import Indexer
from mmore.rag.llm import LLMConfig
from mmore.rag.model import DenseModelConfig, SparseModelConfig
from mmore.rag.retriever import Retriever
from mmore.type import MultimodalSample


class _FakeSparseEmbedding(BaseSparseEmbedding):
    def embed_query(self, query: str) -> Dict[int, float]:
        return {0: 1.0, 1: float(len(query))}

    def embed_documents(self, texts: List[str]) -> List[Dict[int, float]]:
        return [{0: 1.0, i + 1: float(len(t))} for i, t in enumerate(texts)]


_DOCS = [
    MultimodalSample(id="doc-1", document_id="doc-1", text="Paris is the capital of France.", modalities=[], metadata={}),
    MultimodalSample(id="doc-2", document_id="doc-2", text="The Eiffel Tower stands 330 metres tall.", modalities=[], metadata={}),
]

_COLLECTION = "test_col"


@pytest.fixture(scope="module")
def populated_db(tmp_path_factory):
    db_path = str(tmp_path_factory.mktemp("rag_db") / "test.db")
    with patch("mmore.index.indexer.SparseModel.from_config", return_value=_FakeSparseEmbedding()):
        client = MilvusClient(db_path, enable_sparse=True)
        indexer = Indexer(
            dense_model_config=DenseModelConfig(model_name="debug"),
            sparse_model_config=SparseModelConfig(model_name="naver/splade-cocondenser-selfdistil"),
            client=client,
        )
        indexer.index_documents(_DOCS, collection_name=_COLLECTION)
    return db_path


def test_retriever_initialization_real(tmp_path):
    """Retriever initializes correctly against a real (empty) Milvus Lite client."""
    client = MilvusClient(str(tmp_path / "test.db"), enable_sparse=True)
    retriever = Retriever(
        dense_model=FakeEmbeddings(size=2048),
        sparse_model=_FakeSparseEmbedding(),
        client=client,
        hybrid_search_weight=0.5,
        k=2,
        use_web=False,
        reranker_model=None,
        reranker_tokenizer=None,
    )
    assert isinstance(retriever, Retriever)


@pytest.mark.gpu
def test_rerank_real(populated_db):
    """Reranking with bge-reranker-base on a real GPU runner."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
    model = AutoModelForSequenceClassification.from_pretrained(
        "BAAI/bge-reranker-base"
    ).to(device)

    client = MilvusClient(populated_db)
    retriever = Retriever(
        dense_model=FakeEmbeddings(size=2048),
        sparse_model=_FakeSparseEmbedding(),
        client=client,
        hybrid_search_weight=0.5,
        k=2,
        use_web=False,
        reranker_model=model,
        reranker_tokenizer=tokenizer,
    )

    docs = [
        Document(page_content="Paris is the capital of France.", metadata={"rank": 1, "similarity": 0.9, "id": "1", "page_numbers": [], "paragraph_numbers": []}),
        Document(page_content="Milvus is an open-source vector database.", metadata={"rank": 2, "similarity": 0.8, "id": "2", "page_numbers": [], "paragraph_numbers": []}),
    ]

    reranked = retriever.rerank("France capital", docs)
    assert len(reranked) == 2
    assert all(isinstance(d, Document) for d in reranked)


def test_llm_config_generation_kwargs():
    """LLMConfig.generation_kwargs returns correct parameter names per provider."""
    mistral_config = LLMConfig(llm_name="mistral-large-3", max_new_tokens=1200)
    assert mistral_config.provider == "MISTRAL"
    assert mistral_config.generation_kwargs["max_tokens"] == 1200

    anthropic_config = LLMConfig(llm_name="claude-sonnet-4-6", max_new_tokens=1500)
    assert anthropic_config.provider == "ANTHROPIC"
    assert anthropic_config.generation_kwargs["max_tokens"] == 1500

    cohere_config = LLMConfig(llm_name="command-r-08-2024", max_new_tokens=1000)
    assert cohere_config.provider == "COHERE"
    assert cohere_config.generation_kwargs["max_tokens"] == 1000

    hf_config = LLMConfig(llm_name="gpt2", max_new_tokens=800)
    assert hf_config.provider == "HF"
    assert hf_config.generation_kwargs["max_new_tokens"] == 800

    openai_config = LLMConfig(llm_name="gpt-4o", max_new_tokens=2000)
    assert openai_config.provider == "OPENAI"
    assert openai_config.generation_kwargs["max_completion_tokens"] == 2000
