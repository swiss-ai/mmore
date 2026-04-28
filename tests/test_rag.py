from langchain_community.embeddings import FakeEmbeddings
from pymilvus import MilvusClient
from unittest.mock import patch

from conftest import SAMPLE_DOCS, FakeSparseEmbedding
from mmore.index.indexer import Indexer
from mmore.rag.llm import LLMConfig
from mmore.rag.model import DenseModelConfig, SparseModelConfig
from mmore.rag.retriever import Retriever

_COLLECTION = "test_col"


def test_retriever_initialization_real(tmp_path):
    """Retriever initializes correctly against a real (empty) Milvus Lite client."""
    client = MilvusClient(str(tmp_path / "test.db"), enable_sparse=True)
    retriever = Retriever(
        dense_model=FakeEmbeddings(size=2048),
        sparse_model=FakeSparseEmbedding(),
        client=client,
        hybrid_search_weight=0.5,
        k=2,
        use_web=False,
        reranker_model=None,
        reranker_tokenizer=None,
    )
    assert isinstance(retriever, Retriever)


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
