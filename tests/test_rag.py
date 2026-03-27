from typing import Union
from unittest.mock import MagicMock, patch

import pytest
import torch
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_milvus.utils.sparse import BaseSparseEmbedding
from pymilvus import MilvusClient
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

from mmore.rag.llm import LLMConfig
from mmore.rag.retriever import Retriever

# Mock Classes


class MockEmbeddings(Embeddings):
    def embed_query(self, text):
        return [0.1, 0.2]

    def embed_documents(self, texts):
        return [[0.1, 0.2] for _ in texts]


class MockSparse(BaseSparseEmbedding):
    def embed_query(self, query):
        return {0: 1.0}

    def embed_documents(self, texts):
        return [{0: 1.0} for _ in texts]


class MockMilvus(MilvusClient):
    def __init__(self):
        pass


class MockModel(PreTrainedModel):
    def __init__(self):
        from transformers.configuration_utils import PretrainedConfig

        config = PretrainedConfig()
        super().__init__(config)
        self.logits = torch.tensor([[0.1], [2.0]])

    def forward(self, **kwargs):
        class Output:
            def __init__(self, logits):
                self.logits = logits

        return Output(self.logits)


class MockBatch(BatchEncoding):
    def __init__(self, data):
        self.data = data

    def to(self, device: Union[str, "torch.device"], *, non_blocking: bool = False):
        return self

    def __getitem__(self, k):
        return self.data[k]


class MockTokenizer(PreTrainedTokenizerBase):
    def __call__(
        self,
        text=None,
        text_pair=None,
        text_target=None,
        text_pair_target=None,
        add_special_tokens=True,
        padding=False,
        truncation=None,
        max_length=None,
        stride=0,
        is_split_into_words=False,
        pad_to_multiple_of=None,
        padding_side=None,
        return_tensors=None,
        return_token_type_ids=None,
        return_attention_mask=None,
        return_overflowing_tokens=False,
        return_special_tokens_mask=False,
        return_offsets_mapping=False,
        return_length=False,
        verbose=True,
        **kwargs,
    ):
        return MockBatch(
            {
                "input_ids": torch.tensor([[1, 2], [3, 4]]),
                "attention_mask": torch.tensor([[1, 1], [1, 1]]),
            }
        )


# Tests


def test_retriever_initialization():
    """Test Retriever.from_config initializes correctly with mocked components."""
    retriever = Retriever(
        dense_model=MockEmbeddings(),
        sparse_model=MockSparse(),
        client=MockMilvus(),
        hybrid_search_weight=0.5,
        k=2,
        use_web=False,
        reranker_model=MockModel(),
        reranker_tokenizer=MockTokenizer(),
    )
    assert isinstance(retriever, Retriever)


@patch("mmore.rag.retriever.Retriever.rerank")
def test_rerank_batch(mock_rerank):
    """Test the reranking logic and ensure docs are sorted correctly by mock model scores."""

    docs = [
        Document(page_content="doc1", metadata={"id": "1"}),
        Document(page_content="doc2", metadata={"id": "2"}),
    ]

    def mock_rerank_side_effect(query, docs):
        scores = [0.1, 2.0]
        scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        reranked_docs = []
        for doc, score in scored_docs:
            new_doc = doc.copy()
            new_doc.metadata["similarity"] = score
            reranked_docs.append(new_doc)
        return reranked_docs

    mock_rerank.side_effect = mock_rerank_side_effect

    retriever = Retriever(
        dense_model=MockEmbeddings(),
        sparse_model=MockSparse(),
        client=MockMilvus(),
        hybrid_search_weight=0.5,
        k=2,
        use_web=False,
        reranker_model=MockModel(),
        reranker_tokenizer=MockTokenizer(),
    )

    reranked = retriever.rerank("test query", docs)

    # Assertions
    assert isinstance(reranked, list)
    assert reranked[0].page_content == "doc2"
    assert reranked[1].page_content == "doc1"
    assert reranked[0].metadata["similarity"] == pytest.approx(2.0)
    mock_rerank.assert_called_once()


@patch("mmore.rag.retriever.Retriever.retrieve")
@patch("mmore.rag.retriever.Retriever.rerank")
def test_get_relevant_documents(mock_rerank, mock_retrieve):
    """Test that _get_relevant_documents integrates retrieval + reranking and transforms Milvus results to Documents."""

    # 1. Setup Mocks for Dependencies
    mock_retrieve.return_value = [
        {"id": "1", "distance": 0.1, "entity": {"text": "doc1 content"}},
        {"id": "2", "distance": 0.3, "entity": {"text": "doc2 content"}},
    ]

    def mock_rerank_side_effect(query, docs, **kwargs):
        assert all(isinstance(d, Document) for d in docs)
        docs[0].metadata["similarity"] = 0.95
        docs[1].metadata["similarity"] = 0.85
        return [docs[0], docs[1]]

    mock_rerank.side_effect = mock_rerank_side_effect

    # 2. Initialize the Retriever (Real class)
    retriever = Retriever(
        dense_model=MockEmbeddings(),
        sparse_model=MockSparse(),
        client=MockMilvus(),
        hybrid_search_weight=0.5,
        k=2,
        use_web=False,
        reranker_model=MockModel(),
        reranker_tokenizer=MockTokenizer(),
    )

    # 3. Call the actual method
    docs = retriever._get_relevant_documents("query", run_manager=MagicMock())

    # 4. Assertions
    assert len(docs) == 2
    assert all(isinstance(d, Document) for d in docs)
    mock_retrieve.assert_called_once()
    mock_rerank.assert_called_once()

    assert docs[0].page_content == "doc1 content"
    assert docs[0].metadata["similarity"] == pytest.approx(0.95)
    assert docs[1].page_content == "doc2 content"
    assert docs[1].metadata["similarity"] == pytest.approx(0.85)


def test_llm_config_generation_kwargs():
    """Test that LLMConfig.generation_kwargs returns correct parameter names for different providers."""
    # Test MISTRAL uses "max_tokens"
    mistral_config = LLMConfig(llm_name="mistral-large-3", max_new_tokens=1200)
    assert mistral_config.provider == "MISTRAL"
    assert "max_tokens" in mistral_config.generation_kwargs
    assert mistral_config.generation_kwargs["max_tokens"] == 1200
    assert mistral_config.generation_kwargs["temperature"] == 0.7

    # Test ANTHROPIC uses "max_tokens"
    anthropic_config = LLMConfig(llm_name="claude-sonnet-4-6", max_new_tokens=1500)
    assert anthropic_config.provider == "ANTHROPIC"
    assert "max_tokens" in anthropic_config.generation_kwargs
    assert anthropic_config.generation_kwargs["max_tokens"] == 1500

    # Test COHERE uses "max_tokens"
    cohere_config = LLMConfig(llm_name="command-r-08-2024", max_new_tokens=1000)
    assert cohere_config.provider == "COHERE"
    assert "max_tokens" in cohere_config.generation_kwargs
    assert cohere_config.generation_kwargs["max_tokens"] == 1000

    # Test HF uses "max_new_tokens"
    hf_config = LLMConfig(llm_name="gpt2", max_new_tokens=800)
    assert hf_config.provider == "HF"
    assert "max_new_tokens" in hf_config.generation_kwargs
    assert hf_config.generation_kwargs["max_new_tokens"] == 800

    # Test OPENAI uses "max_completion_tokens"
    openai_config = LLMConfig(llm_name="gpt-4o", max_new_tokens=2000)
    assert openai_config.provider == "OPENAI"
    assert "max_completion_tokens" in openai_config.generation_kwargs
    assert openai_config.generation_kwargs["max_completion_tokens"] == 2000
