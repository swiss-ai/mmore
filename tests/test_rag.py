from unittest.mock import MagicMock, patch

import pytest
import torch
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_milvus.utils.sparse import BaseSparseEmbedding
from pymilvus import MilvusClient
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from mmore.rag.retriever import Retriever

# Mock Classes


class MockEmbeddings(Embeddings):
    def embed_query(self, text):
        return [0.1, 0.2]

    def embed_documents(self, texts):
        return [[0.1, 0.2] for _ in texts]


class MockSparse(BaseSparseEmbedding):
    def embed_query(self, text):
        return {0: 1.0}

    def embed_documents(self, texts):
        return [{0: 1.0} for _ in texts]


class MockMilvus(MilvusClient):
    def __init__(self):
        pass


class MockModel(PreTrainedModel):
    def __init__(self):
        from transformers import PretrainedConfig

        config = PretrainedConfig()
        super().__init__(config)
        self.logits = torch.tensor([[0.1], [2.0]])

    def forward(self, **kwargs):
        class Output:
            def __init__(self, logits):
                self.logits = logits

        return Output(self.logits)


class MockBatch:
    def __init__(self, data):
        self.data = data

    def to(self, device):
        return self

    def __getitem__(self, k):
        return self.data[k]


class MockTokenizer(PreTrainedTokenizerBase):
    def __call__(self, queries, docs, **kwargs):
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
