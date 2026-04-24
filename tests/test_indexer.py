# tests/test_indexer.py

import json
from typing import Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from langchain_milvus.utils.sparse import BaseSparseEmbedding

from mmore.index.indexer import Indexer, IndexerConfig
from pymilvus import MilvusClient
from mmore.rag.model import DenseModelConfig, SparseModelConfig

# Import run_index from the correct package path:
from mmore.run_index import index
from mmore.type import MultimodalSample


# ---------------------------------------------------------------------------
# Helpers shared by integration tests
# ---------------------------------------------------------------------------


class _FakeSparseEmbedding(BaseSparseEmbedding):
    """Deterministic sparse embedder — no model download, runs on CPU."""

    def embed_query(self, query: str) -> Dict[int, float]:
        return {0: 1.0, 1: float(len(query))}

    def embed_documents(self, texts: List[str]) -> List[Dict[int, float]]:
        return [{0: 1.0, i + 1: float(len(t))} for i, t in enumerate(texts)]


@pytest.fixture
def sample_documents():
    return [
        MultimodalSample(
            id="doc-1",
            document_id="doc-1",
            text="Paris is the capital of France.",
            modalities=[],
            metadata={},
        ),
        MultimodalSample(
            id="doc-2",
            document_id="doc-2",
            text="The Eiffel Tower stands 330 metres tall.",
            modalities=[],
            metadata={"author": "Alice"},
        ),
        MultimodalSample(
            id="doc-3",
            document_id="doc-3",
            text="Milvus is an open-source vector database.",
            modalities=[],
            metadata={},
        ),
    ]


@pytest.fixture
def sample_jsonl(tmp_path):
    """Creates a temporary JSONL file with sample documents."""
    path = tmp_path / "sample_docs.jsonl"
    sample_data = [
        {"id": "1", "text": "Document text 1", "modalities": [], "metadata": {}},
        {
            "id": "2",
            "text": "Document text 2",
            "modalities": [],
            "metadata": {"author": "Alice"},
        },
    ]
    with open(path, "w", encoding="utf-8") as f:
        for entry in sample_data:
            f.write(json.dumps(entry) + "\n")
    return path


@patch("mmore.run_index.Indexer.from_documents")
def test_index_invocation(mock_from_documents, sample_jsonl):
    """
    Tests that index function loads the config and calls Indexer.from_documents.
    """
    mock_indexer_config = MagicMock()
    mock_indexer_config.collection_name = "test_collection"
    mock_indexer_config.documents_path = str(sample_jsonl)

    # Patch load_config so it returns the mock config
    with patch("mmore.run_index.load_config", return_value=mock_indexer_config):
        index(config_file="fake_config.yml")
    mock_from_documents.assert_called_once()

    # Confirm the correct collection name is passed
    call_args = mock_from_documents.call_args[1]  # call_args is (args, kwargs)
    assert call_args["collection_name"] == "test_collection"


@patch("mmore.index.indexer.MilvusClient")
@patch("mmore.index.indexer.DenseModel.from_config")
@patch("mmore.index.indexer.SparseModel.from_config")
def test_indexer_integration(
    mock_sparse_model, mock_dense_model, mock_milvus_client, sample_jsonl
):
    """
    Tests the Indexer class with mocked embeddings & Milvus.
    """
    mock_dense_model.return_value.embed_documents.return_value = [
        np.array([0.01, 0.02]),
        np.array([0.03, 0.04]),
    ]
    mock_sparse_model.return_value.embed_documents.return_value = [
        np.array([0, 1]),
        np.array([1, 0]),
    ]

    # Mock Milvus
    client_instance = mock_milvus_client.return_value
    client_instance.has_collection.return_value = False
    client_instance.insert.return_value = {"insert_count": 2}

    # Build IndexerConfig
    dense_cfg = MagicMock()
    sparse_cfg = MagicMock()
    db_cfg = MagicMock()
    test_indexer_config = IndexerConfig(
        dense_model=dense_cfg, sparse_model=sparse_cfg, db=db_cfg
    )

    # Load sample documents
    documents = MultimodalSample.from_jsonl(str(sample_jsonl))

    # Index them
    Indexer.from_documents(
        config=test_indexer_config,
        documents=documents,
        collection_name="test_collection",
        batch_size=2,
    )

    # Verify the client did what we expect
    assert client_instance.create_collection.called, (
        "Should create collection if it does not exist"
    )
    assert client_instance.insert.called, "Should insert documents into Milvus"


@patch("mmore.index.indexer.MilvusClient")
def test_index_documents_error(mock_milvus_client, sample_jsonl):
    """
    Tests that an exception is raised if insertion fails.
    """
    # Mock Milvus
    client_instance = mock_milvus_client.return_value
    client_instance.has_collection.return_value = False
    client_instance.insert.side_effect = Exception("Insertion error")

    # Minimal config
    test_indexer_config = IndexerConfig(
        dense_model=MagicMock(), sparse_model=MagicMock()
    )

    # Patch the embeddings to return arrays
    with (
        patch("mmore.index.indexer.DenseModel.from_config") as mock_dense_model,
        patch("mmore.index.indexer.SparseModel.from_config") as mock_sparse_model,
    ):
        mock_dense_model.return_value.embed_documents.return_value = [
            np.array([0.01, 0.02])
        ]
        mock_sparse_model.return_value.embed_documents.return_value = [np.array([0, 1])]

        indexer = Indexer(
            dense_model_config=test_indexer_config.dense_model,
            sparse_model_config=test_indexer_config.sparse_model,
            client=client_instance,
        )
        docs = MultimodalSample.from_jsonl(str(sample_jsonl))[:1]

        with pytest.raises(Exception, match="Insertion error"):
            indexer.index_documents(docs)


# ---------------------------------------------------------------------------
# Real integration tests — Milvus Lite (local .db) + FakeEmbeddings, no GPU
# ---------------------------------------------------------------------------


@patch("mmore.index.indexer.SparseModel.from_config")
def test_indexer_real_insert(mock_sparse, tmp_path, sample_documents):
    """
    Documents are actually written to a local Milvus Lite .db file.
    No mock for Milvus or the dense model — only SPLADE is replaced to
    avoid a ~500 MB download.
    """
    mock_sparse.return_value = _FakeSparseEmbedding()

    client = MilvusClient(str(tmp_path / "test.db"), enable_sparse=True)
    dense_cfg = DenseModelConfig(model_name="debug")  # FakeEmbeddings(size=2048)
    sparse_cfg = SparseModelConfig(model_name="naver/splade-cocondenser-selfdistil")

    indexer = Indexer(
        dense_model_config=dense_cfg,
        sparse_model_config=sparse_cfg,
        client=client,
    )

    inserted = indexer.index_documents(sample_documents, collection_name="test_col")

    assert inserted == len(sample_documents)
    stats = client.get_collection_stats("test_col")
    assert int(stats["row_count"]) == len(sample_documents)


@patch("mmore.index.indexer.SparseModel.from_config")
def test_indexer_real_insert_preserves_metadata(mock_sparse, tmp_path, sample_documents):
    """
    Metadata fields passed in documents are stored in Milvus dynamic fields
    and can be retrieved after insertion.
    """
    mock_sparse.return_value = _FakeSparseEmbedding()

    client = MilvusClient(str(tmp_path / "test.db"), enable_sparse=True)
    dense_cfg = DenseModelConfig(model_name="debug")
    sparse_cfg = SparseModelConfig(model_name="naver/splade-cocondenser-selfdistil")

    indexer = Indexer(
        dense_model_config=dense_cfg,
        sparse_model_config=sparse_cfg,
        client=client,
    )
    indexer.index_documents(sample_documents, collection_name="test_col")

    # doc-2 has metadata {"author": "Alice"}, check it was stored
    results = client.query(
        collection_name="test_col",
        filter='id == "doc-2"',
        output_fields=["id", "text", "author"],
    )
    assert len(results) == 1
    assert results[0]["author"] == "Alice"
    assert results[0]["text"] == "The Eiffel Tower stands 330 metres tall."


@patch("mmore.index.indexer.SparseModel.from_config")
def test_indexer_real_idempotent_collection(mock_sparse, tmp_path, sample_documents):
    """
    Calling index_documents twice on the same collection name appends
    rather than recreating the collection.
    """
    mock_sparse.return_value = _FakeSparseEmbedding()

    client = MilvusClient(str(tmp_path / "test.db"), enable_sparse=True)
    dense_cfg = DenseModelConfig(model_name="debug")
    sparse_cfg = SparseModelConfig(model_name="naver/splade-cocondenser-selfdistil")

    indexer = Indexer(
        dense_model_config=dense_cfg,
        sparse_model_config=sparse_cfg,
        client=client,
    )

    extra = [
        MultimodalSample(
            id="doc-4",
            document_id="doc-4",
            text="A fourth document.",
            modalities=[],
            metadata={},
        )
    ]

    indexer.index_documents(sample_documents, collection_name="test_col")
    indexer.index_documents(extra, collection_name="test_col")

    stats = client.get_collection_stats("test_col")
    assert int(stats["row_count"]) == len(sample_documents) + len(extra)
