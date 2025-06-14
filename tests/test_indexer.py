# tests/test_indexer.py

import json
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.mmore.index.indexer import Indexer, IndexerConfig

# Import run_index from the correct package path:
from src.mmore.run_index import index
from src.mmore.type import MultimodalSample


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


@patch("src.mmore.run_index.Indexer.from_documents")
def test_index_invocation(mock_from_documents, sample_jsonl):
    """
    Tests that index function loads the config and calls Indexer.from_documents.
    """
    mock_indexer_config = MagicMock()
    mock_indexer_config.collection_name = "test_collection"
    mock_indexer_config.documents_path = str(sample_jsonl)

    # Patch load_config so it returns the mock config
    with patch("src.mmore.run_index.load_config", return_value=mock_indexer_config):
        index(config_file="fake_config.yml")
    mock_from_documents.assert_called_once()

    # Confirm the correct collection name is passed
    call_args = mock_from_documents.call_args[1]  # call_args is (args, kwargs)
    assert call_args["collection_name"] == "test_collection"


@patch("src.mmore.index.indexer.MilvusClient")
@patch("src.mmore.index.indexer.DenseModel.from_config")
@patch("src.mmore.index.indexer.SparseModel.from_config")
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


@patch("src.mmore.index.indexer.MilvusClient")
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
        patch("src.mmore.index.indexer.DenseModel.from_config") as mock_dense_model,
        patch("src.mmore.index.indexer.SparseModel.from_config") as mock_sparse_model,
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
