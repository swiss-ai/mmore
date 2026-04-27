# tests/test_indexer.py

import json
from typing import Dict, List
from unittest.mock import patch

import pytest
import yaml
from langchain_milvus.utils.sparse import BaseSparseEmbedding
from pymilvus import MilvusClient

from mmore.index.indexer import Indexer, IndexerConfig
from mmore.rag.model import DenseModelConfig, SparseModelConfig
from mmore.run_index import index
from mmore.type import MultimodalSample


# ---------------------------------------------------------------------------
# Shared helpers
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
    path = tmp_path / "sample_docs.jsonl"
    sample_data = [
        {"id": "1", "text": "Document text 1", "modalities": [], "metadata": {}},
        {"id": "2", "text": "Document text 2", "modalities": [], "metadata": {"author": "Alice"}},
    ]
    with open(path, "w", encoding="utf-8") as f:
        for entry in sample_data:
            f.write(json.dumps(entry) + "\n")
    return path


# ---------------------------------------------------------------------------
# Real integration tests — Milvus Lite (local .db) + FakeEmbeddings, no GPU
# ---------------------------------------------------------------------------


@patch("mmore.index.indexer.SparseModel.from_config")
def test_indexer_real_insert(mock_sparse, tmp_path, sample_documents):
    """Documents are actually written to a local Milvus Lite .db file."""
    mock_sparse.return_value = _FakeSparseEmbedding()

    client = MilvusClient(str(tmp_path / "test.db"), enable_sparse=True)
    indexer = Indexer(
        dense_model_config=DenseModelConfig(model_name="debug"),
        sparse_model_config=SparseModelConfig(model_name="naver/splade-cocondenser-selfdistil"),
        client=client,
    )

    inserted = indexer.index_documents(sample_documents, collection_name="test_col")

    assert inserted == len(sample_documents)
    stats = client.get_collection_stats("test_col")
    assert int(stats["row_count"]) == len(sample_documents)


@patch("mmore.index.indexer.SparseModel.from_config")
def test_indexer_real_insert_preserves_metadata(mock_sparse, tmp_path, sample_documents):
    """Metadata fields are stored as Milvus dynamic fields and retrievable."""
    mock_sparse.return_value = _FakeSparseEmbedding()

    client = MilvusClient(str(tmp_path / "test.db"), enable_sparse=True)
    indexer = Indexer(
        dense_model_config=DenseModelConfig(model_name="debug"),
        sparse_model_config=SparseModelConfig(model_name="naver/splade-cocondenser-selfdistil"),
        client=client,
    )
    indexer.index_documents(sample_documents, collection_name="test_col")

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
    """Calling index_documents twice on the same collection appends rather than recreating."""
    mock_sparse.return_value = _FakeSparseEmbedding()

    client = MilvusClient(str(tmp_path / "test.db"), enable_sparse=True)
    indexer = Indexer(
        dense_model_config=DenseModelConfig(model_name="debug"),
        sparse_model_config=SparseModelConfig(model_name="naver/splade-cocondenser-selfdistil"),
        client=client,
    )

    extra = [
        MultimodalSample(
            id="doc-4", document_id="doc-4", text="A fourth document.",
            modalities=[], metadata={},
        )
    ]

    indexer.index_documents(sample_documents, collection_name="test_col")
    indexer.index_documents(extra, collection_name="test_col")

    stats = client.get_collection_stats("test_col")
    assert int(stats["row_count"]) == len(sample_documents) + len(extra)


@patch("mmore.index.indexer.SparseModel.from_config")
def test_run_index_real(mock_sparse, tmp_path, sample_jsonl):
    """run_index.index() reads a real YAML config and populates a real Milvus Lite DB."""
    mock_sparse.return_value = _FakeSparseEmbedding()

    db_path = str(tmp_path / "test.db")
    config = {
        "indexer": {
            "dense_model": {"model_name": "debug", "is_multimodal": False},
            "sparse_model": {
                "model_name": "naver/splade-cocondenser-selfdistil",
                "is_multimodal": False,
            },
            "db": {"uri": db_path, "name": "my_db"},
        },
        "collection_name": "test_col",
        "documents_path": str(sample_jsonl),
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    index(config_file=str(config_path))

    client = MilvusClient(db_path)
    stats = client.get_collection_stats("test_col")
    assert int(stats["row_count"]) == 2  # sample_jsonl has 2 documents


@patch("mmore.index.indexer.SparseModel.from_config")
def test_indexer_error_on_missing_collection(mock_sparse, tmp_path):
    """Inserting directly to a non-existent collection raises an exception."""
    mock_sparse.return_value = _FakeSparseEmbedding()

    client = MilvusClient(str(tmp_path / "test.db"), enable_sparse=True)
    indexer = Indexer(
        dense_model_config=DenseModelConfig(model_name="debug"),
        sparse_model_config=SparseModelConfig(model_name="naver/splade-cocondenser-selfdistil"),
        client=client,
    )

    doc = MultimodalSample(
        id="x", document_id="x", text="test", modalities=[], metadata={}
    )
    with pytest.raises(Exception):
        # Bypasses index_documents (which creates the collection) and inserts directly
        indexer._index_documents([doc], collection_name="nonexistent_collection")
