"""
Integration tests for the Retriever HTTP API (run_retriever.py).

Uses make_router() for real — only SparseModel.from_config is patched to
avoid the ~500 MB SPLADE download. Everything else (Milvus Lite, dense model,
HTTP routing) is real.
"""

import json
from pathlib import Path
from typing import Dict, List
from unittest.mock import patch

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_milvus.utils.sparse import BaseSparseEmbedding
from pymilvus import MilvusClient

from mmore.index.indexer import Indexer
from mmore.rag.model import DenseModelConfig, SparseModelConfig
from mmore.run_retriever import make_router, read_queries, save_results
from mmore.type import MultimodalSample


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class _FakeSparseEmbedding(BaseSparseEmbedding):
    def embed_query(self, query: str) -> Dict[int, float]:
        return {0: 1.0, 1: float(len(query))}

    def embed_documents(self, texts: List[str]) -> List[Dict[int, float]]:
        return [{0: 1.0, i + 1: float(len(t))} for i, t in enumerate(texts)]


_DOCS = [
    MultimodalSample(
        id="file-1+0",
        document_id="file-1",
        text="Paris is the capital of France.",
        modalities=[],
        metadata={"filename": "paris.txt"},
    ),
    MultimodalSample(
        id="file-2+0",
        document_id="file-2",
        text="The Eiffel Tower stands 330 metres tall.",
        modalities=[],
        metadata={"filename": "eiffel.txt"},
    ),
    MultimodalSample(
        id="file-3+0",
        document_id="file-3",
        text="Milvus is an open-source vector database.",
        modalities=[],
        metadata={"filename": "milvus.txt"},
    ),
]

_COLLECTION = "my_docs"


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    """Populates a Milvus Lite DB once, shared across the module."""
    path = str(tmp_path_factory.mktemp("retriever_api_db") / "test.db")
    with patch(
        "mmore.index.indexer.SparseModel.from_config",
        return_value=_FakeSparseEmbedding(),
    ):
        client = MilvusClient(path, enable_sparse=True)
        indexer = Indexer(
            dense_model_config=DenseModelConfig(model_name="debug"),
            sparse_model_config=SparseModelConfig(
                model_name="naver/splade-cocondenser-selfdistil"
            ),
            client=client,
        )
        indexer.index_documents(_DOCS, collection_name=_COLLECTION)
    return path


@pytest.fixture(scope="module")
def config_path(tmp_path_factory, db_path):
    """Writes a real RetrieverConfig YAML pointing at the populated DB."""
    cfg = {
        "db": {"uri": db_path, "name": "my_db"},
        "hybrid_search_weight": 0.5,
        "k": 2,
        "collection_name": _COLLECTION,
        "use_web": False,
        "reranker_model_name": None,
    }
    path = tmp_path_factory.mktemp("cfg") / "retriever.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    return str(path)


@pytest.fixture(scope="module")
def client(config_path):
    """Builds the real FastAPI app via make_router() and returns a TestClient."""
    with patch(
        "mmore.rag.retriever.SparseModel.from_config",
        return_value=_FakeSparseEmbedding(),
    ):
        router = make_router(config_path)

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /list_files
# ---------------------------------------------------------------------------


def test_list_files_returns_all_documents(client):
    response = client.get("/list_files", params={"collection_name": _COLLECTION})
    assert response.status_code == 200
    ids = {f["id"] for f in response.json()}
    assert ids == {"file-1", "file-2", "file-3"}


def test_list_files_missing_collection_name(client):
    response = client.get("/list_files")
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# POST /v1/retrieve
# ---------------------------------------------------------------------------


def test_retrieve_returns_results(client):
    payload = {
        "fileIds": [],
        "maxMatches": 2,
        "minSimilarity": -1.0,
        "query": "capital of France",
    }
    response = client.post("/v1/retrieve", json=payload)
    assert response.status_code == 200
    results = response.json()
    assert isinstance(results, list)
    assert len(results) <= 2
    assert all("content" in r for r in results)
    assert all("similarity" in r for r in results)
    assert all("fileId" in r for r in results)


def test_retrieve_response_shape(client):
    payload = {
        "fileIds": [],
        "maxMatches": 1,
        "minSimilarity": -1.0,
        "query": "vector database",
    }
    response = client.post("/v1/retrieve", json=payload)
    assert response.status_code == 200
    result = response.json()[0]
    assert "fileId" in result
    assert "content" in result
    assert "similarity" in result
    assert "pageNumbers" in result
    assert "paragraphNumbers" in result


def test_retrieve_with_file_ids_filter(client):
    payload = {
        "fileIds": ["file-1"],
        "maxMatches": 3,
        "minSimilarity": -1.0,
        "query": "France",
    }
    response = client.post("/v1/retrieve", json=payload)
    assert response.status_code == 200
    results = response.json()
    returned_ids = {r["fileId"] for r in results}
    assert returned_ids <= {"file-1"}


def test_retrieve_max_matches_respected(client):
    payload = {
        "fileIds": [],
        "maxMatches": 1,
        "minSimilarity": -1.0,
        "query": "anything",
    }
    response = client.post("/v1/retrieve", json=payload)
    assert response.status_code == 200
    assert len(response.json()) <= 1


def test_retrieve_high_min_similarity_returns_empty(client):
    payload = {
        "fileIds": [],
        "maxMatches": 3,
        "minSimilarity": 0.999,
        "query": "France",
    }
    response = client.post("/v1/retrieve", json=payload)
    assert response.status_code == 200
    assert response.json() == []


def test_retrieve_missing_query_returns_422(client):
    payload = {"fileIds": [], "maxMatches": 2, "minSimilarity": -1.0}
    response = client.post("/v1/retrieve", json=payload)
    assert response.status_code == 422


def test_retrieve_missing_max_matches_returns_422(client):
    payload = {"fileIds": [], "minSimilarity": -1.0, "query": "test"}
    response = client.post("/v1/retrieve", json=payload)
    assert response.status_code == 422


def test_retrieve_invalid_max_matches_returns_422(client):
    """maxMatches has ge=1 constraint."""
    payload = {"fileIds": [], "maxMatches": 0, "minSimilarity": -1.0, "query": "test"}
    response = client.post("/v1/retrieve", json=payload)
    assert response.status_code == 422


def test_retrieve_invalid_min_similarity_returns_422(client):
    """minSimilarity must be between -1.0 and 1.0."""
    payload = {
        "fileIds": [],
        "maxMatches": 2,
        "minSimilarity": 5.0,
        "query": "test",
    }
    response = client.post("/v1/retrieve", json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Utility functions: read_queries / save_results
# ---------------------------------------------------------------------------


def test_read_queries_returns_list(tmp_path):
    queries_file = tmp_path / "queries.jsonl"
    queries_file.write_text(
        json.dumps("What is the capital of France?") + "\n"
        + json.dumps("How tall is the Eiffel Tower?") + "\n"
    )
    result = read_queries(queries_file)
    assert result == ["What is the capital of France?", "How tall is the Eiffel Tower?"]


def test_read_queries_empty_file(tmp_path):
    queries_file = tmp_path / "empty.jsonl"
    queries_file.write_text("")
    result = read_queries(queries_file)
    assert result == []


def test_save_results_writes_valid_json(tmp_path):
    from langchain_core.documents import Document

    queries = ["What is Paris?"]
    docs = [Document(page_content="Paris is the capital.", metadata={"rank": 1, "similarity": 0.9, "id": "1", "page_numbers": [], "paragraph_numbers": []})]
    results = [docs]
    output_file = tmp_path / "results.json"

    save_results(results, queries, output_file)

    assert output_file.exists()
    data = json.loads(output_file.read_text())
    assert len(data) == 1
    assert data[0]["query"] == "What is Paris?"
    assert len(data[0]["context"]) == 1
    assert data[0]["context"][0]["page_content"] == "Paris is the capital."


def test_save_results_multiple_queries(tmp_path):
    from langchain_core.documents import Document

    queries = ["query 1", "query 2"]
    results = [
        [Document(page_content="doc A", metadata={"rank": 1, "similarity": 0.8, "id": "a", "page_numbers": [], "paragraph_numbers": []})],
        [Document(page_content="doc B", metadata={"rank": 1, "similarity": 0.7, "id": "b", "page_numbers": [], "paragraph_numbers": []})],
    ]
    output_file = tmp_path / "results.json"
    save_results(results, queries, output_file)

    data = json.loads(output_file.read_text())
    assert len(data) == 2
    assert data[0]["query"] == "query 1"
    assert data[1]["query"] == "query 2"
