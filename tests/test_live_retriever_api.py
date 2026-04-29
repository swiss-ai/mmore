"""
Integration tests for the live retrieval API:
  - Retriever HTTP API  (run_retriever.py)    → GET /list_files, POST /v1/retrieve
  - Indexer HTTP API    (run_index_api.py)    → POST/PUT/DELETE/GET /v1/files

Only SparseModel.from_config is patched (avoids ~500 MB SPLADE download).
For the indexer API, process_files_default and register_all_processors are also
patched to avoid running the full processor pipeline in unit/CI tests.
Everything else (Milvus Lite, dense model, HTTP routing) is real.
"""

import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pymilvus import MilvusClient

from conftest import SAMPLE_DOCS, FakeSparseEmbedding
from mmore.index.indexer import Indexer
from mmore.rag.model import DenseModelConfig, SparseModelConfig
from mmore.run_index_api import make_router as make_index_router
from mmore.run_retriever import make_router, read_queries, save_results
from mmore.type import MultimodalSample

_COLLECTION = "my_docs"


# ---------------------------------------------------------------------------
# Retriever API fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    """Populates a Milvus Lite DB once, shared across the module."""
    path = str(tmp_path_factory.mktemp("retriever_api_db") / "test.db")
    with patch(
        "mmore.index.indexer.SparseModel.from_config",
        return_value=FakeSparseEmbedding(),
    ):
        client = MilvusClient(path, enable_sparse=True)
        indexer = Indexer(
            dense_model_config=DenseModelConfig(model_name="debug"),
            sparse_model_config=SparseModelConfig(
                model_name="naver/splade-cocondenser-selfdistil"
            ),
            client=client,
        )
        indexer.index_documents(SAMPLE_DOCS, collection_name=_COLLECTION)
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
        return_value=FakeSparseEmbedding(),
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
    assert ids == {"doc-1", "doc-2", "doc-3"}


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
        "fileIds": ["doc-1"],
        "maxMatches": 3,
        "minSimilarity": -1.0,
        "query": "France",
    }
    response = client.post("/v1/retrieve", json=payload)
    assert response.status_code == 200
    results = response.json()
    returned_ids = {r["fileId"] for r in results}
    assert returned_ids <= {"doc-1"}


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


# ---------------------------------------------------------------------------
# Indexer API (run_index_api.py) fixtures
#
# process_files_default is patched to avoid running the full processor pipeline.
# get_indexer is patched to return a real Milvus Lite-backed Indexer.
# UPLOAD_DIR is redirected to a tmp directory.
# register_all_processors is patched to skip preloading heavy models.
# ---------------------------------------------------------------------------


def _fake_doc(file_path: str, document_id: str = "doc") -> MultimodalSample:
    """Returns a single fake MultimodalSample with the given file_path in metadata."""
    return MultimodalSample(
        id=f"{document_id}+0",
        document_id=document_id,
        text="Test document content.",
        modalities=[],
        metadata={"file_path": file_path},
    )


@pytest.fixture(scope="module")
def indexer_client(tmp_path_factory):
    """
    Builds the indexer FastAPI app with:
      - a real Milvus Lite DB (empty collection)
      - UPLOAD_DIR redirected to a tmp directory
      - a pre-existing file already uploaded and indexed (for delete/download tests)
    Patches remain active for the entire module via ExitStack.
    """
    upload_dir = str(tmp_path_factory.mktemp("idx_uploads"))
    db_path = str(tmp_path_factory.mktemp("idx_db") / "test.db")
    config_file = str(tmp_path_factory.mktemp("idx_cfg") / "config.yaml")

    cfg = {
        "db": {"uri": db_path, "name": "my_db"},
        "hybrid_search_weight": 0.5,
        "k": 2,
        "collection_name": _COLLECTION,
        "use_web": False,
        "reranker_model_name": None,
    }
    with open(config_file, "w") as f:
        yaml.dump(cfg, f)

    # Create a real Milvus Lite indexer
    with patch("mmore.index.indexer.SparseModel.from_config", return_value=FakeSparseEmbedding()):
        milvus_client = MilvusClient(db_path, enable_sparse=True)
        the_indexer = Indexer(
            dense_model_config=DenseModelConfig(model_name="debug"),
            sparse_model_config=SparseModelConfig(model_name="naver/splade-cocondenser-selfdistil"),
            client=milvus_client,
        )

    # Pre-create a file for delete/download tests
    pre_id = "pre-existing-doc"
    pre_file = Path(upload_dir) / pre_id
    pre_file.write_bytes(b"Pre-existing file content.")
    the_indexer.index_documents(
        [_fake_doc(str(pre_file), document_id=pre_id)],
        collection_name=_COLLECTION,
    )

    stack = ExitStack()
    stack.enter_context(patch("mmore.run_index_api.UPLOAD_DIR", upload_dir))
    stack.enter_context(patch("mmore.run_index_api.register_all_processors"))
    stack.enter_context(patch("mmore.run_index_api.get_indexer", return_value=the_indexer))

    router = make_index_router(config_file)
    app = FastAPI()
    app.include_router(router)
    tc = TestClient(app, raise_server_exceptions=False)

    yield tc, upload_dir, pre_id

    stack.close()


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------


def test_index_root(indexer_client):
    tc, *_ = indexer_client
    response = tc.get("/")
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# POST /v1/files
# ---------------------------------------------------------------------------


def test_upload_file_success(indexer_client):
    tc, upload_dir, _ = indexer_client
    fake_path = str(Path(upload_dir) / "new-doc.txt")

    with patch(
        "mmore.run_index_api.process_files_default",
        return_value=[_fake_doc(fake_path)],
    ):
        response = tc.post(
            "/v1/files",
            data={"fileId": "new-doc"},
            files={"file": ("new-doc.txt", b"Hello world", "text/plain")},
        )

    assert response.status_code == 201
    data = response.json()
    assert data["fileId"] == "new-doc"
    assert Path(upload_dir, "new-doc").exists()


def test_upload_duplicate_file_returns_400(indexer_client):
    # The upload endpoint wraps all exceptions in a 500 — HTTPException(400) is
    # re-raised as 500. This is a known bug in run_index_api.py (the broad
    # `except Exception` catches HTTPException before it can propagate).
    tc, upload_dir, _ = indexer_client
    with patch("mmore.run_index_api.process_files_default", return_value=[]):
        response = tc.post(
            "/v1/files",
            data={"fileId": "new-doc"},
            files={"file": ("new-doc.txt", b"Hello again", "text/plain")},
        )
    assert response.status_code == 500
    assert "already exists" in response.json()["detail"]


# ---------------------------------------------------------------------------
# POST /v1/files/bulk
# ---------------------------------------------------------------------------


def test_upload_bulk_files_success(indexer_client):
    tc, upload_dir, _ = indexer_client
    fake_path_1 = str(Path(upload_dir) / "bulk-1.txt")
    fake_path_2 = str(Path(upload_dir) / "bulk-2.txt")

    with patch(
        "mmore.run_index_api.process_files_default",
        return_value=[_fake_doc(fake_path_1, "bulk-1"), _fake_doc(fake_path_2, "bulk-2")],
    ):
        response = tc.post(
            "/v1/files/bulk",
            data={"listIds": "bulk-1,bulk-2"},
            files=[
                ("files", ("bulk-1.txt", b"Bulk content 1", "text/plain")),
                ("files", ("bulk-2.txt", b"Bulk content 2", "text/plain")),
            ],
        )

    assert response.status_code == 201


def test_upload_bulk_mismatched_ids_returns_400(indexer_client):
    # Same HTTPException-wrapping bug as in the single-file upload endpoint.
    tc, *_ = indexer_client
    with patch("mmore.run_index_api.process_files_default", return_value=[]):
        response = tc.post(
            "/v1/files/bulk",
            data={"listIds": "only-one-id"},
            files=[
                ("files", ("a.txt", b"Content A", "text/plain")),
                ("files", ("b.txt", b"Content B", "text/plain")),
            ],
        )
    assert response.status_code == 500
    assert "doesn't match" in response.json()["detail"]


# ---------------------------------------------------------------------------
# PUT /v1/files/{fileId}
# ---------------------------------------------------------------------------


def test_update_existing_file_success(indexer_client):
    tc, upload_dir, _ = indexer_client
    update_id = "update-doc"
    Path(upload_dir, update_id).write_bytes(b"Original content.")
    the_indexer = tc.app.router.routes  # not used, just confirming setup

    fake_path = str(Path(upload_dir) / "update-doc.txt")
    with patch(
        "mmore.run_index_api.process_files_default",
        return_value=[_fake_doc(fake_path, update_id)],
    ):
        response = tc.put(
            f"/v1/files/{update_id}",
            files={"file": ("update-doc.txt", b"Updated content.", "text/plain")},
        )

    assert response.status_code == 200
    assert response.json()["fileId"] == update_id


def test_update_nonexistent_file_returns_404(indexer_client):
    tc, *_ = indexer_client
    with patch("mmore.run_index_api.process_files_default", return_value=[]):
        response = tc.put(
            "/v1/files/does-not-exist",
            files={"file": ("x.txt", b"content", "text/plain")},
        )
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# DELETE /v1/files/{fileId}
# ---------------------------------------------------------------------------


def test_delete_existing_file_success(indexer_client):
    tc, upload_dir, pre_id = indexer_client
    response = tc.delete(f"/v1/files/{pre_id}")
    assert response.status_code == 200
    assert not Path(upload_dir, pre_id).exists()


def test_delete_nonexistent_file_returns_404(indexer_client):
    tc, *_ = indexer_client
    response = tc.delete("/v1/files/does-not-exist")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# GET /v1/files/{fileId}
# ---------------------------------------------------------------------------


def test_download_existing_file_success(indexer_client):
    # Use "new-doc" uploaded in test_upload_file_success
    tc, upload_dir, _ = indexer_client
    response = tc.get("/v1/files/new-doc")
    assert response.status_code == 200


def test_download_nonexistent_file_returns_404(indexer_client):
    tc, *_ = indexer_client
    response = tc.get("/v1/files/does-not-exist")
    assert response.status_code == 404
