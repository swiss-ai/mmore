"""
Integration tests for the RAG HTTP API (run_rag.py).
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from mmore.run_rag import create_api


@pytest.fixture(scope="module")
def client():
    """Builds the FastAPI app via create_api() with a mocked RAGPipeline."""
    mock_pipeline = MagicMock()
    mock_pipeline.rag_chain.invoke.return_value = {
        "input": "What is the capital of France?",
        "context": "Paris is the capital of France.",
        "answer": "Paris.",
    }

    app = create_api(mock_pipeline, "/rag")
    return TestClient(app)


# ---------------------------------------------------------------------------
# POST /rag
# ---------------------------------------------------------------------------


def test_rag_endpoint_returns_200(client):
    response = client.post(
        "/rag",
        json={
            "input": {
                "input": "What is the capital of France?",
                "collection_name": "test_col",
            }
        },
    )
    assert response.status_code == 200


def test_rag_endpoint_response_shape(client):
    response = client.post(
        "/rag",
        json={
            "input": {
                "input": "How tall is the Eiffel Tower?",
                "collection_name": "test_col",
            }
        },
    )
    data = response.json()
    assert set(data.keys()) == {"input", "context", "answer"}
    assert data["answer"] == "Paris."


def test_rag_endpoint_includes_final_documents_from_dicts():
    """Pipeline validate_output returns docs as dicts after model_dump()."""
    mock_pipeline = MagicMock()
    mock_pipeline.rag_chain.invoke.return_value = {
        "input": "q",
        "context": "[1] chunk text",
        "answer": "a",
        "docs": [
            {
                "page_content": "chunk text",
                "metadata": {
                    "id": "file-1+chunk-0",
                    "similarity": 0.91,
                    "rerank_score": 2.1,
                    "rank": 1,
                },
            }
        ],
    }
    tc = TestClient(create_api(mock_pipeline, "/rag"))
    data = tc.post(
        "/rag", json={"input": {"input": "q", "collection_name": "test_col"}}
    ).json()

    assert len(data["documents"]) == 1
    assert data["documents"][0]["page_content"] == "chunk text"
    assert data["documents"][0]["metadata"]["id"] == "file-1+chunk-0"


def test_rag_endpoint_includes_final_documents():
    mock_pipeline = MagicMock()
    mock_pipeline.rag_chain.invoke.return_value = {
        "input": "q",
        "context": "[1] chunk text",
        "answer": "a",
        "docs": [
            MagicMock(
                page_content="chunk text",
                metadata={
                    "id": "file-1+chunk-0",
                    "similarity": 0.91,
                    "rerank_score": 2.1,
                    "rank": 1,
                },
            )
        ],
    }
    tc = TestClient(create_api(mock_pipeline, "/rag"))
    data = tc.post(
        "/rag", json={"input": {"input": "q", "collection_name": "test_col"}}
    ).json()

    assert len(data["documents"]) == 1
    assert data["documents"][0]["page_content"] == "chunk text"
    assert data["documents"][0]["metadata"]["id"] == "file-1+chunk-0"
    assert data["documents"][0]["metadata"]["similarity"] == 0.91


def test_rag_endpoint_missing_input_returns_422(client):
    response = client.post("/rag", json={})
    assert response.status_code == 422

    response = client.post("/rag", json={"input": {"collection_name": "test_col"}})
    assert response.status_code == 422


def test_rag_endpoint_invokes_pipeline_with_inner_dict():
    """The endpoint should call rag_chain.invoke() with the inner input dict."""
    mock_pipeline = MagicMock()
    mock_pipeline.rag_chain.invoke.return_value = {
        "input": "x",
        "context": "y",
        "answer": "z",
    }
    tc = TestClient(create_api(mock_pipeline, "/rag"))

    tc.post(
        "/rag", json={"input": {"input": "test query", "collection_name": "my_col"}}
    )

    mock_pipeline.rag_chain.invoke.assert_called_once_with(
        {"input": "test query", "collection_name": "my_col"}
    )


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


def test_health_check_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
