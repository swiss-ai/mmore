from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from mmore.rag.llm import LLMConfig
from mmore.rag.pipeline import RAGConfig, RAGPipeline
from mmore.rag.retriever import DBConfig, RetrieverConfig


class RAGInput(BaseModel):
    """
    Defines the expected input structure for the /rag endpoint.
    This structure must align with the MMOREInput model used internally
    by the RAGPipeline.
    """

    input: str = Field(..., description="The user query or question.")
    collection_name: str = Field(
        ..., description="The Milvus collection name to search."
    )


@pytest.fixture(scope="module")
def app():
    retriever_cfg = RetrieverConfig(
        db=DBConfig(uri="./proc_demo.db", name="my_db"), hybrid_search_weight=0.5, k=2
    )
    llm_cfg = LLMConfig(llm_name="gpt2")
    rag_cfg = RAGConfig(retriever=retriever_cfg, llm=llm_cfg)

    with patch("mmore.rag.pipeline.RAGPipeline.from_config") as mock_from_config:

        def mock_runnable(input_data, return_dict=False):
            if return_dict:
                return [{"answer": f"Mocked answer for query: {input_data['input']}"}]

            return [{"answer": f"Mocked answer for query: {input_data['input']}"}]

        # Create a mock RAGPipeline instance
        mock_pipeline = MagicMock(spec=RAGPipeline)

        mock_pipeline.side_effect = mock_runnable

        mock_from_config.return_value = mock_pipeline

        rag_pipeline = RAGPipeline.from_config(rag_cfg)

        api = FastAPI()

        @api.post("/rag")
        def rag_endpoint(input_data: RAGInput):
            return rag_pipeline(input_data.model_dump(), return_dict=True)[0]

        return api


@pytest.fixture(scope="module")
def client(app):
    return TestClient(app)


def test_rag_endpoint(client):
    """Test that the /rag endpoint returns a valid response structure."""

    response = client.post(
        "/rag", json={"input": "What is RAG?", "collection_name": "my_docs"}
    )

    assert response.status_code == 200

    data = response.json()
    assert "answer" in data
    assert isinstance(data["answer"], str)
    assert data["answer"].startswith("Mocked answer")
