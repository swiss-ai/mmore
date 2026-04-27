from typing import Dict, List
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import FakeEmbeddings
from langchain_milvus.utils.sparse import BaseSparseEmbedding
from pydantic import BaseModel, Field
from pymilvus import MilvusClient

from mmore.index.indexer import Indexer
from mmore.rag.model import DenseModelConfig, SparseModelConfig
from mmore.rag.pipeline import DEFAULT_PROMPT, RAGPipeline
from mmore.rag.retriever import Retriever
from mmore.type import MultimodalSample


class _FakeSparseEmbedding(BaseSparseEmbedding):
    def embed_query(self, query: str) -> Dict[int, float]:
        return {0: 1.0, 1: float(len(query))}

    def embed_documents(self, texts: List[str]) -> List[Dict[int, float]]:
        return [{0: 1.0, i + 1: float(len(t))} for i, t in enumerate(texts)]


class RAGInput(BaseModel):
    input: str = Field(..., description="The user query or question.")
    collection_name: str = Field(..., description="The Milvus collection name to search.")


_DOCS = [
    MultimodalSample(id="doc-1", document_id="doc-1", text="Paris is the capital of France.", modalities=[], metadata={}),
    MultimodalSample(id="doc-2", document_id="doc-2", text="The Eiffel Tower stands 330 metres tall.", modalities=[], metadata={}),
]

_COLLECTION = "test_col"


@pytest.fixture(scope="module")
def populated_db(tmp_path_factory):
    db_path = str(tmp_path_factory.mktemp("api_db") / "test.db")
    with patch("mmore.index.indexer.SparseModel.from_config", return_value=_FakeSparseEmbedding()):
        client = MilvusClient(db_path, enable_sparse=True)
        indexer = Indexer(
            dense_model_config=DenseModelConfig(model_name="debug"),
            sparse_model_config=SparseModelConfig(model_name="naver/splade-cocondenser-selfdistil"),
            client=client,
        )
        indexer.index_documents(_DOCS, collection_name=_COLLECTION)
    return db_path


@pytest.fixture(scope="module")
def app(populated_db):
    retriever = Retriever(
        dense_model=FakeEmbeddings(size=2048),
        sparse_model=_FakeSparseEmbedding(),
        client=MilvusClient(populated_db),
        hybrid_search_weight=0.5,
        k=2,
        use_web=False,
        reranker_model=None,
        reranker_tokenizer=None,
    )

    llm = FakeListChatModel(responses=["Paris is the capital of France."] * 20)
    prompt = ChatPromptTemplate.from_messages(
        [("system", DEFAULT_PROMPT), ("human", "{input}")]
    )
    pipeline = RAGPipeline(retriever=retriever, prompt_template=prompt, llm=llm)

    api = FastAPI()

    @api.post("/rag")
    def rag_endpoint(input_data: RAGInput):
        return pipeline(input_data.model_dump(), return_dict=True)[0]

    return api


@pytest.fixture(scope="module")
def client(app):
    return TestClient(app)


def test_rag_endpoint_returns_200(client):
    response = client.post(
        "/rag", json={"input": "What is the capital of France?", "collection_name": _COLLECTION}
    )
    assert response.status_code == 200


def test_rag_endpoint_has_answer_field(client):
    response = client.post(
        "/rag", json={"input": "How tall is the Eiffel Tower?", "collection_name": _COLLECTION}
    )
    data = response.json()
    assert "answer" in data
    assert isinstance(data["answer"], str)
    assert len(data["answer"]) > 0


def test_rag_endpoint_missing_field_returns_422(client):
    response = client.post("/rag", json={"input": "What is RAG?"})
    assert response.status_code == 422
