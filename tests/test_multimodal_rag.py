"""RAG vision + retriever (image paths, fallback Milvus)."""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_milvus.utils.sparse import BaseSparseEmbedding
from PIL import Image
from pymilvus import MilvusClient
from pymilvus.exceptions import MilvusException

from mmore.rag.llm import LLMConfig
from mmore.rag.model.vision.adapters import (
    DEFAULT_HF_VISION_MODEL,
    HuggingFaceVisionAdapter,
    get_multimodal_llm,
)
from mmore.rag.pipeline import RAGPipeline
from mmore.rag.retriever import Retriever, _parse_image_paths


class MockEmb(Embeddings):
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


def _ret():
    return Retriever(
        dense_model=MockEmb(),
        sparse_model=MockSparse(),
        client=MockMilvus(),
        hybrid_search_weight=0.5,
        k=2,
        use_web=False,
        reranker_model=None,
        reranker_tokenizer=None,
    )


def test_parse_paths_and_retriever_fallback_and_pipeline(monkeypatch, tmp_path):
    assert _parse_image_paths(None) == [] and _parse_image_paths(["/a.png"]) == [
        "/a.png"
    ]

    with patch.object(Retriever, "retrieve") as mr:
        mr.side_effect = [
            MilvusException(message="field image_paths not found in schema"),
            [{"id": "1", "distance": 0.1, "entity": {"text": "t"}}],
        ]
        assert (
            _ret()
            ._get_relevant_documents("q", run_manager=MagicMock())[0]
            .metadata["image_paths"]
            == []
        )
        assert mr.call_count == 2

    img = tmp_path / "i.png"
    Image.new("RGB", (2, 2)).save(img)

    class MM:
        def __init__(self):
            self.calls = []

        def invoke_with_images(self, text, images=None, system_prompt=None):
            self.calls.append(images or [])
            return "ok"

    mm = MM()
    monkeypatch.setattr(
        "mmore.rag.pipeline.load_images_from_paths",
        lambda paths, max_images=20: [f"L-{p}" for p in paths[:max_images]],
    )
    r = RunnableLambda(
        lambda _: [
            Document(page_content="c", metadata={"rank": 1, "image_paths": [str(img)]})
        ]
    )
    out = RAGPipeline(
        r,
        ChatPromptTemplate.from_messages(
            [("system", "C:{context}"), ("human", "{input}")]
        ),
        None,
        use_vision=True,
        multimodal_llm=mm,
        max_images_per_request=3,
    )({"input": "q", "collection_name": "x"}, return_dict=True)[0]
    assert (
        out["answer"] == "ok"
        and out["image_paths"] == [str(img)]
        and mm.calls == [[f"L-{img}"]]
    )

    hf = get_multimodal_llm(LLMConfig(llm_name="gpt2", provider="HF", max_new_tokens=5))
    assert (
        isinstance(hf, HuggingFaceVisionAdapter)
        and hf.model_id == DEFAULT_HF_VISION_MODEL
    )
