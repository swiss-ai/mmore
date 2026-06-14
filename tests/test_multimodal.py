"""Multimodal stack: dense embeddings, multimodal indexer, vision RAG (mocked, no GPU)."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
import torch
import transformers
from conftest import FakeSparseEmbedding
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_milvus.utils.sparse import BaseSparseEmbedding
from PIL import Image
from pymilvus import MilvusClient
from pymilvus.exceptions import MilvusException

from mmore.index.indexer import Indexer
from mmore.rag.llm import LLMConfig
from mmore.rag.model import DenseModelConfig, SparseModelConfig
from mmore.rag.model.dense.multimodal import MultimodalEmbeddings
from mmore.rag.model.vision.adapters import (
    DEFAULT_HF_VISION_MODEL,
    BaseMultimodalLLM,
    HuggingFaceVisionAdapter,
    get_multimodal_llm,
)
from mmore.rag.pipeline import RAGPipeline
from mmore.rag.retriever import Retriever, _parse_image_paths
from mmore.type import DocumentMetadata, MultimodalRawInput, MultimodalSample


class _ProcessorBatch(dict):
    def to(self, *args, **kwargs):
        return self


class _MockDense(Embeddings):
    def embed_query(self, text):
        return [0.1, 0.2]

    def embed_documents(self, texts):
        return [[0.1, 0.2] for _ in texts]


class _MockSparse(BaseSparseEmbedding):
    def embed_query(self, query):
        return {0: 1.0}

    def embed_documents(self, texts):
        return [{0: 1.0} for _ in texts]


def _make_embedding_instance(
    *,
    hidden: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    last_hidden_state=None,
    hidden_states=None,
):
    emb = MultimodalEmbeddings.__new__(MultimodalEmbeddings)
    emb.max_images = 20
    emb.max_image_side = 768
    emb.device = torch.device("cpu")  # type: ignore[reportPrivateImportUsage]
    batch = {"input_ids": torch.ones(1, 2)}  # type: ignore[reportPrivateImportUsage]
    if attention_mask is not None:
        batch["attention_mask"] = attention_mask
    emb.processor = MagicMock(
        side_effect=lambda **kw: _ProcessorBatch(batch),
        image_token="<|image_pad|>",
    )
    if last_hidden_state is None and hidden_states is None:
        last_hidden_state = hidden
    model = MagicMock()
    model.return_value = MagicMock(
        last_hidden_state=last_hidden_state, hidden_states=hidden_states
    )
    emb.model = model
    return emb


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, []),
        (["/a.png"], ["/a.png"]),
        ([None, "/a.png", "", 42], ["/a.png"]),
        ('["/b.png", null]', ["/b.png"]),
    ],
)
def test_parse_image_paths(raw, expected):
    assert _parse_image_paths(raw) == expected


def test_embedding_static_helpers(tmp_path):
    text = " ".join(f"<|image|>/img{i}.png<|image|>" for i in range(6))
    paths, prompt = MultimodalEmbeddings._cap_image_tags_in_text(
        text, "<|image|>", "<|image_pad|>", max_images=4
    )
    assert len(paths) == 4 and prompt.count("<|image_pad|>") == 4

    paths, _ = MultimodalEmbeddings._cap_image_tags_in_text(
        "x <|image|>/p.png<|image|> <|image|>/p.png<|image|> y",
        "<|image|>",
        "<|image_pad|>",
        max_images=4,
    )
    assert paths == ["/p.png"]

    sample = MultimodalSample(
        text="t",
        modalities=[MultimodalRawInput(type="image", value="/q.png")],
        metadata=DocumentMetadata(file_path="/d.pdf"),
    )
    assert (
        MultimodalEmbeddings._multimodal_to_doc(sample).metadata["file_path"]
        == "/d.pdf"
    )
    assert "/q.png" in MultimodalEmbeddings._multimodal_to_text(sample)

    from mmore.rag.model.vision.image_utils import load_images_from_paths

    assert load_images_from_paths([str(tmp_path / "nope.png")], max_images=5) == []


@pytest.mark.parametrize(
    "use_hidden_states_tuple",
    [False, True],
    ids=["last_hidden_state", "hidden_states_fallback"],
)
def test_embed_documents_pools_sequence_hidden(use_hidden_states_tuple):
    hidden = torch.ones(1, 2, 4)  # type: ignore[reportPrivateImportUsage]
    mask = torch.ones(1, 2)  # type: ignore[reportPrivateImportUsage]
    if use_hidden_states_tuple:
        emb = _make_embedding_instance(
            hidden=hidden,
            attention_mask=mask,
            last_hidden_state=None,
            hidden_states=(hidden,),
        )
    else:
        emb = _make_embedding_instance(hidden=hidden)

    with patch(
        "mmore.rag.model.dense.multimodal.load_images_from_paths", return_value=[]
    ):
        out = MultimodalEmbeddings.embed_documents(emb, ["hello"])

    model = cast(MagicMock, emb.model)
    model.assert_called_once()
    assert model.call_args.kwargs.get("output_hidden_states") is True
    assert out[0] == [1.0, 1.0, 1.0, 1.0]


def test_embed_documents_strips_attachment_and_caps_tags():
    emb = _make_embedding_instance(
        hidden=torch.ones(1, 1, 2)  # type: ignore[reportPrivateImportUsage]
    )
    with patch(
        "mmore.rag.model.dense.multimodal.load_images_from_paths", return_value=[]
    ):
        MultimodalEmbeddings.embed_documents(
            emb, ["z<attachment><|image|>/a.png<|image|>"]
        )
    assert "<attachment>" not in emb.processor.call_args[1].get("text", "")


@patch("mmore.index.indexer.DenseModel.from_config")
@patch("mmore.index.indexer.SparseModel.from_config")
def test_multimodal_indexer_dense_then_sparse(
    mock_sparse_cfg, mock_dense_cfg, tmp_path
):
    dense = MagicMock()
    dense.embed_query.return_value = [0.1] * 8
    dense.embed_documents.side_effect = lambda texts: [[0.5] * 8 for _ in texts]
    mock_dense_cfg.return_value = dense
    sparse = FakeSparseEmbedding()
    mock_sparse_cfg.return_value = sparse

    indexer = Indexer(
        DenseModelConfig(model_name="Qwen/Qwen2.5-VL-3B-Instruct", is_multimodal=True),
        SparseModelConfig(model_name="splade"),
        MilvusClient(str(tmp_path / "db.db"), enable_sparse=True),
    )
    assert indexer.sparse_model is None

    sample = MultimodalSample(
        id="1",
        document_id="1",
        text="body",
        modalities=[MultimodalRawInput(type="image", value="/f.png")],
        metadata=DocumentMetadata(),
    )
    indexer.sparse_model = sparse
    with patch.object(sparse, "embed_documents", wraps=sparse.embed_documents) as spy:
        assert indexer.index_documents([sample], "col") == 1

    dense_text = MultimodalEmbeddings._multimodal_to_text(sample)
    assert dense.embed_documents.call_args[0][0] == [dense_text]
    assert spy.call_args[0][0] == ["body"]


def test_retriever_image_paths_fallback():
    retriever = Retriever(
        dense_model=_MockDense(),
        sparse_model=_MockSparse(),
        client=MagicMock(spec=MilvusClient),
        hybrid_search_weight=0.5,
        k=2,
        use_web=False,
        reranker_model=None,
        reranker_tokenizer=None,
    )
    with patch.object(Retriever, "retrieve") as retrieve:
        retrieve.side_effect = [
            MilvusException(message="field image_paths not found in schema"),
            [{"id": "1", "distance": 0.1, "entity": {"text": "t"}}],
        ]
        doc = retriever._get_relevant_documents("q", run_manager=MagicMock())[0]
    assert doc.metadata["image_paths"] == [] and retrieve.call_count == 2


def test_vision_rag_pipeline_and_adapter(monkeypatch, tmp_path):
    img = tmp_path / "i.png"
    Image.new("RGB", (2, 2)).save(img)

    class _VisionLLM(BaseMultimodalLLM):
        def __init__(self):
            self.calls = []

        def invoke_with_images(self, text, images=None, system_prompt=None):
            self.calls.append(images or [])
            return "ok"

    vlm = _VisionLLM()
    monkeypatch.setattr(
        "mmore.rag.pipeline.load_images_from_paths",
        lambda paths, max_images=20: [f"loaded-{p}" for p in paths[:max_images]],
    )
    retriever = RunnableLambda(
        lambda _: [
            Document(
                page_content="c",
                metadata={"rank": 1, "image_paths": [str(img)]},
            )
        ]
    )
    out = RAGPipeline(
        cast(Any, retriever),
        ChatPromptTemplate.from_messages(
            [("system", "C:{context}"), ("human", "{input}")]
        ),
        None,
        use_vision=True,
        multimodal_llm=vlm,
        max_images_per_request=3,
    )({"input": "q", "collection_name": "x"}, return_dict=True)[0]

    assert out["answer"] == "ok"
    assert out["image_paths"] == [str(img)]
    assert vlm.calls == [[f"loaded-{img}"]]

    hf = get_multimodal_llm(LLMConfig(llm_name="gpt2", provider="HF", max_new_tokens=5))
    assert (
        isinstance(hf, HuggingFaceVisionAdapter)
        and hf.model_id == DEFAULT_HF_VISION_MODEL
    )

    with pytest.raises(ValueError, match="Vision mode requires a multimodal LLM"):
        RAGPipeline(
            MagicMock(),
            ChatPromptTemplate.from_messages([("human", "{input}")]),
            None,
            use_vision=True,
            multimodal_llm=None,
        )


@pytest.mark.parametrize(
    "model_id,expected_cls_attr",
    [
        ("Qwen/Qwen2-VL-2B-Instruct", "Qwen2VLForConditionalGeneration"),
        ("Qwen/Qwen2.5-VL-3B-Instruct", "Qwen2_5_VLForConditionalGeneration"),
    ],
)
def test_hf_vision_adapter_model_class(model_id, expected_cls_attr):
    adapter = HuggingFaceVisionAdapter(model_id=model_id)

    def _config(mid: str):
        cfg = MagicMock()
        cfg.model_type = "qwen2_vl" if "2-VL" in mid else "qwen2_5_vl"
        return cfg

    with patch(
        "mmore.rag.model.vision.adapters.AutoConfig.from_pretrained",
        side_effect=_config,
    ):
        resolved = adapter._resolve_model_cls()
    assert resolved is getattr(transformers, expected_cls_attr)
