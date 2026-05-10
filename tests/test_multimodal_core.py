"""Embeddings multimodaux + indexeur dense multimodal."""

from unittest.mock import MagicMock, patch

import torch
from conftest import FakeSparseEmbedding
from pymilvus import MilvusClient

from mmore.index.indexer import Indexer
from mmore.rag.model import DenseModelConfig, SparseModelConfig
from mmore.rag.model.dense.multimodal import MultimodalEmbeddings
from mmore.type import MultimodalRawInput, MultimodalSample


class _PB(dict):
    def to(self, *a, **k):
        return self


def test_embeddings_helpers_embed_and_index_multimodal(tmp_path):
    cl, paths = MultimodalEmbeddings._extract_multimodal_inputs(
        "x <|image|>/p.png<|image|> y", proc_token="<|image|>"
    )
    assert paths == ["/p.png"] and ".png" not in cl.replace("<|image|>", "")
    s = MultimodalSample(
        text="t",
        modalities=[MultimodalRawInput(type="image", value="/q.png")],
        metadata={"file_path": "/d.pdf"},
    )
    doc = MultimodalEmbeddings._multimodal_to_doc(s)
    assert doc.metadata["file_path"] == "/d.pdf"

    with patch(
        "mmore.rag.model.dense.multimodal.load_images_from_paths", return_value=[]
    ):
        emb = MultimodalEmbeddings.__new__(MultimodalEmbeddings)
        emb.processor = MagicMock(
            side_effect=lambda **kw: _PB({"input_ids": torch.ones(1, 2)})
        )
        emb.device, emb.model = (
            "cpu",
            MagicMock(return_value=MagicMock(hidden_states=[torch.ones(1, 2, 4)])),
        )
        assert len(MultimodalEmbeddings.embed_documents(emb, ["hi"])[0]) == 4
        emb.model.return_value = MagicMock(hidden_states=[torch.ones(1, 1, 2)])
        MultimodalEmbeddings.embed_documents(
            emb, ["z<attachment><|image|>/a.png<|image|>"]
        )
        assert "<attachment>" not in emb.processor.call_args[1].get("text", "")

    with (
        patch("mmore.index.indexer.DenseModel.from_config") as md,
        patch("mmore.index.indexer.SparseModel.from_config") as ms,
    ):
        fs = FakeSparseEmbedding()
        ms.return_value = fs
        d = MagicMock()
        d.embed_query.return_value = [0.0] * 8
        d.embed_documents.side_effect = lambda texts: [[0.5] * 8 for _ in texts]
        md.return_value = d
        ix = Indexer(
            DenseModelConfig(
                model_name="Qwen/Qwen2.5-VL-3B-Instruct",
                is_multimodal=True,
            ),
            SparseModelConfig(
                model_name="naver/splade-cocondenser-selfdistil",
            ),
            MilvusClient(str(tmp_path / "db.db"), enable_sparse=True),
        )
        sm = MultimodalSample(
            id="1",
            document_id="1",
            text="b",
            modalities=[MultimodalRawInput("image", "/f.png")],
            metadata={},
        )
        exp = MultimodalEmbeddings._multimodal_to_text(sm)
        with patch.object(fs, "embed_documents", wraps=fs.embed_documents) as spy:
            assert ix.index_documents([sm], "c") == 1
        assert d.embed_documents.call_args[0][0] == [exp] and spy.call_args[0][0] == [
            "b"
        ]
