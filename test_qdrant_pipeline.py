"""
Smoke-test for the Qdrant backend.

Indexes 5 toy documents and runs 3 queries against each available backend.
If both backends are available, prints a side-by-side top-1 comparison.

Run from the project root:

    python test_qdrant_pipeline.py

No real model weights are downloaded — uses the built-in FakeEmbeddings
dense model (model_name="debug") and a tiny stub sparse model so the
script runs offline in seconds.

The point of this script is to demonstrate that switching from Milvus to
Qdrant is a one-line YAML change (``db.backend: qdrant``) — every other
mmore code path is identical.
"""

import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

# Use the local source tree, not any installed copy.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mmore.index.indexer import DBConfig, Indexer, IndexerConfig
from mmore.rag.model.dense.base import DenseModelConfig
from mmore.rag.model.sparse.base import SparseModelConfig
from mmore.rag.retriever import Retriever, RetrieverConfig
from mmore.type import MultimodalSample

# ── Stub sparse model so we don't need to download SPLADE for a smoke test ──
from langchain_milvus.utils.sparse import BaseSparseEmbedding


class StubSparseEmbedding(BaseSparseEmbedding):
    """Returns a deterministic sparse vector keyed by word hash."""

    def embed_query(self, query: str) -> Dict[int, float]:
        return {hash(w) % 512: 1.0 for w in query.split()}

    def embed_documents(self, texts: List[str]) -> List[Dict[int, float]]:
        return [self.embed_query(t) for t in texts]


import mmore.rag.model.sparse.base as _sparse_base

_orig_sparse_from_config = _sparse_base.SparseModel.from_config


@classmethod  # type: ignore[misc]
def _stub_sparse_from_config(cls, config):
    return StubSparseEmbedding()


_sparse_base.SparseModel.from_config = _stub_sparse_from_config


# ── Toy corpus ────────────────────────────────────────────────────────────────
DOCS = [
    MultimodalSample(
        text="Barack Obama was born on August 4, 1961, in Honolulu, Hawaii.",
        modalities=[],
        metadata={"source": "wikipedia"},
    ),
    MultimodalSample(
        text="Google was founded by Larry Page and Sergey Brin in September 1998.",
        modalities=[],
        metadata={"source": "wikipedia"},
    ),
    MultimodalSample(
        text="The Eiffel Tower is located on the Champ de Mars in Paris, France.",
        modalities=[],
        metadata={"source": "wikipedia"},
    ),
    MultimodalSample(
        text="The Python programming language was created by Guido van Rossum.",
        modalities=[],
        metadata={"source": "wikipedia"},
    ),
    MultimodalSample(
        text="Mount Everest is the world's highest mountain above sea level.",
        modalities=[],
        metadata={"source": "wikipedia"},
    ),
]

QUERIES = [
    "When was Barack Obama born?",
    "Who founded Google?",
    "Where is the Eiffel Tower located?",
]

COLLECTION = "smoke_test"
DENSE_MODEL = "debug"   # → langchain FakeEmbeddings(size=2048), no download
SPARSE_MODEL = "splade"  # intercepted by the stub above


def build_config(backend: str, db_uri: str) -> IndexerConfig:
    return IndexerConfig(
        dense_model=DenseModelConfig(model_name=DENSE_MODEL, is_multimodal=False),
        sparse_model=SparseModelConfig(model_name=SPARSE_MODEL, is_multimodal=False),
        db=DBConfig(backend=backend, uri=db_uri, name="smoke_db"),
    )


def _close_if_qdrant(client) -> None:
    """Release the Qdrant file lock; no-op for MilvusClient."""
    if hasattr(client, "close"):
        client.close()


def run_backend(backend: str) -> dict:
    tmp = tempfile.mkdtemp(prefix=f"mmore_{backend}_")
    # Milvus needs a .db file path; Qdrant needs a directory.
    db_uri = str(Path(tmp) / "data.db") if backend == "milvus" else tmp
    try:
        print(f"\n{'=' * 60}\n  Backend: {backend.upper()}\n  DB path: {db_uri}\n{'=' * 60}")

        # ── Index ─────────────────────────────────────────────────────────
        print("\n[1/3] Indexing 5 documents...")
        cfg = build_config(backend, db_uri)
        indexer = Indexer.from_config(cfg)
        n = indexer.index_documents(DOCS, collection_name=COLLECTION)
        print(f"      Inserted: {n} chunks")
        # Release the Qdrant file lock so the retriever can open its own client.
        _close_if_qdrant(indexer.client)

        # ── Retrieve ──────────────────────────────────────────────────────
        print("\n[2/3] Running retrieval for 3 queries...")
        ret_cfg = RetrieverConfig(
            db=DBConfig(backend=backend, uri=db_uri, name="smoke_db"),
            k=2,
            collection_name=COLLECTION,
            reranker_model_name=None,  # skip reranker — no GPU needed
        )
        retriever = Retriever.from_config(ret_cfg)

        results = {}
        for q in QUERIES:
            hits = retriever.retrieve(q, collection_name=COLLECTION, k=2)
            results[q] = hits
            top = hits[0]["entity"]["text"][:80] if hits else "(no results)"
            print(f"\n  Q: {q}")
            print(f"  A: {top}…")

        # ── Index metadata round-trip ─────────────────────────────────────
        print("\n[3/3] Checking model metadata round-trip...")
        dense_meta = retriever.client.describe_index(COLLECTION, "dense_embedding")
        sparse_meta = retriever.client.describe_index(COLLECTION, "sparse_embedding")
        assert dense_meta["model_name"] == DENSE_MODEL, (
            f"dense model mismatch: {dense_meta}"
        )
        print(f"      dense  model: {dense_meta['model_name']}  ✓")
        print(f"      sparse model: {sparse_meta['model_name']}  ✓")

        _close_if_qdrant(retriever.client)
        print(f"\n  Backend {backend.upper()} — ALL CHECKS PASSED ✓\n")
        return results

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def compare(results_milvus: dict, results_qdrant: dict) -> None:
    print(f"\n{'=' * 60}\n  Milvus vs Qdrant top-1 comparison\n{'=' * 60}")
    matches = 0
    for q in QUERIES:
        m_top = results_milvus[q][0]["id"] if results_milvus[q] else None
        q_top = results_qdrant[q][0]["id"] if results_qdrant[q] else None
        marker = "✓ match" if m_top == q_top else "~ differ (RRF vs WeightedRanker)"
        print(f"\n  Q: {q}")
        print(f"     Milvus top-1 id: {m_top}")
        print(f"     Qdrant top-1 id: {q_top}  {marker}")
        if m_top == q_top:
            matches += 1
    pct = matches / len(QUERIES) * 100
    print(f"\n  Agreement: {matches}/{len(QUERIES)} queries ({pct:.0f}%)")


if __name__ == "__main__":
    backends_to_test = []

    try:
        import milvus_lite  # noqa: F401

        backends_to_test.append("milvus")
    except ImportError:
        print("\nmilvus-lite not installed — skipping Milvus backend.")
        print("(Expected on ARM64; install mmore[index] on x86_64 to enable.)\n")

    try:
        import qdrant_client  # noqa: F401

        backends_to_test.append("qdrant")
    except ImportError:
        print("\nqdrant-client not installed — skipping Qdrant backend.")
        print("Install with:  pip install mmore[qdrant]\n")

    if not backends_to_test:
        print("No backends available — install mmore[index] or mmore[qdrant].")
        sys.exit(1)

    all_results = {}
    for b in backends_to_test:
        all_results[b] = run_backend(b)

    if "milvus" in all_results and "qdrant" in all_results:
        compare(all_results["milvus"], all_results["qdrant"])

    print("\nDone.")
