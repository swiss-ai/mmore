"""End-to-end GPU running the README pipeline: process, postprocess, index, rag."""

import json
from pathlib import Path

import pytest
import yaml

from mmore.run_index import index
from mmore.run_postprocess import postprocess
from mmore.run_process import process
from mmore.run_rag import rag

SAMPLE_DATA = Path(__file__).resolve().parent.parent / "examples" / "sample_data"
EXAMPLE_QUERIES = (
    Path(__file__).resolve().parent.parent / "examples" / "rag" / "queries.jsonl"
)


@pytest.mark.gpu
def test_full_pipeline_runs_end_to_end(tmp_path):
    """Run process, postprocess, index and rag on a real GPU and verify outputs."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available (this test requires a GPU)")

    # Setup with data paths
    process_out = tmp_path / "process_out"
    pp_results = tmp_path / "pp_results.jsonl"
    db_path = str(tmp_path / "test.db")
    rag_out = tmp_path / "rag_output.json"
    collection = "my_docs"

    # Process
    process_cfg = {
        "data_path": str(SAMPLE_DATA),
        "google_drive_ids": [],
        "previous_results": None,
        "dispatcher_config": {
            "output_path": str(process_out),
            "use_fast_processors": False,
            "distributed": False,
            "extract_images": True,
        },
    }
    process_cfg_path = tmp_path / "process.yaml"
    with open(process_cfg_path, "w") as f:
        yaml.dump(process_cfg, f)
    process(str(process_cfg_path))

    merged = process_out / "merged" / "merged_results.jsonl"
    assert merged.exists() and merged.stat().st_size > 0

    # Postprocess
    pp_cfg = {
        "pp_modules": [
            {
                "type": "chunker",
                "args": {
                    "chunking_strategy": "sentence",
                    "table_handling": "single_row",
                },
            }
        ],
        "output": {"output_path": str(pp_results), "save_each_step": False},
    }
    pp_cfg_path = tmp_path / "pp.yaml"
    with open(pp_cfg_path, "w") as f:
        yaml.dump(pp_cfg, f)
    postprocess(str(pp_cfg_path), str(merged))

    assert pp_results.exists() and pp_results.stat().st_size > 0

    # Index
    index_cfg = {
        "indexer": {
            "dense_model": {
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "is_multimodal": False,
            },
            "sparse_model": {"model_name": "splade", "is_multimodal": False},
            "db": {"uri": db_path, "name": "my_db"},
        },
        "collection_name": collection,
        "documents_path": str(pp_results),
    }
    index_cfg_path = tmp_path / "index.yaml"
    with open(index_cfg_path, "w") as f:
        yaml.dump(index_cfg, f)
    index(config_file=str(index_cfg_path))

    # RAG
    rag_cfg = {
        "rag": {
            "llm": {
                "llm_name": "HuggingFaceTB/SmolLM2-135M-Instruct",
                "max_new_tokens": 50,
            },
            "retriever": {
                "db": {"uri": db_path, "name": "my_db"},
                "hybrid_search_weight": 0.5,
                "k": 2,
                "use_web": False,
                "reranker_model_name": None,
                "collection_name": collection,
            },
        },
        "mode": "local",
        "mode_args": {"input_file": str(EXAMPLE_QUERIES), "output_file": str(rag_out)},
    }
    rag_cfg_path = tmp_path / "rag.yaml"
    with open(rag_cfg_path, "w") as f:
        yaml.dump(rag_cfg, f)
    rag(str(rag_cfg_path))

    assert rag_out.exists() and rag_out.stat().st_size > 0
    results = json.loads(rag_out.read_text())
    assert isinstance(results, list) and len(results) > 0
    assert all("answer" in r for r in results)
