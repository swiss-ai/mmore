import argparse
import concurrent.futures
from dataclasses import dataclass
import json
import logging
import time
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from fastapi import FastAPI, APIRouter
from pydantic import BaseModel, Field
import uvicorn

from ..utils import load_config
from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.utils.torch_utils import ListDataset
from torch.utils.data import DataLoader

from milvuscolpali import MilvusColpaliManager

RETRIEVER_EMOJI = "🔍"
logger = logging.getLogger(__name__)
logging.basicConfig(
    format=f"[RETRIEVER {RETRIEVER_EMOJI} -- %(asctime)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

@dataclass
class RetrieverConfig:
    db_path: str = "./milvus_data"
    collection_name: str = "pdf_pages"
    model_name: str = "vidore/colpali-v1.3"
    mode: str = "api" # "api", "batch", or "single"
    host: str = "0.0.0.0"
    port: int = 8001
    input_file: Optional[str] = None
    output_file: Optional[str] = None
    query: Optional[str] = None
    top_k: int = 3
    dim: int = 128
    metric_type: str = "IP"

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_name: str, device: str):
    logger.info(f"Loading ColPali model: {model_name}")
    model = ColPali.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ).eval()
    processor = ColPaliProcessor.from_pretrained(model_name)
    return model, processor


def embed_queries(texts: List[str], model, processor) -> List[np.ndarray]:
    dataloader = DataLoader(
        dataset=ListDataset[str](texts),
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: processor.process_queries(x),
    )

    vectors = []
    for batch_query in dataloader:
        with torch.no_grad():
            batch_query = {k: v.to(model.device) for k, v in batch_query.items()}
            emb = model(**batch_query)
            vectors.extend(list(torch.unbind(emb.to("cpu"))))
    return [v.float().numpy() for v in vectors]


def query_indexer(
    query: str,
    model,
    processor,
    manager: MilvusColpaliManager,
    config: RetrieverConfig,
    top_k: int = 3,
):
    """
    Embed the query using ColPali and search the local Milvus database.
    """
    vecs = embed_queries([query], model, processor)[0]
    results = manager.search_embeddings(vecs, top_k=top_k)
    return results

class RetrieverQuery(BaseModel):
    query: str = Field(..., description="Search query text")
    top_k: int = Field(3, description="Number of top matches to return")


def make_router(model, processor, manager, config):
    router = APIRouter()

    @router.post("/v1/retrieve", tags=["Retrieval"])
    def retrieve_docs(request: RetrieverQuery):
        matches = query_indexer(request.query, model, processor, manager, config, top_k=request.top_k)
        return {"query": request.query, "results": matches}

    return router

def read_queries_from_file(input_file: Path) -> List[str]:
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if content.startswith("["):
            queries = json.loads(content)
        else:
            queries = [json.loads(line) for line in content.splitlines()]
    if isinstance(queries[0], dict) and "query" in queries[0]:
        queries = [q["query"] for q in queries]
    return queries


def save_results_to_file(results: List[dict], output_file: Path):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved results to {output_file}")

def run_api(config: RetrieverConfig):
    device = get_device()
    model, processor = load_model(config.model_name, device)

    manager = MilvusColpaliManager(
        db_path=config.db_path,
        collection_name=config.collection_name,
        dim=config.dim,
        metric_type=config.metric_type,
        create_collection=False,
    )

    app = FastAPI(
        title="ColPali Retriever API (Local Milvus)",
        description="Retrieve documents using ColPali embeddings stored locally in Milvus.",
        version="1.0.0",
    )
    app.include_router(make_router(model, processor, manager, config))
    uvicorn.run(app, host=config.host, port=config.port)


def run_batch(config: RetrieverConfig):
    device = get_device()
    model, processor = load_model(config.model_name, device)

    manager = MilvusColpaliManager(
        db_path=config.db_path,
        collection_name=config.collection_name,
        dim=config.dim,
        metric_type=config.metric_type,
        create_collection=False,
    )

    queries = read_queries_from_file(Path(config.input_file))
    logger.info(f"Loaded {len(queries)} queries from {config.input_file}")

    all_results = []
    start = time.time()

    for query in queries:
        matches = query_indexer(query, model, processor, manager, config, top_k=config.top_k)
        all_results.append({"query": query, "results": matches})

    elapsed = time.time() - start
    logger.info(f"Processed {len(queries)} queries in {elapsed:.2f} seconds.")
    save_results_to_file(all_results, Path(config.output_file))


def run_single_query(config: RetrieverConfig):
    device = get_device()
    model, processor = load_model(config.model_name, device)

    manager = MilvusColpaliManager(
        db_path=config.db_path,
        collection_name=config.collection_name,
        dim=config.dim,
        metric_type=config.metric_type,
        create_collection=False,
    )

    results = query_indexer(config.query, model, processor, manager, config, top_k=config.top_k)
    for r in results:
        print(f"{r['rank']}. {r['pdf_name']} (page {r['page_number']}) — score={r['score']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retrieve documents from local Milvus database using ColPali embeddings."
    )
    parser.add_argument("--config_file", type=str, required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    config = load_config(args.config_file, RetrieverConfig)

    if config.mode == "batch":
        run_batch(config)
    elif config.mode == "single":
        run_single_query(config)
    else:
        run_api(config)
