import argparse
import json
import logging
import time
from pathlib import Path
from typing import List

import uvicorn
from fastapi import APIRouter, FastAPI
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from tqdm import tqdm

from ..utils import load_config
from .retriever import ColPaliRetriever, ColPaliRetrieverConfig

RETRIEVER_EMOJI = "🔍"
logger = logging.getLogger(__name__)
logging.basicConfig(
    format=f"[RETRIEVER {RETRIEVER_EMOJI} -- %(asctime)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def read_queries(input_file: Path) -> List[str]:
    with open(input_file, "r") as f:
        return [json.loads(line) for line in f]


def save_results(results: List[List[Document]], queries: List[str], output_file: Path):
    """Save retrieval results to a JSON file."""
    formatted_results = [
        {
            "query": query,
            "context": [
                {"page_content": doc.page_content, "metadata": doc.metadata}
                for doc in docs
            ],
        }
        for query, docs in zip(queries, results)
    ]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(formatted_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved results to {output_file}")


def retrieve(config_file: str, input_file: str, output_file: str):
    """Retrieve documents for specified queries via ColPali-based similarity search."""
    # Load the config file
    config = load_config(config_file, ColPaliRetrieverConfig)

    logger.info("Running ColPali retriever...")
    retriever = ColPaliRetriever.from_config(config)
    logger.info("Retriever loaded!")

    queries = read_queries(Path(input_file))
    logger.info(f"Loaded {len(queries)} queries from {input_file}")

    logger.info("Starting document retrieval...")
    start_time = time.time()

    retrieved_docs_for_all_queries = []

    # Call invoke for each query
    for query in tqdm(queries, desc="Retrieving documents", unit="query"):
        docs_for_query = retriever.invoke(query, k=config.top_k)
        retrieved_docs_for_all_queries.append(docs_for_query)

    end_time = time.time()
    time_taken = end_time - start_time
    logger.info(f"Document retrieval completed in {time_taken:.2f} seconds.")
    logger.info("Retrieved documents!")

    # Save results to output file
    save_results(retrieved_docs_for_all_queries, queries, Path(output_file))
    logger.info(f"Done! Results saved to {output_file}")


class RetrieverQuery(BaseModel):
    query: str = Field(..., description="Search query text")
    top_k: int = Field(default=3, ge=1, description="Number of top results to return")


def make_router(config_file: str) -> APIRouter:
    """Create API router with retriever endpoint."""
    router = APIRouter()

    # Load the config file
    config = load_config(config_file, ColPaliRetrieverConfig)

    logger.info("Running ColPali retriever...")
    retriever_obj = ColPaliRetriever.from_config(config)
    logger.info("Retriever loaded!")

    @router.post("/v1/retrieve", tags=["Retrieval"])
    def retriever(query: RetrieverQuery):
        """Query the ColPali retriever."""
        docs_for_query = retriever_obj.invoke(query.query, k=query.top_k)

        docs_info = []
        for doc in docs_for_query:
            meta = doc.metadata
            pdf_path = meta.get("pdf_path", "")
            pdf_name = Path(pdf_path).name if pdf_path else ""
            docs_info.append(
                {
                    "pdf_name": pdf_name,
                    "pdf_path": pdf_path,
                    "page_number": meta.get("page_number"),
                    "content": doc.page_content,
                    "similarity": meta.get("similarity", 0.0),
                    "rank": meta.get("rank", 0),
                }
            )

        return {"query": query.query, "results": docs_info}

    return router


def run_api(config_file: str, host: str, port: int):
    """Run the ColPali retriever API server."""
    router = make_router(config_file)

    app = FastAPI(
        title="ColPali Retriever API",
        description="""This API is based on the OpenAPI 3.1 specification. You can find out more about Swagger at [https://swagger.io](https://swagger.io).

    ## Overview

    This API defines the ColPali retriever API of mmore, handling:

    1. **Document Retrieval** - Semantic search using ColPali embeddings stored in Milvus.
    2. **PDF Page Search** - Retrieve relevant PDF pages based on query similarity.""",
        version="1.0.0",
    )
    app.include_router(router)

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retrieve documents from local Milvus database using ColPali embeddings."
    )
    parser.add_argument(
        "--config-file",
        required=True,
        help="Path to the retriever configuration file.",
    )
    parser.add_argument(
        "--input-file",
        required=False,
        type=str,
        default=None,
        help="Path to the input file of queries. If not provided, the retriever is run in API mode.",
    )
    parser.add_argument(
        "--output-file",
        required=False,
        type=str,
        default=None,
        help="Path to the output file of selected documents. Must be provided together with --input-file.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host on which the API should be run.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port on which the API should be run.",
    )
    args = parser.parse_args()

    if (args.input_file is None) != (args.output_file is None):
        parser.error(
            "Both --input-file and --output-file must be provided together or not at all."
        )

    if args.input_file:  # an input file is provided
        retrieve(args.config_file, args.input_file, args.output_file)
    else:
        run_api(args.config_file, args.host, args.port)
