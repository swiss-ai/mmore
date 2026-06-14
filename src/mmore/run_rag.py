import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.documents import Document
from pydantic import BaseModel

from mmore.profiler import enable_profiling_from_env, profile_function
from mmore.rag.judge import extract_judge_output
from mmore.rag.pipeline import RAGConfig, RAGPipeline
from mmore.utils import load_config

RAG_EMOJI = "🧠"
logger = logging.getLogger(__name__)
logging.basicConfig(
    format=f"[RAG {RAG_EMOJI} -- %(asctime)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

load_dotenv()


@dataclass
class LocalConfig:
    input_file: str
    output_file: str


@dataclass
class APIConfig:
    endpoint: str = "/rag"
    port: int = 8000
    host: str = "0.0.0.0"


@dataclass
class RAGInferenceConfig:
    rag: RAGConfig
    mode: str
    mode_args: Optional[Union[LocalConfig, APIConfig]] = None

    def __post_init__(self):
        if self.mode_args is None and self.mode == "api":
            self.mode_args = APIConfig()


def read_queries(input_file: Union[Path, str]) -> List[Dict[str, str]]:
    with open(input_file, "r") as f:
        return [json.loads(line) for line in f]


# Standard RAG fields (always written to JSON / API)
_RAG_KEYS = ("input", "context", "answer")


def _serialize_document(doc: Union[Document, Dict[str, Any]]) -> Dict[str, Any]:
    """LangChain Documents are not JSON-serializable; clients need plain dicts."""
    if isinstance(doc, dict):
        page_content = doc.get("page_content", doc.get("content", ""))
        metadata = doc.get("metadata", {})
        return {"page_content": page_content, "metadata": dict(metadata)}
    return {"page_content": doc.page_content, "metadata": dict(doc.metadata)}


def _serialize_documents(
    docs: List[Union[Document, Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Same as _serialize_document, for the full list written to results.json / API."""
    return [_serialize_document(doc) for doc in docs]


def _to_public_output(pipeline_result: Dict[str, Any]) -> Dict[str, Any]:
    """Bridge pipeline dict (internal keys like docs) to the public RAGOutput / JSON schema."""
    out = {key: pipeline_result[key] for key in _RAG_KEYS if key in pipeline_result}
    out.update(extract_judge_output(pipeline_result))
    if pipeline_result.get("image_paths"):
        out["image_paths"] = pipeline_result["image_paths"]
    docs = pipeline_result.get("docs")
    if docs:
        out["documents"] = _serialize_documents(docs)
    return out


def save_results(results: List[Dict], output_file: Union[Path, str]):
    # JSON-safe dicts (answer, context, judge fields, documents) for JSON export.
    serialized = [_to_public_output(d) for d in results]
    with open(output_file, "w") as f:
        json.dump(serialized, f, indent=2)


class InnerInput(BaseModel):
    input: str
    collection_name: Optional[str] = None


class RAGInput(BaseModel):
    input: InnerInput


class RAGOutput(BaseModel):
    input: Optional[str] = None
    context: Optional[str] = None
    answer: Optional[str] = None
    image_paths: Optional[List[str]] = None
    documents: Optional[List[Dict[str, Any]]] = None
    judge_decision: Optional[str] = None
    judge_reason: Optional[str] = None
    judge_actions: Optional[List[str]] = None
    judge_llm_calls: Optional[int] = None
    judge_steps: Optional[List[Dict[str, Any]]] = None
    hit_max_corrective_steps: Optional[float] = None
    retrieval_metrics: Optional[Dict[str, float]] = None
    retrieval_corrections: Optional[List[Dict[str, Any]]] = None


def create_api(rag: RAGPipeline, endpoint: str):
    app = FastAPI(
        title="RAG Pipeline API",
        description="API for question answering using RAG",
        version="2.0",
    )

    # Omit unset judge fields (same rule as _to_public_output for JSON export).
    @app.post(endpoint, response_model=RAGOutput, response_model_exclude_none=True)
    async def run_rag(request: RAGInput):
        # Extract the inner input dict to pass to rag_chain
        pipeline_input = request.input.model_dump()
        output_dict = rag.rag_chain.invoke(pipeline_input)
        return RAGOutput(**_to_public_output(output_dict))

    @app.get("/health")
    def health_check():
        return {"status": "healthy"}

    return app


@profile_function()
def rag(config_file):
    """Run RAG in local or API"""
    config = load_config(config_file, RAGInferenceConfig)

    logger.info("Creating the RAG Pipeline...")
    rag_pp = RAGPipeline.from_config(config.rag)
    logger.info("RAG pipeline initialized!")

    if config.mode == "local":
        config_args = cast(LocalConfig, config.mode_args)

        queries = read_queries(config_args.input_file)
        if not queries:
            raise ValueError(
                f"No queries found in {config_args.input_file}. "
                "Add one JSON object per line, e.g. "
                '{"input": "your question", "collection_name": "my_docs"}'
            )
        n_queries = len(queries)
        logger.info(
            "Running RAG on %d queries from %s",
            n_queries,
            config_args.input_file,
        )
        results: List[Dict] = []
        try:
            for i, query in enumerate(queries, start=1):
                preview = (query.get("input") or "")[:80]
                if len(query.get("input") or "") > 80:
                    preview += "..."
                logger.info("Query %d/%d: %s", i, n_queries, preview)
                batch_result = rag_pp(query, return_dict=True)
                results.extend(batch_result)
                answer_preview = (batch_result[0].get("answer") or "")[:120]
                if len(batch_result[0].get("answer") or "") > 120:
                    answer_preview += "..."
                logger.info(
                    "Query %d/%d done. Answer: %s", i, n_queries, answer_preview
                )
        except Exception:
            logger.exception("RAG failed on query %d/%d", len(results) + 1, n_queries)
            raise
        save_results(results, config_args.output_file)
        out_path = Path(config_args.output_file)
        out_size = out_path.stat().st_size if out_path.is_file() else 0
        logger.info(
            "RAG finished: %d/%d queries answered. Wrote %s (%d bytes).",
            len(results),
            n_queries,
            config_args.output_file,
            out_size,
        )

    elif config.mode == "api":
        config_args = cast(APIConfig, config.mode_args)

        app = create_api(rag_pp, config_args.endpoint)
        uvicorn.run(app, host=config_args.host, port=config_args.port)

    else:
        raise ValueError(f"Unknown mode: {config.mode}. Should be either api or local")


if __name__ == "__main__":
    enable_profiling_from_env()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file", required=True, help="Path to the rag configuration file."
    )
    args = parser.parse_args()

    rag(args.config_file)
