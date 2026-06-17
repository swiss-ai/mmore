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
from mmore.rag.pipeline import PRIVACY_OUTPUT_KEYS, RAGConfig, RAGPipeline
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
    # Privacy mode surfaces a PII-free report record + advisory summary, if present.
    for key in PRIVACY_OUTPUT_KEYS:
        if key in pipeline_result:
            out[key] = pipeline_result[key]
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
    documents: Optional[List[Dict[str, Any]]] = None
    judge_decision: Optional[str] = None
    judge_reason: Optional[str] = None
    judge_actions: Optional[List[str]] = None
    judge_llm_calls: Optional[int] = None
    judge_steps: Optional[List[Dict[str, Any]]] = None
    hit_max_corrective_steps: Optional[float] = None
    retrieval_metrics: Optional[Dict[str, float]] = None
    retrieval_corrections: Optional[List[Dict[str, Any]]] = None
    # Privacy mode only: PII-free report record and advisory type+count summary.
    privacy_report: Optional[Dict[str, Any]] = None
    privacy_warnings: Optional[List[Dict[str, Any]]] = None


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


def _build_privacy_graph(privacy_config_file: str):
    """Load + validate the privacy config and compile its pipeline graph.

    Errors (missing answer.llm, unknown domain, unregistered engine) surface
    here, eagerly, before any query runs.
    """
    from langgraph.checkpoint.memory import MemorySaver

    from mmore.privacy.pipeline import build_privacy_pipeline
    from mmore.privacy.runner import load_privacy_config

    privacy_config = load_privacy_config(privacy_config_file)
    # One checkpointer for the graph; the runner threads each request by id.
    return build_privacy_pipeline(privacy_config, MemorySaver())


@profile_function()
def rag(config_file, privacy_config_file: Optional[str] = None):
    """Run RAG in local or API, optionally through the privacy pipeline."""
    config = load_config(config_file, RAGInferenceConfig)

    privacy_graph = None
    if privacy_config_file is not None:
        logger.info("Privacy mode enabled, building the privacy pipeline...")
        privacy_graph = _build_privacy_graph(privacy_config_file)

    logger.info("Creating the RAG Pipeline...")
    rag_pp = RAGPipeline.from_config(config.rag, privacy_graph=privacy_graph)
    logger.info("RAG pipeline initialized!")

    if config.mode == "local":
        config_args = cast(LocalConfig, config.mode_args)

        queries = read_queries(config_args.input_file)
        results = rag_pp(queries, return_dict=True)
        save_results(results, config_args.output_file)

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
    parser.add_argument(
        "--privacy",
        default=None,
        help="Path to a privacy config; its presence enables privacy mode.",
    )
    args = parser.parse_args()

    rag(args.config_file, privacy_config_file=args.privacy)
