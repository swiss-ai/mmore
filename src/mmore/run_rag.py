import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union, cast

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

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


def save_results(results: List[Dict], output_file: Union[Path, str]):
    results = [
        {key: d[key] for key in {"input", "context", "answer"} if key in d}
        for d in results
    ]
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)


class InnerInput(BaseModel):
    input: str
    collection_name: Optional[str] = None


class RAGInput(BaseModel):
    input: InnerInput


class RAGOutput(BaseModel):
    input: Optional[str] = None
    context: Optional[str] = None
    answer: Optional[str] = None


def create_api(rag: RAGPipeline, endpoint: str):
    app = FastAPI(
        title="RAG Pipeline API",
        description="API for question answering using RAG",
        version="2.0",
    )

    @app.post(endpoint, response_model=RAGOutput)
    async def run_rag(request: RAGInput):
        # Extract the inner input dict to pass to rag_chain
        pipeline_input = request.input.model_dump()
        output_dict = rag.rag_chain.invoke(pipeline_input)
        return RAGOutput(**output_dict)

    @app.get("/health")
    def health_check():
        return {"status": "healthy"}

    return app


def rag(config_file):
    """Run RAG in local or API"""
    config = load_config(config_file, RAGInferenceConfig)

    logger.info("Creating the RAG Pipeline...")
    rag_pp = RAGPipeline.from_config(config.rag)
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file", required=True, help="Path to the rag configuration file."
    )
    args = parser.parse_args()

    rag(args.config_file)
