import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union, cast

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from langserve import add_routes

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


def create_api(rag: RAGPipeline, endpoint: str):
    app = FastAPI(
        title="RAG Pipeline API",
        description="API for question answering using RAG",
        version="1.0",
    )

    # Add routes for the RAG chain
    add_routes(app, rag.rag_chain, path=endpoint, playground_type="chat")

    @app.get("/health")
    def health_check():
        return {"status": "healthy"}

    return app


def rag(config_file):
    """Run RAG."""
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
        raise ValueError(
            f"Unknown inference mode: {config.mode}. Should be in [api, local]"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file", required=True, help="Path to the rag configuration file."
    )
    args = parser.parse_args()

    rag(args.config_file)
