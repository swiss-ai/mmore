import argparse

from typing import Literal, List, Dict, Union
from dataclasses import dataclass

from pathlib import Path
import json

import uvicorn

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from langserve import add_routes

from src.mmore.rag.pipeline import RAGPipeline, RAGConfig
from src.mmore.rag.types import MMOREOutput, MMOREInput
from src.mmore.utils import load_config

import logging
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv() 


@dataclass
class LocalConfig:
    input_file: str
    output_file: str

@dataclass
class APIConfig:
    endpoint: str = '/rag'
    port: int = 8000
    host: str = '0.0.0.0'
    
@dataclass
class RAGInferenceConfig:
    rag: RAGConfig
    mode: str
    mode_args: Union[LocalConfig, APIConfig] = None

    def __post_init__(self):
        if self.mode_args is None and self.mode == 'api':
            self.mode_args = APIConfig()

def get_args():
    parser = argparse.ArgumentParser(description='Run RAG pipeline with API or CLI mode')
    parser.add_argument('--config-file', type=str, required=True, help='Path to a config file')
    return parser.parse_args()


def read_queries(input_file: Path) -> List[str]:
    with open(input_file, 'r') as f:
        return [json.loads(line) for line in f]

def save_results(results: List[Dict], output_file: Path):
    results = [
        {key: d[key] for key in {'input', 'context', 'answer'} if key in d} 
        for d in results
    ]   
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

def create_api(rag: RAGPipeline, endpoint: str):
    app = FastAPI(
        title="RAG Pipeline API",
        description="API for question answering using RAG",
        version="1.0",
    )

    # Add routes for the RAG chain
    add_routes(
        app,
        rag.rag_chain.with_types(input_type=MMOREInput, output_type=MMOREOutput),
        path=endpoint,
        playground_type="chat"
    )

    @app.get("/health")
    def health_check():
        return {"status": "healthy"}

    return app

if __name__ == "__main__":
    args = get_args()

    config = load_config(args.config_file, RAGInferenceConfig)
    
    logger.info('Creating the RAG Pipeline...')
    rag = RAGPipeline.from_config(config.rag)
    logger.info('RAG pipeline initialized!')

    if config.mode == 'local':
        queries = read_queries(config.mode_args.input_file)
        results = rag(queries, return_dict=True)
        save_results(results, config.mode_args.output_file)
    elif config.mode == 'api':
        app = create_api(rag, config.mode_args.endpoint)
        uvicorn.run(app, host=config.mode_args.host, port=config.mode_args.port)
    else:
        raise ValueError(f"Unknown inference mode: {config.mode}. Should be in [api, local]")