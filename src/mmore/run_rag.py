import argparse

from typing import Literal, List, Dict, Union
from dataclasses import dataclass

from pathlib import Path
import json

import uvicorn

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from langserve import add_routes

from .rag.pipeline import RAGPipeline, RAGConfig
from .rag.types import MMOREOutput, MMOREInput
from .utils import load_config

import logging
RAG_EMOJI = "🧠"
logger = logging.getLogger(__name__)
logging.basicConfig(format=f'[RAG {RAG_EMOJI} -- %(asctime)s] %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

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

def read_queries(input_file: Path) -> List[Union[str, dict]]:
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


def rag(config_file):
    """Run RAG."""
    config = load_config(config_file, RAGInferenceConfig)
    
    logger.info('Creating the RAG Pipeline...')
    rag = RAGPipeline.from_config(config.rag)
    logger.info('RAG pipeline initialized!')

    if config.mode == 'local':
        # Read Queries
        queries = read_queries(Path(config.mode_args.input_file))

        # Pass them to the pipeline
        results = rag(queries, return_dict=True)

        # Write results out
        save_results(results, config.mode_args.output_file)
        
    elif config.mode == 'api':
        app = create_api(rag, config.mode_args.endpoint)
        uvicorn.run(app, host=config.mode_args.host, port=config.mode_args.port)
    else:
        raise ValueError(f"Unknown inference mode: {config.mode}. Should be in [api, local]")
    

if __name__ == '__main__':
    rag()