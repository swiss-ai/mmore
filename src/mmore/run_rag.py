from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Literal, List, Dict, Union
import argparse
import json
import logging
import uvicorn

RAG_EMOJI = "🧠"
logger = logging.getLogger(__name__)
logging.basicConfig(format=f'[RAG {RAG_EMOJI} -- %(asctime)s] %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

from mmore.rag.pipeline import RAGPipeline, RAGConfig
from mmore.rag.types import MMOREOutput, MMOREInput
from mmore.utils import load_config

load_dotenv() 


class LocalConfig(BaseModel):
    input_file: str
    output_file: str

class APIConfig(BaseModel):
    endpoint: str = Field(default='/rag')
    port: int = Field(default=8000)
    host: str = Field(default='0.0.0.0')

class LocalRAGInferenceConfig(BaseModel):
    rag: RAGConfig
    mode: Literal["local"]
    mode_args: LocalConfig

class APIRAGInferenceConfig(BaseModel):
    rag: RAGConfig
    mode: Literal["api"]
    mode_args: APIConfig = Field(default_factory=lambda: APIConfig())

RAGInferenceConfig = Union[LocalRAGInferenceConfig, APIRAGInferenceConfig]

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


def rag(config_file):
    """Run RAG."""
    config = load_config(config_file, RAGInferenceConfig)
    
    logger.info('Creating the RAG Pipeline...')
    rag_pp = RAGPipeline.from_config(config.rag)
    logger.info('RAG pipeline initialized!')

    if config.mode == 'local':
        queries = read_queries(config.mode_args.input_file)
        results = rag_pp(queries, return_dict=True)
        save_results(results, config.mode_args.output_file)
    elif config.mode == 'api':
        app = create_api(rag_pp, config.mode_args.endpoint)
        uvicorn.run(app, host=config.mode_args.host, port=config.mode_args.port)
    else:
        raise ValueError(f"Unknown inference mode: {config.mode}. Should be in [api, local]")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True, help="Path to the rag configuration file.")
    args = parser.parse_args()

    rag(args.config_file)