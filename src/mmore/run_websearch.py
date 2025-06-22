# mmore/run_websearch.py

import argparse
import logging
import time
import torch
from dataclasses import dataclass, asdict, fields
from typing import Any, Dict, Optional, Union, cast

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from langserve import add_routes
from dotenv import load_dotenv

from .websearchRAG.logging_config import logger
from .utils import load_config
from .websearchRAG.config import WebsearchConfig
from .websearchRAG.pipeline import WebsearchPipeline
from .rag.pipeline import RAGConfig, RAGPipeline
from .run_rag import LocalConfig, APIConfig, RAGInferenceConfig

from .run_rag import create_api as create_api_rag


from .websearchRAG.logging_config import logger  # Import the shared logger


#à quoi ça sert?
load_dotenv()


# CUDA tweaks for best perf
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)



@dataclass
class WebsearchInferenceConfig:
    websearch: WebsearchConfig
    mode_args: Optional[Union[LocalConfig, APIConfig]] = None

    def __post_init__(self):
        if self.websearch.mode == "api" and self.mode_args is None:
            self.mode_args = APIConfig()


def run_websearch(config_file):

    # 1) Load config
    cfg = load_config(config_file, WebsearchInferenceConfig)
    ws = cfg.websearch
    if ws.mode == "local":
        pipeline = WebsearchPipeline(config=ws)
        logger.info("Running Websearch pipeline in LOCAL mode...")
        start = time.time()
        pipeline.run()
        logger.info(f"Completed in {time.time() - start:.2f}s")

    elif ws.mode == "api":
        logger.info("Starting Websearch pipeline in API mode...")
        app = create_api(cfg)
        uvicorn.run(app, host="0.0.0.0", port=8000)

    else:
        raise ValueError(f"Unknown mode: {cfg.mode!r}. Must be 'local' or 'api'.")



class QueryInput(BaseModel):
    input: str = Field(..., description="The user query")
    collection_name: Optional[str] = Field(
        None, description="The collection to search (optional)"
    )

class WebQuery(BaseModel):
    query: QueryInput = Field(
        ...,
        description="Search query with input and optional collection name"
    )
    use_rag: bool = Field(
        False,
        description="Include RAG context",
        example=True
    )
    use_summary: bool = Field(
        True,
        description="Enable subquery summary",
        example=False
    )

     



def create_api(config_file: str):
    app = FastAPI(
        title="mmore Websearch API",
        description="""This API is based on the OpenAPI 3.1 specification. You can find out more about Swagger at [https://swagger.io](https://swagger.io).

## Overview

This API defines the retriever API of mmore, handling:

1. **File Operations** - Direct file management within mmore.
2. **Rag and websearch** - Search based on the query/documents.""",
        version="1.0.0",
    )

    logger.info("Websearch loaded!")

    @app.post("/websearch")
    # query = query parameter
    def websearch(query: WebQuery):
        #charge la pipeline directement depuis rag_pp
        #changer le config_file avec le config file du rag --> ajouter ce que l'utilisateur demande
        pipeline = WebsearchPipeline(config=config_file.websearch)

        if query.use_rag:
            logger.info("Launch RAG")
            config_RAG = load_config(config_file.websearch.rag_config_path, RAGInferenceConfig)
            logger.info("Creating the RAG Pipeline...")
            rag_pp = RAGPipeline.from_config(config_RAG.rag)
            data = rag_pp([query.query.dict()], return_dict=True)
            logger.info("RAG done")
            logger.info("##RAG##", data)
        else: 
            data = query.query

        logger.info("Launch websearch")

        answers = pipeline.run_api(query.use_rag, query.use_summary, data)
        logger.info("Websearch done")


        return answers

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Websearch (+ optional RAG) pipeline.")
    parser.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="Path to the Websearch configuration file (YAML)."
    )
    args = parser.parse_args()
    run_websearch(args.config_file)




# {
#   "query": {
#     "input": "When was Barack Obama born?",
#     "collection_name": "my_docs"
#   },
#   "use_rag": true,
#   "use_summary": true
# }