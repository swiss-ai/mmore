# mmore/run_websearch.py

import argparse
import logging
import time
import torch
from dataclasses import dataclass, asdict, fields
from typing import Any, Dict, Optional, Union, cast

import uvicorn
from fastapi import FastAPI
from langserve import add_routes

from .websearchRAG.logging_config import logger
from .utils import load_config
from .websearchRAG.config import WebsearchConfig
from .websearchRAG.pipeline import WebsearchPipeline
from .run_rag import LocalConfig, APIConfig, create_api


from .websearchRAG.logging_config import logger  # Import the shared logger


# CUDA tweaks for best perf
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


@dataclass
class WebsearchSection:
    use_rag: bool
    rag_config_path: str
    use_summary: bool
    n_subqueries: int
    input_file: str
    input_queries: str
    output_file: str
    n_loops: int
    max_searches: int
    llm_config: Dict[str, Any]
    mode: str


@dataclass
class WebsearchInferenceConfig:
    websearch: WebsearchSection
    mode: str   = "local"                   # "local" or "api"
    mode_args: Optional[Union[LocalConfig, APIConfig]] = None

    def __post_init__(self):
        if self.mode == "api" and self.mode_args is None:
            self.mode_args = APIConfig()


def build_pipeline(ws: WebsearchSection) -> WebsearchPipeline:
    ws_dict = asdict(ws)
    config_fields = {f.name for f in fields(WebsearchConfig)}
    filtered_dict = {k: v for k, v in ws_dict.items() if k in config_fields}
    web_cfg = WebsearchConfig(**filtered_dict)
    return WebsearchPipeline(config=web_cfg)

def run_websearch(config_file):

    # 1) Load config
    cfg = load_config(config_file, WebsearchInferenceConfig)
    ws  = cfg.websearch
    #logger.info("Configuration file", ws)
    if not cfg.mode:
        raise ValueError("Configuration is missing the 'mode' field. Ensure it is set to 'local' or 'api'.")


    # 2) Build pipeline once
    # web_cfg = WebsearchConfig(**asdict(ws))
    # pipeline = WebsearchPipeline(config=web_cfg)
    pipeline = build_pipeline(ws)

    # 3) Dispatch on mode
    if cfg.mode == "local":
        logger.info("Running Websearch pipeline in LOCAL mode...")
        start = time.time()
        pipeline.run()
        logger.info(f"Completed in {time.time() - start:.2f}s")

    elif cfg.mode == "api":
        logger.info("Starting Websearch pipeline in API mode...")
        # wrap pipeline.__call__ via create_api
        app: FastAPI = create_api(pipeline, cfg.mode_args.endpoint)

        @app.get("/health")
        def _health():
            return {"status": "healthy"}

        uvicorn.run(
            app,
            host=cfg.mode_args.host,
            port=cfg.mode_args.port,
            log_level="info",
        )

    else:
        raise ValueError(f"Unknown mode: {cfg.mode!r}. Must be 'local' or 'api'.")


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