# mmore/run_websearch.py

import argparse
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict

import torch

from .websearch.config import WebsearchConfig
from .websearch.pipeline import WebsearchPipeline
from .utils import load_config

WEBSRCH_EMOJI = "🌐"
logger = logging.getLogger(__name__)
logging.basicConfig(
    format=f"[WebSearch {WEBSRCH_EMOJI} -- %(asctime)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Disable certain CUDA optimizations for stability (same as in run_process.py)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


@dataclass
class WebsearchInference:
    """
    Inference configuration for Websearch (+ optional RAG) pipeline.
    This should mirror the keys under your top-level 'websearch:' in the YAML.
    """
    rag_config_path: str         # path to RAG config (e.g. "../rag/config.yaml")
    use_rag: bool                # whether to run RAG first
    rag_summary: bool            # whether to summarize the RAG answer
    input_file: str              # path to JSON input (either queries or RAG output)
    output_file: str             # path to write final JSON results
    n_subqueries: int            # number of sub-queries to generate
    max_searches: int            # max DuckDuckGo hits per sub-query
    llm_config: Dict[str, Any]   # passed into LLMConfig (llm_name, max_new_tokens, temperature, etc.)

    # If you add more fields under `websearch:` in your YAML, add them here.


def run_websearch(config_file: str):
    """
    Run the Websearch (+ optional RAG) pipeline according to the YAML at config_file.

    1) Load WebsearchInference via load_config.
    2) Convert that to a WebsearchConfig.
    3) Instantiate and run WebsearchPipeline.
    """
    click_msg = f"Websearch configuration file: {config_file}"
    logger.info(click_msg)

    start_time = time.time()

    # 1) Load the YAML into our dataclass
    cfg: WebsearchInference = load_config(config_file, WebsearchInference)

    # 2) Build WebsearchConfig from our dataclass
    #    WebsearchConfig.from_dict expects a dict with exactly the same keys, so:
    web_cfg = WebsearchConfig.from_dict({
        "rag_config_path": cfg.rag_config_path,
        "use_rag": cfg.use_rag,
        "rag_summary": cfg.rag_summary,
        "input_file": cfg.input_file,
        "output_file": cfg.output_file,
        "n_subqueries": cfg.n_subqueries,
        "max_searches": cfg.max_searches,
        "llm_config": cfg.llm_config,
    })

    logger.info(f"Using WebsearchConfig: {web_cfg}")

    # 3) Instantiate and run
    pipeline = WebsearchPipeline(config=web_cfg)

    pipeline_start = time.time()
    pipeline.run()
    pipeline_end = time.time()

    elapsed = pipeline_end - pipeline_start
    logger.info(f"Websearch pipeline completed in {elapsed:.2f} seconds.")

    total_elapsed = time.time() - start_time
    logger.info(f"Total execution time: {total_elapsed:.2f} seconds.")


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
