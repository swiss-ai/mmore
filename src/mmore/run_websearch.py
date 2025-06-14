# mmore/run_websearch.py

import argparse
import logging
import time
import torch
from dataclasses import dataclass
from typing import Any, Dict


from .websearchRAG.logging_config import logger  # Import the shared logger

from .utils import load_config
from .websearchRAG.config import WebsearchConfig
from .websearchRAG.pipeline import WebsearchPipeline



# WEBSRCH_EMOJI = "🌐"
# logger = logging.getLogger(__name__)
# logging.basicConfig(
#     format=f"[WebSearch {WEBSRCH_EMOJI} -- %(asctime)s] %(message)s",
#     level=logging.INFO,
#     datefmt="%Y-%m-%d %H:%M:%S",
# )

# CUDA tweaks (as before)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


@dataclass
class WebsearchSection:
    use_rag: bool
    rag_config_path: str
    use_summary: bool
    n_subqueries : int
    input_file: str
    input_queries: str
    output_file: str
    n_loops: int
    max_searches: int
    llm_config: Dict[str, Any]


@dataclass
class WebsearchAppConfig:
    websearch: WebsearchSection


def run_websearch(config_file: str):
    """
    Run the Websearch (+ optional RAG) pipeline according to the YAML at config_file.
    """
    logger.info(f"Websearch configuration file: {config_file}")

    # 1) Load and parse the YAML using the wrapper
    app_cfg = load_config(config_file, WebsearchAppConfig)
    ws = app_cfg.websearch
    logger.info(f"Parsed Websearch section: {ws}")

    # 2) Map to the pipeline's config dict
    web_cfg_dict = {
        "use_rag": ws.use_rag,
        "rag_config_path": ws.rag_config_path,
        "rag_summary": ws.use_summary,
        "input_file": ws.input_file,
        "input_queries": ws.input_queries,
        "output_file": ws.output_file,
        "n_subqueries": ws.n_subqueries,
        "n_loops": ws.n_loops,
        "max_searches": ws.max_searches,
        "llm_config": ws.llm_config,
    }
    web_cfg = WebsearchConfig.from_dict(web_cfg_dict)
    logger.info(f"Using WebsearchConfig: {web_cfg}")

    # 3) Instantiate and run
    pipeline = WebsearchPipeline(config=web_cfg)
    start = time.time()
    pipeline.run()
    logger.info(f"Websearch pipeline completed in {time.time() - start:.2f} seconds.")


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
