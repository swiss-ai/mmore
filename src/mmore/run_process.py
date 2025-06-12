import argparse
import logging
import os
import time
from dataclasses import dataclass

import click
import torch

from mmore.dashboard.backend.client import DashboardClient
from mmore.process.crawler import Crawler, CrawlerConfig
from mmore.process.dispatcher import Dispatcher, DispatcherConfig
from mmore.type import MultimodalSample
from mmore.utils import load_config

PROCESS_EMOJI = "🚀"
logger = logging.getLogger(__name__)
logging.basicConfig(
    format=f"[Process {PROCESS_EMOJI} -- %(asctime)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

overall_start_time = time.time()

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


@dataclass
class ProcessInference:
    """Inference configuration."""

    data_path: str
    dispatcher_config: DispatcherConfig


def process(config_file: str):
    """Process documents from a directory."""
    click.echo(f"Dispatcher configuration file path: {config_file}")

    overall_start_time = time.time()

    config: ProcessInference = load_config(config_file, ProcessInference)

    if config.data_path:
        data_path = config.data_path
        crawler_config = CrawlerConfig(
            root_dirs=[data_path],
            supported_extensions=[
                ".pdf",
                ".docx",
                ".pptx",
                ".md",
                ".txt",  # Document files
                ".xlsx",
                ".xls",
                ".csv",  # Spreadsheet files
                ".mp4",
                ".avi",
                ".mov",
                ".mkv",  # Video files
                ".mp3",
                ".wav",
                ".aac",  # Audio files
                ".eml",  # Emails
            ],
            output_path=config.dispatcher_config.output_path,
        )
    else:
        raise ValueError("Data path not provided in the configuration")

    logger.info(f"Using crawler configuration: {crawler_config}")
    crawler = Crawler(config=crawler_config)

    crawl_start_time = time.time()
    crawl_result = crawler.crawl()
    crawl_end_time = time.time()
    crawl_time = crawl_end_time - crawl_start_time
    logger.info(f"Crawling completed in {crawl_time:.2f} seconds")

    dispatcher_config: DispatcherConfig = config.dispatcher_config

    url = dispatcher_config.dashboard_backend_url
    DashboardClient(url).init_db(len(crawl_result))

    logger.info(f"Using dispatcher configuration: {dispatcher_config}")
    dispatcher = Dispatcher(result=crawl_result, config=dispatcher_config)

    dispatch_start_time = time.time()
    results = list(dispatcher())

    dispatch_end_time = time.time()
    dispatch_time = dispatch_end_time - dispatch_start_time
    logger.info(f"Dispatching and processing completed in {dispatch_time:.2f} seconds")

    output_path = config.dispatcher_config.output_path
    merged_output_path = os.path.join(output_path, "merged")
    output_file = os.path.join(merged_output_path, "merged_results.jsonl")

    os.makedirs(merged_output_path, exist_ok=True)
    for res in results:
        MultimodalSample.to_jsonl(output_file, res)

    logger.info(f"Merged results ({len(results)} items) saved to {output_file}")

    overall_end_time = time.time()
    overall_time = overall_end_time - overall_start_time
    logger.info(f"Total execution time: {overall_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the processing pipeline.")
    parser.add_argument(
        "--config_file", required=True, help="Path to the process configuration file."
    )
    args = parser.parse_args()

    process(args.config_file)
