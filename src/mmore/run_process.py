import time
from typing import List

from .type import MultimodalSample
from .utils import load_config

from src.mmore.process.crawler import Crawler, CrawlerConfig
from src.mmore.process.dispatcher import Dispatcher, DispatcherConfig
import yaml

overall_start_time = time.time()
import os
import argparse
import torch
import click

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

import logging
PROCESS_EMOJI = "🚀"
logger = logging.getLogger(__name__)
logging.basicConfig(format=f'[Process {PROCESS_EMOJI} -- %(asctime)s] %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

# python src/mmore/process/run_process.py ./test_data --output_result_path=/mloscratch/homes/sallinen/End2End/tmp/all.pkl

from dataclasses import dataclass

@dataclass
class ProcessInference:
    """Inference configuration."""
    data_path: str
    dispatcher_config: DispatcherConfig

def process(config_file):
    """Process documents from a directory."""
    click.echo(f'Dispatcher configuration file path: {config_file}')

    overall_start_time = time.time()

    config = load_config(config_file, ProcessInference)
        
    if config.data_path:
        data_path = config.data_path
        crawler_config = CrawlerConfig(
        root_dirs=[data_path],
        supported_extensions=[
            ".pdf", ".docx", ".pptx", ".md", ".txt",  # Document files
            ".xlsx", ".xls", ".csv",  # Spreadsheet files
            ".mp4", ".avi", ".mov", ".mkv",  # Video files
            ".mp3", ".wav", ".aac",  # Audio files
            ".eml", # Emails 
        ],
    )
    # elif isinstance(config.crawler_config, str):
    #     # TODO: Bug? crawler_config was never specified in args
    #     crawler_config = CrawlerConfig.from_yaml(args.crawler_config)
    # elif isinstance(config.crawler_config, dict):
    #     # TODO: Bug? crawler_config was never specified in args
    #     crawler_config = CrawlerConfig.from_dict(args.crawler_config)
    # else:
    #     raise ValueError("Invalid crawler configuration")
    
    logger.info(f"Using crawler configuration: {crawler_config}")
    crawler = Crawler(config=crawler_config)

    crawl_start_time = time.time()
    crawl_result = crawler.crawl()
    crawl_end_time = time.time()
    crawl_time = crawl_end_time - crawl_start_time
    #logger.info(crawl_result)
    logger.info(f"Crawling completed in {crawl_time:.2f} seconds")

    dispatcher_config = config.dispatcher_config
    
    logger.info(f"Using dispatcher configuration: {dispatcher_config}")
    dispatcher = Dispatcher(result=crawl_result, config=dispatcher_config)

    dispatch_start_time = time.time()
    results = []
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
    parser.add_argument("--config_file", required=True, help="Path to the process configuration file.")
    args = parser.parse_args()
    
    process(args.config_file)