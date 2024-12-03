import pypdfium2
import time
from typing import List
from src.mmore.process.crawler import Crawler, CrawlerConfig
from src.mmore.process.dispatcher import Dispatcher, DispatcherConfig
import yaml
from src.mmore.process.processors.processor import ProcessorResult

overall_start_time = time.time()
import os
import argparse
import torch
import click

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

import logging

logger = logging.getLogger(__name__)
# block log from libraries
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("pypdfium2").setLevel(logging.ERROR)
logging.getLogger("pypdfium2").setLevel(logging.ERROR)


from contextlib import contextmanager
import warnings
import sys

@contextmanager
def suppress_warnings_and_stdout():
    # Suppress specific warnings
    warnings.filterwarnings('ignore', category=FutureWarning, 
                          message="The input name `inputs` is deprecated*")
    
    pypdfium_message = "-> Cannot close object, library is destroyed. This may cause a memory leak!*"
    # Redirect stdout to devnull to catch pypdfium messages
    old_stdout = sys.stdout
    devnull = open(os.devnull, 'w')
    # Suppress pypdfium warnings
    sys.stdout = devnull
    
    try:
        sys.stdout = devnull
        yield
        
    finally:
        sys.stdout = old_stdout
        devnull.close()

with suppress_warnings_and_stdout():
    pass


# python src/mmore/process/run_process.py ./test_data --output_result_path=/mloscratch/homes/sallinen/End2End/tmp/all.pkl

def main():
    parser = argparse.ArgumentParser(description='Process documents from a directory')
    
    # Add arguments
    
    parser.add_argument('--config_file', type=str, 
                      help='Dispatcher configuration file path',
                      required=True)


    args = parser.parse_args()
    overall_start_time = time.time()

    with open(args.config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    if config.get("data_path"):
        data_path = config.get("data_path")
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
    elif isinstance(config.get("crawler_config"), str):
        crawler_config = CrawlerConfig.from_yaml(args.crawler_config)
    elif isinstance(config.get("crawler_config"), dict):
        crawler_config = CrawlerConfig.from_dict(args.crawler_config)
    else:
        raise ValueError("Invalid crawler configuration")
    
    logger.info(f"Using crawler configuration: {crawler_config}")
    crawler = Crawler(config=crawler_config)

    crawl_start_time = time.time()
    crawl_result = crawler.crawl()
    crawl_end_time = time.time()
    crawl_time = crawl_end_time - crawl_start_time
    logger.info(crawl_result)
    logger.info(f"Crawling completed in {crawl_time:.2f} seconds")

    if isinstance(config.get("dispatcher_config") , str):
        dispatcher_config = DispatcherConfig.from_yaml(config.get("dispatcher_config"))
    elif isinstance(config.get("dispatcher_config") , dict):
        dispatcher_config = DispatcherConfig.from_dict(config.get("dispatcher_config"))
    else:
        logger.warning("Using default dispatcher configuration")
        dispatcher_config = DispatcherConfig(
            use_fast_processors=True,
            distributed=False,
        )
    
    logger.info(f"Using dispatcher configuration: {dispatcher_config}")
    dispatcher = Dispatcher(result=crawl_result, config=dispatcher_config)

    dispatch_start_time = time.time()
    results = []
    results = list(dispatcher())
    
    dispatch_end_time = time.time()
    dispatch_time = dispatch_end_time - dispatch_start_time
    logger.info(f"Dispatching and processing completed in {dispatch_time:.2f} seconds")

    overall_end_time = time.time()
    overall_time = overall_end_time - overall_start_time
        
    logger.info(f"\nTotal execution time: {overall_time:.2f} seconds")

    def save_merged_results(results: List[ProcessorResult]) -> None:
        if not config.get("dispatcher_config").get("output_path"):
            return
        output_path = config.get("dispatcher_config").get("output_path")
        merged_output_path = os.path.join(output_path, "merged")
        os.makedirs(merged_output_path, exist_ok=True)
        output_file = os.path.join(merged_output_path, "merged_results.jsonl")
        
        merged_results = ProcessorResult.merge(results)
        merged_results.to_jsonl(output_file)
        logger.info(f"Merged results saved to {output_file}")
    
    save_merged_results(results)
    
    
if __name__ == "__main__":
    with suppress_warnings_and_stdout():
        main()