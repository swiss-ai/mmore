import os
import yaml
import torch
import logging
import warnings
import argparse
from typing import List

from src.mmore.process.crawler import Crawler, CrawlerConfig
from src.mmore.process.dispatcher import Dispatcher, DispatcherConfig
from src.mmore.type import MultimodalSample

# Configure torch backend
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log", mode="a")
    ]
)

logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("pypdfium2").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Process documents from a directory")
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Dispatcher configuration file path"
    )

    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    logger.info(os.path.abspath(__file__))
    crawler = Crawler(config=load_crawler_config(config))
    crawl_result = crawler.crawl()

    dispatcher = Dispatcher(result=crawl_result, config=load_dispatcher_config(config))
    results = list(dispatcher())

    logger.info(f"Processed {len(results)} files")
    save_merged_results(results, config)

def load_crawler_config(config: dict) -> CrawlerConfig:
    if config.get("data_path"):
        data_path = config.get("data_path")
        return CrawlerConfig(
            root_dirs=[data_path],
            supported_extensions=[
                ".pdf", ".docx", ".pptx", ".md", ".txt",
                ".xlsx", ".xls", ".csv",
                ".mp4", ".avi", ".mov", ".mkv",
                ".mp3", ".wav", ".aac",
                ".eml",
            ]
        )
    elif isinstance(config.get("crawler_config"), str):
        return CrawlerConfig.from_yaml(config.get("crawler_config"))
    elif isinstance(config.get("crawler_config"), dict):
        return CrawlerConfig.from_dict(config.get("crawler_config"))
    else:
        raise ValueError("Invalid crawler configuration.")

def load_dispatcher_config(config: dict) -> DispatcherConfig:
    if isinstance(config.get("dispatcher_config"), str):
        return DispatcherConfig.from_yaml(config.get("dispatcher_config"))
    elif isinstance(config.get("dispatcher_config"), dict):
        return DispatcherConfig.from_dict(config.get("dispatcher_config"))
    else:
        logger.warning("Using default dispatcher configuration: use_fast_processors=True, distributed=False")
        return DispatcherConfig(use_fast_processors=True, distributed=False)

def save_merged_results(results: List[List[MultimodalSample]], config: dict) -> None:
    output_path = config.get("dispatcher_config", {}).get("output_path")
    if not output_path:
        return

    merged_output_path = os.path.join(output_path, "merged")
    os.makedirs(merged_output_path, exist_ok=True)

    output_file = os.path.join(merged_output_path, f"merged_results_{os.getpid()}.jsonl")
    # merged_results = ProcessorResult.merge(results)
    # merged_results.to_jsonl(output_file)
    for res in results:
        MultimodalSample.to_jsonl(output_file, res)

    logger.info(f"Merged results saved to {output_file}")

if __name__ == "__main__":
    main()