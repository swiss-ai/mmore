import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Union

import click
import torch

from mmore.dashboard.backend.client import DashboardClient
from mmore.process.crawler import Crawler, CrawlerConfig
from mmore.process.dispatcher import Dispatcher, DispatcherConfig
from mmore.process.drive_download import GoogleDriveDownloader
from mmore.process.previous_results import is_reusable_process, load_previous_results
from mmore.profiler import enable_profiling_from_env, profile_function
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

    data_path: Union[List[str], str]
    google_drive_ids: List[str]
    dispatcher_config: DispatcherConfig
    skip_already_processed: bool = False
    previous_results: Optional[str] = None


def _clear_processor_results(output_path):
    """Remove per-processor results.jsonl files before a fresh dispatch.
    Needed because MultimodalSample.to_jsonl uses append mode."""
    processors_dir = os.path.join(output_path, "processors")
    if not os.path.isdir(processors_dir):
        return
    for processor_name in os.listdir(processors_dir):
        results_path = os.path.join(processors_dir, processor_name, "results.jsonl")
        if os.path.exists(results_path):
            os.remove(results_path)


def _build_merged_results(output_path, reused_samples=None):
    """Build merged_results.jsonl from per-processor results + reused samples."""
    merged_output_path = os.path.join(output_path, "merged")
    output_file = os.path.join(merged_output_path, "merged_results.jsonl")
    os.makedirs(merged_output_path, exist_ok=True)

    new_results = []
    processors_dir = os.path.join(output_path, "processors")
    if os.path.isdir(processors_dir):
        for processor_name in sorted(os.listdir(processors_dir)):
            results_path = os.path.join(processors_dir, processor_name, "results.jsonl")
            if os.path.exists(results_path):
                with open(results_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            new_results.append(json.loads(line))

    all_results = (reused_samples or []) + new_results

    with open(output_file, "w") as f:
        for sample in all_results:
            f.write(json.dumps(sample) + "\n")

    logger.info(f"Merged results ({len(all_results)} samples) saved to {output_file}")


@profile_function()
def process(config_file: str):
    """Process documents from a directory."""
    click.echo(f"Dispatcher configuration file path: {config_file}")

    overall_start_time = time.time()

    config: ProcessInference = load_config(config_file, ProcessInference)

    ggdrive_downloader, ggdrive_download_dir = None, None
    if config.google_drive_ids:
        google_drive_ids = config.google_drive_ids
        ggdrive_downloader = GoogleDriveDownloader(google_drive_ids)
        ggdrive_downloader.download_all()
        ggdrive_download_dir = ggdrive_downloader.download_dir

    data_path = config.data_path or ggdrive_download_dir

    if data_path:
        if isinstance(data_path, str):
            data_path = [data_path]

        # add the ggdrive_download_dir only if needed
        if config.data_path and ggdrive_download_dir:
            data_path += [ggdrive_download_dir]

        crawler_config = CrawlerConfig(
            root_dirs=data_path,
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
                ".html",
                ".htm",  # HTML pages
            ],
            output_path=config.dispatcher_config.output_path,
        )
    else:
        raise ValueError("Data path not provided in the configuration")

    logger.info(f"Using crawler configuration: {crawler_config}")
    crawler = Crawler(config=crawler_config)

    crawl_start_time = time.time()
    crawl_result = crawler.crawl(skip_already_processed=config.skip_already_processed)
    crawl_end_time = time.time()
    crawl_time = crawl_end_time - crawl_start_time
    logger.info(f"Crawling completed in {crawl_time:.2f} seconds")

    # Collect all crawled file paths for deletion filtering
    all_crawled_paths = {
        fd.file_path
        for file_list in crawl_result.file_paths.values()
        for fd in file_list
    }

    previous = None
    reused_samples = []
    reusable_paths = set()

    if config.previous_results:
        previous = load_previous_results(config.previous_results)

        for fp in all_crawled_paths:
            if is_reusable_process(fp, previous):
                reusable_paths.add(fp)

        # Collect reused samples (only for files still present in the crawl)
        for fp in reusable_paths:
            reused_samples.extend(previous[fp])

        # Remove reusable files from crawl_result so they are not re-processed
        crawl_result.file_paths = {
            root_dir: [fd for fd in file_list if fd.file_path not in reusable_paths]
            for root_dir, file_list in crawl_result.file_paths.items()
        }

        n_deleted = len(set(previous.keys()) - all_crawled_paths)
        logger.info(
            f"Change detection: {len(reusable_paths)} reused, "
            f"{len(crawl_result)} to process, {n_deleted} deleted"
        )

    output_path = config.dispatcher_config.output_path

    if len(crawl_result) == 0 and not reused_samples:
        logger.warning("\u26a0\ufe0f Found no file to process")
        return

    if len(crawl_result) == 0 and reused_samples:
        logger.info("No new files to process; writing reused samples only.")
        _clear_processor_results(output_path)
        _build_merged_results(output_path, reused_samples)
        if ggdrive_downloader:
            ggdrive_downloader.remove_downloads()
        overall_end_time = time.time()
        overall_time = overall_end_time - overall_start_time
        logger.info(f"Total execution time: {overall_time:.2f} seconds")
        return

    dispatcher_config: DispatcherConfig = config.dispatcher_config

    url = dispatcher_config.dashboard_backend_url
    DashboardClient(url).init_db(len(crawl_result))

    logger.info(f"Using dispatcher configuration: {dispatcher_config}")

    # Clear per-processor files before dispatch — to_jsonl uses append mode,
    # so stale files from a prior run would cause duplicates.
    _clear_processor_results(output_path)

    dispatcher = Dispatcher(result=crawl_result, config=dispatcher_config)

    dispatch_start_time = time.time()
    list(dispatcher())

    dispatch_end_time = time.time()
    dispatch_time = dispatch_end_time - dispatch_start_time

    logger.info(f"Dispatching and processing completed in {dispatch_time:.2f} seconds")

    _build_merged_results(output_path, reused_samples if previous is not None else None)

    if ggdrive_downloader:
        ggdrive_downloader.remove_downloads()

    overall_end_time = time.time()
    overall_time = overall_end_time - overall_start_time
    logger.info(f"Total execution time: {overall_time:.2f} seconds")


if __name__ == "__main__":
    enable_profiling_from_env()
    parser = argparse.ArgumentParser(description="Run the processing pipeline.")
    parser.add_argument(
        "--config_file", required=True, help="Path to the process configuration file."
    )
    args = parser.parse_args()

    process(args.config_file)
