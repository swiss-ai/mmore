import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import click
import torch

from mmore.process.crawler import Crawler, CrawlerConfig
from mmore.process.dispatcher import Dispatcher, DispatcherConfig
from mmore.process.drive_download import GoogleDriveDownloader
from mmore.process.incremental import (
    is_reusable_process,
    load_previous_process_results,
)
from mmore.profiler import enable_profiling_from_env, profile_function
from mmore.utils import load_config

PROCESS_EMOJI = "🚀"
logger = logging.getLogger(__name__)
logging.basicConfig(
    format=f"[Process {PROCESS_EMOJI} -- %(asctime)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


@dataclass
class ProcessConfig:
    """Top-level configuration for the ``mmore process`` command.

    All fields map directly to top-level keys in the YAML config file —
    there is no nesting.

    Example config file::

        input_path: data/documents/
        output_path: data/outputs/
        previous_results: null

        use_fast_processors: false
        extract_images: true

        # processors:
        #   MediaProcessor:
        #     normal_model: openai/whisper-large-v3-turbo
        #     fast_model: openai/whisper-tiny
        #     frame_sample_rate: 10

    Attributes:
        input_path: Path (or list of paths) to directories containing the
            files to process.  Supports local directories and URLs.
        output_path: Directory where results will be written.
        previous_results: Path to a previous JSON result file to reuse.
        use_fast_processors: Use faster but lower-quality processing modes
            where available (default: ``False``).
        extract_images: Extract embedded images from documents
            (default: ``True``).
        distributed: Use distributed processing via Dask (default:
            ``False``).  Requires a running Dask cluster.
        google_drive_ids: List of Google Drive folder IDs to download and
            process (default: empty).
        scheduler_file: Path to a Dask scheduler file, required when
            ``distributed=True``.
        batch_sizes: Per-processor batch sizes (document pages per batch),
            keyed by processor class name.
        batch_multiplier: Scale all batch sizes by this factor (default:
            ``1``).
        processors: Per-processor setting overrides, keyed by
            processor class name.  See each processor's config class for
            available fields.
        file_type_processors: Override which processor handles a given file
            extension, e.g. ``{".pdf": "PDFProcessor"}``.
    """

    input_path: Union[List[str], str]
    output_path: str
    previous_results: Optional[str] = None
    use_fast_processors: bool = False
    extract_images: bool = True
    distributed: bool = False
    google_drive_ids: List[str] = field(default_factory=list)
    scheduler_file: Optional[str] = None
    batch_sizes: Dict[str, int] = field(default_factory=dict)
    batch_multiplier: int = 1
    processors: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    file_type_processors: Dict[str, str] = field(default_factory=dict)


def _write_merged_results(output_path, reused_samples, dispatched=True):
    """Merge per-processor JSONL files and reused samples into a single output."""
    merged_output_path = os.path.join(output_path, "merged")
    output_file = os.path.join(merged_output_path, "merged_results.jsonl")
    os.makedirs(merged_output_path, exist_ok=True)

    total_results = 0
    with open(output_file, "w") as f:
        for sample in reused_samples:
            f.write(json.dumps(sample.to_dict()) + "\n")
            total_results += 1
        if dispatched:
            processors_dir = os.path.join(output_path, "processors")
            if os.path.isdir(processors_dir):
                for processor_name in sorted(os.listdir(processors_dir)):
                    results_path = os.path.join(
                        processors_dir, processor_name, "results.jsonl"
                    )
                    if os.path.exists(results_path):
                        with open(results_path, "r") as processor_file:
                            for line in processor_file:
                                stripped_line = line.strip()
                                if stripped_line:
                                    f.write(stripped_line + "\n")
                                    total_results += 1

    logger.info(f"Merged results ({total_results} samples) saved to {output_file}")


@profile_function()
def process(config_file: str):
    """Process documents from a directory."""
    click.echo(f"Dispatcher configuration file path: {config_file}")

    overall_start_time = time.time()

    config: ProcessConfig = load_config(config_file, ProcessConfig)

    ggdrive_downloader, ggdrive_download_dir = None, None
    if config.google_drive_ids:
        ggdrive_downloader = GoogleDriveDownloader(config.google_drive_ids)
        ggdrive_downloader.download_all()
        ggdrive_download_dir = ggdrive_downloader.download_dir

    input_path = config.input_path or ggdrive_download_dir

    if input_path:
        if isinstance(input_path, str):
            input_path = [input_path]

        # Add the Google Drive download dir only if needed
        if config.input_path and ggdrive_download_dir:
            input_path += [ggdrive_download_dir]

        crawler_config = CrawlerConfig(
            root_dirs=input_path,
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
            output_path=config.output_path,
        )
    else:
        raise ValueError("input_path not provided in the configuration")

    logger.info(f"Using crawler configuration: {crawler_config}")
    crawler = Crawler(config=crawler_config)

    crawl_start_time = time.time()
    crawl_result = crawler.crawl()
    logger.info(f"Crawling completed in {time.time() - crawl_start_time:.2f} seconds")

    # Collect all crawled file paths and urls (excluding this way deleted files)
    all_crawled_paths = {
        fd.file_path
        for file_list in crawl_result.file_paths.values()
        for fd in file_list
    }
    all_crawled_paths.update(url.file_path for url in crawl_result.urls)

    previous = None
    reused_samples = []
    reusable_paths = set()

    if config.previous_results:
        previous = load_previous_process_results(config.previous_results)

        for fp in all_crawled_paths:
            if is_reusable_process(fp, previous):
                reusable_paths.add(fp)

        reused_samples = [previous[fp] for fp in sorted(reusable_paths)]

        # Remove reusable files from crawl_result so they are not re-processed
        crawl_result.file_paths = {
            root_dir: [fd for fd in file_list if fd.file_path not in reusable_paths]
            for root_dir, file_list in crawl_result.file_paths.items()
        }

        n_deleted = len(set(previous.keys()) - all_crawled_paths)
        logger.info(
            f"Process pipeline: {len(reusable_paths)} reused, "
            f"{len(crawl_result)} to process, {n_deleted} deleted"
        )

    dispatched = len(crawl_result) > 0

    if not dispatched and not reused_samples:
        logger.warning("⚠️ Found no file to process")
        if previous is None:
            return

    if dispatched:
        dispatcher_config = DispatcherConfig(
            output_path=config.output_path,
            use_fast_processors=config.use_fast_processors,
            extract_images=config.extract_images,
            distributed=config.distributed,
            scheduler_file=config.scheduler_file,
            batch_sizes=config.batch_sizes,
            batch_multiplier=config.batch_multiplier,
            processor_configs=config.processors,
            file_type_processors=config.file_type_processors,
        )

        logger.info(f"Using dispatcher configuration: {dispatcher_config}")
        dispatcher = Dispatcher(result=crawl_result, config=dispatcher_config)

        dispatch_start_time = time.time()
        dispatcher()
        logger.info(
            f"Dispatching and processing completed in "
            f"{time.time() - dispatch_start_time:.2f} seconds"
        )
    elif reused_samples:
        logger.info("No new files to process, reusing previous samples only.")
    else:
        logger.info("No new files to process and no samples to reuse.")

    _write_merged_results(
        config.output_path,
        reused_samples,
        dispatched=dispatched,
    )

    if ggdrive_downloader:
        ggdrive_downloader.remove_downloads()

    logger.info(f"Total execution time: {time.time() - overall_start_time:.2f} seconds")


if __name__ == "__main__":
    enable_profiling_from_env()
    parser = argparse.ArgumentParser(description="Run the processing pipeline.")
    parser.add_argument(
        "--config_file", required=True, help="Path to the process configuration file."
    )
    args = parser.parse_args()
    process(args.config_file)
