import argparse
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import click
import torch

from mmore.dashboard.backend.client import DashboardClient
from mmore.process.crawler import Crawler, CrawlerConfig
from mmore.process.dispatcher import Dispatcher, DispatcherConfig
from mmore.process.drive_download import GoogleDriveDownloader
from mmore.profiler import enable_profiling_from_env, profile_function
from mmore.type import MultimodalSample
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

        use_fast_processors: false
        extract_images: true
        skip_already_processed: false

        # processors:
        #   MediaProcessor:
        #     normal_model: openai/whisper-large-v3-turbo
        #     fast_model: openai/whisper-tiny
        #     frame_sample_rate: 10

    Attributes:
        input_path: Path (or list of paths) to directories containing the
            files to process.  Supports local directories and URLs.
        output_path: Directory where results will be written.
        use_fast_processors: Use faster but lower-quality processing modes
            where available (default: ``False``).
        extract_images: Extract embedded images from documents
            (default: ``True``).
        distributed: Use distributed processing via Dask (default:
            ``False``).  Requires a running Dask cluster.
        skip_already_processed: Skip files whose output already exists in
            *output_path* from a previous run (default: ``False``).
        google_drive_ids: List of Google Drive folder IDs to download and
            process (default: empty).
        scheduler_file: Path to a Dask scheduler file, required when
            ``distributed=True``.
        dashboard_backend_url: Optional mmore dashboard URL for live
            progress tracking.
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
    use_fast_processors: bool = False
    extract_images: bool = True
    distributed: bool = False
    skip_already_processed: bool = False
    google_drive_ids: List[str] = field(default_factory=list)
    scheduler_file: Optional[str] = None
    dashboard_backend_url: Optional[str] = None
    batch_sizes: Dict[str, int] = field(default_factory=dict)
    batch_multiplier: int = 1
    processors: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    file_type_processors: Dict[str, str] = field(default_factory=dict)


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
    crawl_result = crawler.crawl(skip_already_processed=config.skip_already_processed)
    logger.info(f"Crawling completed in {time.time() - crawl_start_time:.2f} seconds")

    if len(crawl_result) == 0:
        logger.warning("⚠️ Found no file to process")
        return

    dispatcher_config = DispatcherConfig(
        output_path=config.output_path,
        use_fast_processors=config.use_fast_processors,
        extract_images=config.extract_images,
        distributed=config.distributed,
        scheduler_file=config.scheduler_file,
        dashboard_backend_url=config.dashboard_backend_url,
        batch_sizes=config.batch_sizes,
        batch_multiplier=config.batch_multiplier,
        processor_configs=config.processors,
        file_type_processors=config.file_type_processors,
    )

    url = dispatcher_config.dashboard_backend_url
    DashboardClient(url).init_db(len(crawl_result))

    logger.info(f"Using dispatcher configuration: {dispatcher_config}")
    dispatcher = Dispatcher(result=crawl_result, config=dispatcher_config)

    dispatch_start_time = time.time()
    results = list(dispatcher())
    logger.info(
        f"Dispatching and processing completed in "
        f"{time.time() - dispatch_start_time:.2f} seconds"
    )

    merged_output_path = os.path.join(config.output_path, "merged")
    output_file = os.path.join(merged_output_path, "merged_results.jsonl")
    os.makedirs(merged_output_path, exist_ok=True)
    for res in results:
        MultimodalSample.to_jsonl(output_file, res)

    logger.info(f"Merged results ({len(results)} items) saved to {output_file}")

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
