import click
import argparse
import yaml

from contextlib import contextmanager
import warnings
import sys
import os

from mmore.websearch.pipeline import WebsearchPipeline
from mmore.websearch.config import WebsearchConfig

# @contextmanager
# def suppress_warnings_and_stdout():
#     # Suppress specific warnings
#     warnings.filterwarnings('ignore', category=FutureWarning, message="The input name `inputs` is deprecated*")
#     warnings.filterwarnings('ignore', category=FutureWarning, message="*UserWarning:*")
    
#     pypdfium_message = "-> Cannot close object, library is destroyed. This may cause a memory leak!*"
#     # Redirect stdout to devnull to catch pypdfium messages
#     old_stdout = sys.stdout
#     devnull = open(os.devnull, 'w')
#     # Suppress pypdfium warnings
#     sys.stdout = devnull
    
#     try:
#         sys.stdout = devnull
#         yield
        
#     finally:
#         sys.stdout = old_stdout
#         devnull.close()

# import logging
# logger = logging.getLogger(__name__)
# MMORE_EMOJI = "🐮"
# logging.basicConfig(format=f'[MMORE {MMORE_EMOJI} -- %(asctime)s] %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

@click.group()
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="MMORE command line interface")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Websearch command
    websearch_parser = subparsers.add_parser("websearch", help="Run websearch pipeline")
    websearch_parser.add_argument("--config-file", required=True, help="Path to config file")

    args = parser.parse_args()

    if args.command == "websearch":
        with open(args.config_file, "r") as f:
            config = yaml.safe_load(f)
        pipeline = WebsearchPipeline(WebsearchConfig.from_dict(config))
        pipeline.run()
    else:
        parser.print_help()


@main.command()
@click.option(
    "--config-file", type=str, required=True, help="Dispatcher configuration file path."
)
def process(config_file):
    """Process documents from a directory."""
    from .run_process import process as run_process

    run_process(config_file)


@main.command()
@click.option(
    "--config-file",
    type=str,
    required=True,
    help="Path to the postprocess configuration file.",
)
@click.option(
    "--input-data", type=str, required=True, help="Path to the jsonl of the documents."
)
def postprocess(config_file, input_data):
    """Run the post-processors pipeline."""
    from .run_postprocess import postprocess as run_postprocess

    run_postprocess(config_file, input_data)


@main.command()
@click.option(
    "--config-file",
    "-c",
    type=str,
    required=True,
    help="Path to the configuration file.",
)
@click.option(
    "--documents-path", "-f", type=str, required=False, help="Path to the JSONL data."
)
@click.option(
    "--collection-name",
    "-n",
    type=str,
    required=False,
    help="Name of the collection to index.",
)
def index(config_file, documents_path, collection_name):
    """Run the indexer."""
    from .run_index import index as run_index

    run_index(config_file, documents_path, collection_name)


@main.command()
@click.option(
    "--config-file",
    "-c",
    type=str,
    required=True,
    help="Dispatcher configuration file path.",
)
@click.option(
    "--input-file", "-f", type=str, required=True, help="Path to the input file."
)
@click.option(
    "--output-file", "-o", type=str, required=True, help="Path to the output file."
)
def retrieve(config_file, input_file, output_file):
    """Retrieve documents for specified queries."""
    from .run_retriever import retrieve as run_retrieve

    run_retrieve(config_file, input_file, output_file)


@main.command()
@click.option(
    "--config-file", type=str, required=True, help="Dispatcher configuration file path."
)
def rag(config_file):
    """Run the Retrieval-Augmented Generation (RAG) pipeline."""
    from .run_rag import rag as run_rag

    run_rag(config_file)


@main.command()
@click.option('--config-file', type=str, required=True, help='Configuration file path.')
def websearch(config_file):
    """Run the websearch pipeline."""
    from .run_websearch import run_websearch
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    run_websearch(config)


if __name__ == "__main__":
    main()
