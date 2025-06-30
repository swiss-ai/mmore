import argparse
import json
import logging
from dataclasses import dataclass
from typing import Optional, Union

from dotenv import load_dotenv

from mmore.index.indexer import Indexer, IndexerConfig
from mmore.type import MultimodalSample
from mmore.utils import load_config

logger = logging.getLogger(__name__)
INDEX_EMOJI = "🗂️"
logging.basicConfig(
    format=f"[INDEX {INDEX_EMOJI}  -- %(asctime)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

load_dotenv()


@dataclass
class IndexConfig:
    indexer: IndexerConfig
    collection_name: str
    documents_path: str


def load_results(path: str):
    # Load the results computed and saved by 'run_process.py'
    results = []
    logger.info(f"Loading results from {path}")
    with open(path, "rb") as f:
        for line in f:
            results.append(MultimodalSample.from_dict(json.loads(line)))
    logger.info(f"Loaded {len(results)} results")
    return results


def index(
    config_file: Union[IndexConfig, str],
    documents_path: Optional[str] = None,
    collection_name: Optional[str] = None,
):
    """Index files for specified documents."""
    # Load the config file
    config: IndexConfig = load_config(config_file, IndexConfig)
    if collection_name is None:
        collection_name = config.collection_name
    if documents_path is None:
        documents_path = config.documents_path

    documents = MultimodalSample.from_jsonl(documents_path)

    logger.info("Creating the indexer...")
    Indexer.from_documents(
        config=config.indexer, documents=documents, collection_name=collection_name
    )
    logger.info("Documents indexed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file", required=True, help="Path to the index configuration file."
    )
    parser.add_argument(
        "--documents-path", "-f", required=False, help="Path to the JSONL data."
    )
    parser.add_argument(
        "--collection-name",
        "-n",
        required=False,
        help="Name of the collection to index.",
    )
    args = parser.parse_args()

    index_config = load_config(args.config_file, IndexConfig)
    index(
        index_config.indexer, index_config.documents_path, index_config.collection_name
    )
