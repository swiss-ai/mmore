import argparse
from dataclasses import dataclass
from typing import Optional, Union

from dotenv import load_dotenv

from mmore.index.indexer import Indexer, IndexerConfig
from mmore.profiler import enable_profiling_from_env, profile_function
from mmore.type import MultimodalSample
from mmore.utils import load_config
from mmore.ux import quiet_noisy_libs, setup_logging

INDEX_EMOJI = "📇"
logger = setup_logging("INDEX", INDEX_EMOJI)

load_dotenv()


@dataclass
class IndexConfig:
    indexer: IndexerConfig
    collection_name: str
    documents_path: str


@profile_function()
def index(
    config_file: Union[IndexConfig, str],
    documents_path: Optional[str] = None,
    collection_name: Optional[str] = None,
):
    """Index files for specified documents."""
    quiet_noisy_libs()
    # Load the config file
    config: IndexConfig = load_config(config_file, IndexConfig)
    if collection_name is None:
        collection_name = config.collection_name
    if documents_path is None:
        documents_path = config.documents_path

    documents = MultimodalSample.from_jsonl(documents_path)

    logger.info(f"Indexing {len(documents)} documents into '{collection_name}'")
    Indexer.from_documents(
        config=config.indexer, documents=documents, collection_name=collection_name
    )
    logger.info(f"Done: {len(documents)} documents indexed into '{collection_name}'")


if __name__ == "__main__":
    enable_profiling_from_env()
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

    index(args.config_file, args.documents_path, args.collection_name)
