from .utils import load_config
from .types.type import MultimodalSample
from .index.implementations.regular_rag.indexer import Indexer, IndexerConfig
from .index.implementations.graphrag.graphrag_indexer import GraphRAGIndexer, GraphRAGIndexerConfig
from .index import BaseIndexerConfig, BaseIndexer 
from typing import Dict, Type

import json

import logging
logger = logging.getLogger(__name__)
INDEX_EMOJI = "🗂️"
logging.basicConfig(format=f'[INDEX {INDEX_EMOJI}  -- %(asctime)s] %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

from dotenv import load_dotenv
load_dotenv() 

_indexers: Dict[str, Type[BaseIndexer]] = {
    "graphrag": GraphRAGIndexer,
    "regular": Indexer
}
    
_config_types: Dict[str, Type[BaseIndexerConfig]] = {
    "graphrag": GraphRAGIndexerConfig,
    "regular": IndexerConfig
}

def load_results(path: str, file_type: str = None):
    # Load the results computed and saved by 'run_process.py'
    results = []
    logger.info(f"Loading results from {path}")
    with open(path, "rb") as f:
        for line in f:
            results.append(MultimodalSample.from_dict(json.loads(line)))
    logger.info(f"Loaded {len(results)} results")
    return results

def index(config_file, input_data, collection_name, indexer_type="regular"):
    """Index files for specified documents."""
    # Load the config file
    config = load_config(config_file, _config_types[indexer_type])

    documents = MultimodalSample.from_jsonl(input_data)
    
    logger.info("Creating the indexer...")
    indexer = _indexers[indexer_type].from_documents(config, documents, collection_name)
    logger.info("Documents indexed!")

if __name__ == '__main__':
    index()