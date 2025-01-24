from .utils import load_config
from .type import MultimodalSample
from .index.indexer import IndexerConfig, Indexer

from dataclasses import dataclass, field
import json

import logging
logger = logging.getLogger(__name__)
INDEX_EMOJI = "🗂️"
logging.basicConfig(format=f'[INDEX {INDEX_EMOJI}  -- %(asctime)s] %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

from dotenv import load_dotenv
load_dotenv() 

def load_results(path: str, file_type: str = None):
    # Load the results computed and saved by 'run_process.py'
    results = []
    logger.info(f"Loading results from {path}")
    with open(path, "rb") as f:
        for line in f:
            results.append(MultimodalSample.from_dict(json.loads(line)))
    logger.info(f"Loaded {len(results)} results")
    return results

def index(config_file, input_data, collection_name):
    """Index files for specified documents."""
    # Load the config file
    config = load_config(config_file, IndexerConfig) 

    documents = MultimodalSample.from_jsonl(input_data)
    
    logger.info("Creating the indexer...")
    indexer = Indexer.from_documents(
        config=config, 
        documents=documents,
        collection_name=collection_name
    )
    logger.info("Documents indexed!")

if __name__ == '__main__':
    index()