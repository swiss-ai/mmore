import os

import argparse

from src.mmore.utils import load_config
from src.mmore.type import MultimodalSample
from src.mmore.index.indexer import IndexerConfig, Indexer

from dataclasses import dataclass, field
import json

import logging
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv() 

@dataclass
class IndexerRunConfig:
    documents_path: str
    indexer: IndexerConfig
    collection_name: str = 'my_docs'
    batch_size: int = 64

def load_results(path: str, file_type: str = None):
    # Load the results computed and saved by 'run_process.py'
    results = []
    logger.info(f"Loading results from {path}")
    with open(path + '/merged/merged_results.jsonl', "rb") as f:
        for line in f:
            results.append(MultimodalSample.from_dict(json.loads(line)))
    logger.info(f"Loaded {len(results)} results")
    return results

def get_args():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Index files for specified documents')
    parser.add_argument('--config-file', type=str, required=True, help='Path to a config file.')

    # Parse the arguments
    return parser.parse_args()

if __name__ == "__main__":
    # get script args
    args = get_args()

    # Load the config file
    config = load_config(args.config_file, IndexerRunConfig) 
    
    logger.info("Creating the indexer...")
    indexer = Indexer.from_documents(
        config=config.indexer, 
        documents=load_results(config.documents_path),
        collection_name=config.collection_name,
        batch_size=config.batch_size
    )
    logger.info("Documents indexed!")
