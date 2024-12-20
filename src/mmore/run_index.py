import os

import argparse

from .utils import load_config
from .type import MultimodalSample
from .index.indexer import IndexerConfig, Indexer

import click 

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

def load_results(path: str, file_type: str = None):
    # Load the results computed and saved by 'run_process.py'
    results = []
    logger.info(f"Loading results from {path}")
    with open(path + '/merged/merged_results.jsonl', "rb") as f:
        for line in f:
            results.append(MultimodalSample.from_dict(json.loads(line)))
    logger.info(f"Loaded {len(results)} results")
    return results

@click.command()
@click.option('--config-file', type=str, required=True, help='Path to a config file.')
def index(config_file):
    """Index files for specified documents."""
    # Load the config file
    config = load_config(config_file, IndexerRunConfig) 
    
    logger.info("Creating the indexer...")
    indexer = Indexer.from_documents(
        config=config.indexer, 
        documents=load_results(config.documents_path),
        collection_name=config.collection_name
    )
    logger.info("Documents indexed!")


if __name__ == '__main__':
    index()