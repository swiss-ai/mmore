"""
Indexing script.
Example usage:
    python run_index.py --config-file ./examples/index/indexer_config.yaml
"""

# Remove warnings
import torchvision
torchvision.disable_beta_transforms_warning()
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*TypedStorage is deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, message="BertForMaskedLM has generative capabilities.*")

import os

import argparse

from src.mmore.utils import load_config
from src.mmore.type import MultimodalSample
from src.mmore.index.indexer import IndexerConfig, Indexer

from dataclasses import dataclass, field
import json

from dotenv import load_dotenv
load_dotenv() 

# Set up logging
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Global logging configuration
#logging.basicConfig(format='%(asctime)s: %(message)s')
#logging.basicConfig(format='%(message)s')
logging.basicConfig(format='[INDEX 🗂️ ] %(message)s', level=logging.INFO)

# Suppress overly verbose logs from third-party libraries
logging.getLogger("transformers").setLevel(logging.CRITICAL)

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
    logger.debug(f"Loaded {len(results)} results")
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