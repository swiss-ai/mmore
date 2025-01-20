from dotenv import load_dotenv
load_dotenv() 

from src.mmore.rag.retriever import Retriever, RetrieverConfig
from tqdm import tqdm
import argparse
import time

from typing import Literal, List, Dict, Union
from dataclasses import dataclass
from langchain_core.documents import Document

from pathlib import Path
import json

import uvicorn
import logging
from src.mmore.utils import load_config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(format='[Retriever 🔍-- %(asctime)s] %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
# Suppress overly verbose logs from third-party libraries
logging.getLogger("transformers").setLevel(logging.CRITICAL)
logging.getLogger("torch").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("fastapi").setLevel(logging.WARNING)

@dataclass
class LocalConfig:
    input_file: str
    output_file: str

@dataclass
class RetrieverInferenceConfig:
    retriever: RetrieverConfig
    retriever_args: LocalConfig = None

def get_args():
    parser = argparse.ArgumentParser(description='Run Retriever: CLI mode')
    parser.add_argument('--config-file', type=str, required=True, help='Path to a config file')
    parser.add_argument('--input-file', type=str, required=True, help='Path to the input file')
    parser.add_argument('--output-file', type=str, required=True, help='Path to the output file')
    return parser.parse_args()


def read_queries(input_file: Path) -> List[str]:
    with open(input_file, 'r') as f:
        return [json.loads(line) for line in f]

def save_results(results: List[List[Document]], queries: List[str], output_file: Path):
    formatted_results = [
        {
            "query": query,
            "context": [
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in docs
            ]
        }
        for query, docs in zip(queries, results)
    ]

    # Write to the specified output file
    with open(output_file, 'w') as f:
        json.dump(formatted_results, f, indent=2)

if __name__ == "__main__":
    args = get_args()

    config = load_config(args.config_file, RetrieverInferenceConfig)

    logger.info('Running retriever...')
    retriever = Retriever.from_config(config.retriever)
    logger.info('Retriever loaded!')
    
    queries = read_queries(Path(args.input_file))  # Added missing argument
    # Measure time for the retrieval process
    logger.info("Starting document retrieval...")
    start_time = time.time()  # Start timer
    retrieved_docs = [retriever.invoke(query) for query in tqdm(queries, desc="Retrieving documents", unit="query")] 
    end_time = time.time()  # End timer
    
    time_taken = end_time - start_time
    logger.info(f"Document retrieval completed in {time_taken:.2f} seconds.")
    logger.info(f'Retrieved documents!')

    save_results(retrieved_docs, queries, Path(args.output_file))  # Added missing argument
    logger.info(f"Done! Results saved to {args.output_file}")