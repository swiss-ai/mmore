from dotenv import load_dotenv
load_dotenv() 

from src.mmore.rag.retriever import Retriever, RetrieverConfig
from tqdm import tqdm
import time

from typing import Literal, List, Dict, Union
from langchain_core.documents import Document

from pathlib import Path
import json

import logging
from src.mmore.utils import load_config

logger = logging.getLogger(__name__)
RETRIVER_EMOJI = "🔍"
logging.basicConfig(format=f'[RETRIEVER {RETRIVER_EMOJI} -- %(asctime)s] %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

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

def retrieve(config_file, input_file, output_file, document_ids=None):
    """Retrieve documents for specified queries.
    
    If 'document_ids' is provided, function bypasses the vectore-based similarity search and instead retrieved only the documents with the given IDs"""

    # Load the config file
    config = load_config(config_file, RetrieverConfig)

    logger.info('Running retriever...')
    retriever = Retriever.from_config(config.retriever)
    logger.info('Retriever loaded!')
    
    queries = read_queries(Path(input_file))  # Added missing argument

    # Process the document_ids provided, split into individual IDs, strip any extra whitespace, and filter out any empty strings
    if document_ids:
        doc_ids_list = [doc_id.strip() for doc_id in document_ids.split(",") if doc_id.strip()]
    else:
        doc_ids_list = []

    if doc_ids_list:
        logger.info(f"Manual document IDs specified: {doc_ids_list}")

        # Retrieve specified documents once from Milvus
        # Since these documents are to be used for all queries, we replicate the same list of retrieved documents for each query
        retrieved_docs_all = [retriever.get_documents_by_ids(doc_ids_list)] * len(queries)

        # Save the results along with the queries to the output file
        save_results(retrieved_docs_all, queries, Path(output_file))
        logger.info(f"Done! Results saved to {output_file}")

    else:
        # Measure time for the retrieval process
        logger.info("Starting document retrieval...")
        start_time = time.time()  # Start timer
        retrieved_docs = [retriever.invoke(query) for query in tqdm(queries, desc="Retrieving documents", unit="query")] 
        end_time = time.time()  # End timer
        
        time_taken = end_time - start_time
        logger.info(f"Document retrieval completed in {time_taken:.2f} seconds.")
        logger.info(f'Retrieved documents!')

        save_results(retrieved_docs, queries, Path(output_file))  # Added missing argument
        logger.info(f"Done! Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, help="Path to the index configuration file.")
    parser.add_argument("--input_file", required=True, help="Path to the input file of queries.")
    parser.add_argument("--output_file", required=True, help="Path to the output file of selected documents.")

    args = parser.parse_args()
    retrieve()