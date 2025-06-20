import os

from ..run_index import IndexConfig, index
from ..run_postprocess import postprocess
from ..run_process import process
from ..utils import load_config

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESS_CONFIG_FILE = os.path.join(SCRIPT_DIR, 'process_config.yaml')
INDEX_CONFIG_FILE = os.path.join(SCRIPT_DIR, 'index_config.yaml')
POSTPROCESS_CONFIG_FILE = os.path.join(SCRIPT_DIR, 'postprocess_config.yaml')
POSTPROCESS_INPUT_DATA = os.path.join(os.path.join(SCRIPT_DIR, 'outputs'), 'merged/merged_results.jsonl')


if __name__ == "__main__":
    process(PROCESS_CONFIG_FILE)
    postprocess(POSTPROCESS_CONFIG_FILE, POSTPROCESS_INPUT_DATA)
    index_config = load_config(INDEX_CONFIG_FILE, IndexConfig)
    index(
        INDEX_CONFIG_FILE, index_config.documents_path, index_config.collection_name
    )
