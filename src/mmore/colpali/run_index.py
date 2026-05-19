import argparse
import logging
from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd

from mmore.profiler import enable_profiling_from_env, profile_function

from ..utils import load_config
from .milvuscolpali import MilvusColpaliManager

INDEX_EMOJI = "🗂️"
logger = logging.getLogger(__name__)
logging.basicConfig(
    format=f"[INDEX {INDEX_EMOJI}  -- %(asctime)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


@dataclass
class MilvusConfig:
    db_path: str = "./milvus_data"
    collection_name: str = "pdf_pages"
    create_collection: bool = False
    metric_type: str = "IP"


@dataclass
class IndexConfig:
    milvus: MilvusConfig
    parquet_path: str


def _get_embedding_dim(df: pd.DataFrame) -> int:
    arr = np.asarray(df.iloc[0]["embedding"])
    if arr.dtype == object:
        arr = np.stack([np.asarray(x) for x in arr])
    return int(arr.shape[-1])


@profile_function()
def index(config_file: Union[IndexConfig, str]):
    """
    Main indexing function.
    Loads embeddings from parquet and inserts them into Milvus local DB.
    """
    config = load_config(config_file, IndexConfig)

    parquet_path = config.parquet_path
    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded {len(df)} rows from {parquet_path}")

    if df.empty:
        logger.info(f"Parquet {parquet_path} is empty — nothing to index, exiting.")
        return

    dim = _get_embedding_dim(df)
    logger.info(f"Detected embedding dim={dim} from parquet")

    manager = MilvusColpaliManager(
        db_path=config.milvus.db_path,
        collection_name=config.milvus.collection_name,
        dim=dim,
        metric_type=config.milvus.metric_type,
        create_collection=config.milvus.create_collection,
    )
    manager.insert_from_dataframe(df)
    manager.create_index()


if __name__ == "__main__":
    enable_profiling_from_env()
    parser = argparse.ArgumentParser(
        description="Index ColPali PDF embeddings into a local Milvus database."
    )
    parser.add_argument(
        "--config-file", required=True, help="Path to the YAML config file."
    )
    args = parser.parse_args()

    index(args.config_file)
