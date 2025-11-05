import argparse
import logging
from dataclasses import dataclass
from typing import Union
from pathlib import Path
import pandas as pd
import torch
import yaml

from colpali_engine.models import ColPali, ColPaliProcessor
from milvuscolpali import MilvusColpaliManager

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
    dim: int = 128
    create_collection: bool = False
    metric_type: str = "IP"

@dataclass
class IndexConfig:
    milvus: MilvusConfig
    parquet_path: str

def load_config(config_path: Union[str, Path]) -> IndexConfig:
    """
    Load YAML config file and return IndexConfig.
    """
    with open(config_path, "r") as f:
        raw_cfg = yaml.safe_load(f)

    milvus_cfg = MilvusConfig(**raw_cfg.get("milvus", {}))
    return IndexConfig(
        milvus=milvus_cfg,
        parquet_path=raw_cfg["parquet_path"],
    )


def load_model(model_name: str = "vidore/colpali-v1.3", device: str = "cuda:0"):
    """
    Load ColPali model and processor.
    """
    logger.info(f"Loading model: {model_name}")
    model = ColPali.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device
    ).eval()
    processor = ColPaliProcessor.from_pretrained(model_name)
    return model, processor


def index(config_file: Union[IndexConfig, str]):
    """
    Main indexing function.
    Loads embeddings from parquet and inserts them into Milvus local DB.
    """
    config: IndexConfig = load_config(config_file)

    parquet_path = config.parquet_path
    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded {len(df)} rows from {parquet_path}")

    manager = MilvusColpaliManager(
        db_path=config.milvus.db_path,
        collection_name=config.milvus.collection_name,
        dim=config.milvus.dim,
        metric_type=config.milvus.metric_type,
        create_collection=config.milvus.create_collection,
    )
    manager.insert_from_dataframe(df)
    manager.create_index()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Index ColPali PDF embeddings into a local Milvus database."
    )
    parser.add_argument("--config_file", required=True, help="Path to the YAML config file.")
    args = parser.parse_args()

    index(args.config_file)
