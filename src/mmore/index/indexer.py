"""
Simple vector database indexer using Milvus for document storage.
Supports multimodal documents with chunking capabilities.
"""
from typing import List, Dict, Any, Literal, Optional
from dataclasses import dataclass, field
from ..utils import load_config
from ..type import MultimodalSample
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema

from langchain_core.embeddings import Embeddings
from langchain_milvus.utils.sparse import BaseSparseEmbedding

from ..rag.model.dense.multimodal import MultimodalEmbeddings
from ..rag.model import DenseModel, SparseModel, DenseModelConfig, SparseModelConfig

from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)
import scipy

@dataclass
class DBConfig:
    uri: str = 'demo.db'
    name: str = 'my_db'

@dataclass
class IndexerConfig:
    """Configuration for the Indexer class. Currently db is local, if you wish to use Milvus standalone please check the Milvus documentation."""
    dense_model: DenseModelConfig
    sparse_model: SparseModelConfig
    db: DBConfig = field(default_factory=DBConfig)

    def __post_init__(self):
        if isinstance(self.db, dict):
            self.db = DBConfig(**self.db)

class Indexer:
    """Handles document chunking, embedding computation, and Milvus storage."""
    dense_model: Embeddings
    sparse_model: BaseSparseEmbedding
    client: MilvusClient

    _DEFAULT_FIELDS = ['id', 'text', 'dense_embedding', 'sparse_embedding']

    def __init__(
                self, 
                dense_model_config: DenseModelConfig, 
                sparse_model_config: SparseModelConfig, 
                client: MilvusClient, 
            ):
        # Load the embedding models
        self.dense_model_config = dense_model_config
        self.dense_model = DenseModel.from_config(dense_model_config)
        self.sparse_model_config = sparse_model_config
        self.sparse_model = SparseModel.from_config(sparse_model_config)

        self.client = client

    @classmethod
    def from_config(cls, config: str | IndexerConfig):
        # Load the config if it's a string
        if isinstance(config, str):
            config_obj = load_config(config, IndexerConfig)
        else:
            config_obj = config

        # Create the milvus client
        milvus_client = MilvusClient(
            config_obj.db.uri,
            db_name=config_obj.db.name,
            enable_sparse=True,
        )

        return cls(
            dense_model_config=config_obj.dense_model,
            sparse_model_config=config_obj.sparse_model,
            client=milvus_client,
        )

    @classmethod
    def from_documents(
        cls,
        config: str | IndexerConfig,
        documents: List[MultimodalSample],
        collection_name: str = 'my_docs',
        partition_name: Optional[str] = None,
        batch_size: int = 64,
    ):
        indexer = Indexer.from_config(config)
        indexer.index_documents(
            documents,
            collection_name=collection_name,
            partition_name=partition_name,
            batch_size=batch_size
        )
        return indexer
    
    @staticmethod
    def _get_texts(documents: List[MultimodalSample], is_multimodal: bool) -> List[str]:
        if is_multimodal:
            return [MultimodalEmbeddings._multimodal_to_text(doc) for doc in documents]
        else:
            return [doc.text.replace("<attachment>", "") for doc in documents]
    
    def _create_collection_with_schema(self, collection_name: str):
        """Create Milvus collection with fields for both embeddings."""
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),  # Add doc_id field
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(
                name="dense_embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=len(self.dense_model.embed_query("test")),
            ),
            FieldSchema(
                name="sparse_embedding",
                dtype=DataType.SPARSE_FLOAT_VECTOR,
            ),
        ]

        schema = CollectionSchema(
            fields,
            enable_dynamic_field=True
        )

        self.client.create_collection(
            collection_name=collection_name, 
            schema=schema, 
            index_params=self._create_index(),
        )

    def _create_index(self):
        """Create index on the embeddings fields."""
        index_params = self.client.prepare_index_params()

        logger.info(f"Creating index for dense embeddings with model {self.dense_model_config.model_name}")
        index_params.add_index(
            field_name="dense_embedding",
            model_name=self.dense_model_config.model_name,
            is_multimodal=self.dense_model_config.is_multimodal,
            metric_type="COSINE",
            index_type="IVF_FLAT",
            params={"nlist": 128},
        )

        logger.info(f"Creating index for sparse embeddings with model {self.sparse_model_config.model_name}")
        index_params.add_index(
            field_name="sparse_embedding",
            model_name=self.sparse_model_config.model_name,
            is_multimodal=self.sparse_model_config.is_multimodal,
            metric_type="IP",
            index_type="SPARSE_INVERTED_INDEX",
        )
        return index_params
    
    def _index_documents(self, 
            documents: List[MultimodalSample],
            collection_name: str = 'my_docs',
            partition_name: Optional[str] = None,
            batch_size: int = 64
        ) -> int:
        # Process new documents in batches
        inserted = 0
        for i in tqdm(range(0, len(documents), batch_size), desc="Indexing documents..."):
            # Get the batch
            batch = documents[i:i + batch_size]

            # Compute embeddings
            dense_embeddings = self.dense_model.embed_documents(Indexer._get_texts(batch, self.dense_model_config.is_multimodal))
            sparse_embeddings: scipy.sparse._coo.coo_array = self.sparse_model.embed_documents(Indexer._get_texts(batch, self.sparse_model_config.is_multimodal))
            
            # Insert the batch
            data = [{
                    "id": sample.id,
                    "text": sample.text,
                    "dense_embedding": d,
                    "sparse_embedding": s.reshape(1, -1),
                    **sample.metadata
                } for sample, d, s in zip(batch, dense_embeddings, sparse_embeddings)]
            
            batch_inserted = self.client.insert(
                data=data,
                collection_name=collection_name,
                partition_name=partition_name,
            )

            inserted += list(batch_inserted.values())[0]

        return inserted
    
    def _log_collection_stats(self, collection_name: str):
        logger.info("-"*50)    
        logger.info(f"Collection stats (before inserting):")
        for k,v in self.client.get_collection_stats(collection_name).items():
            logger.info(f"  - {k}: {v}")
        logger.info("-"*50)

    def index_documents(
            self, 
            documents: List[MultimodalSample],
            collection_name: str = 'my_docs',
            partition_name: Optional[str] = None,
            batch_size: int = 64
        ) -> int:   
        # Create collection
        if not self.client.has_collection(collection_name):
            logger.info(f"Creating collection {collection_name}")
            self._create_collection_with_schema(collection_name)
        else:
            logger.info(f"{collection_name} already exists, adding documents to it")

        self._log_collection_stats(collection_name)

        # Index documents 
        inserted = self._index_documents(
            documents,
            collection_name=collection_name,
            partition_name=partition_name,
            batch_size=batch_size
        )

        self._log_collection_stats(collection_name)

        return inserted


def get_model_from_index(client: MilvusClient, index_name: Literal['dense_embedding', 'sparse_embedding'], collection_name: Optional[str] = None) -> DenseModelConfig | SparseModelConfig:
    collection_name = collection_name or client.list_collections()[0]
    if index_name == 'dense_embedding':
        index_config = client.describe_index(collection_name, index_name)
        return DenseModelConfig(
            model_name=index_config['model_name'], 
            is_multimodal=index_config['is_multimodal'] == 'True',
        )
    elif index_name == 'sparse_embedding':
        index_config = client.describe_index(collection_name, index_name)
        return SparseModelConfig(
            model_name=index_config['model_name'], 
            is_multimodal=index_config['is_multimodal'] == 'True',
        )
    else:
        raise ValueError(f"Invalid index_name: {index_name}. Must be 'dense_embedding' or 'sparse_embedding'.")
