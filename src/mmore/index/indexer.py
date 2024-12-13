"""
Simple vector database indexer using Milvus for document storage.
Supports multimodal documents with chunking capabilities.
"""
from typing import List
from dataclasses import dataclass, field
from src.mmore.utils import load_config
from ..rag.models import load_dense_model, load_sparse_model
from src.mmore.type import MultimodalSample
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema
from langchain_core.embeddings import Embeddings
from langchain_milvus.utils.sparse import BaseSparseEmbedding
from ..rag.models.multimodal_model import MultimodalEmbeddings
from tqdm import tqdm

from .postprocessor import load_postprocessor
from .postprocessor.base import BasePostProcessor, BasePostProcessorConfig
from .postprocessor.autoid import AutoID

from .filter import load_filter
from .filter.base import BaseFilter, BaseFilterConfig

import logging
logger = logging.getLogger(__name__)

@dataclass
class DBConfig:
    uri: str = 'demo.db'
    name: str = 'my_db'


@dataclass
class IndexerConfig:
    """Configuration for the Indexer class. Currently db is local, if you wish to use Milvus standalone please check the Milvus documentation."""
    dense_model_name: str
    sparse_model_name: str
    db: DBConfig = field(default_factory=DBConfig)
    filters: List[BaseFilterConfig] = None
    pp: List[BasePostProcessorConfig] = None
    batch_size: int = 8

    def __post_init__(self):
        if isinstance(self.db, dict):
            self.db = DBConfig(**self.db)

class Indexer:
    """Handles document chunking, embedding computation, and Milvus storage."""
    dense_model: Embeddings
    sparse_model: BaseSparseEmbedding
    filters: List[BaseFilter]
    post_processors: List[BasePostProcessor]
    client: MilvusClient
    batch_size: int

    _DEFAULT_FIELDS = ['id', 'text', 'dense_embedding', 'sparse_embedding']

    def __init__(self, 
                 dense_model, 
                 dense_model_name, 
                 sparse_model, 
                 sparse_model_name, 
                 filters: List[BaseFilter],
                 postprocessors: List[BasePostProcessor], 
                 client: MilvusClient, 
                 batch_size: int = 8
            ):
        self.dense_model_name = dense_model_name
        self.dense_model = dense_model
        self.sparse_model_name = sparse_model_name
        self.sparse_model = sparse_model

        self.filters = filters
        self.post_processors = postprocessors

        self.client = client

        self.batch_size = batch_size

    @classmethod
    def from_config(cls, config: str | IndexerConfig):
        # Load the config if it's a string
        if isinstance(config, str):
            config = load_config(config, IndexerConfig)

        # Load the embedding models
        dense_model = load_dense_model(config.dense_model_name)
        sparse_model = load_sparse_model(config.sparse_model_name)

        # Load the filters
        if config.filters is None:
            filters = []
        else:
            filters = [load_filter(filter_config) for filter_config in config.filters]

        # Load the postprocessors
        if config.pp is None:
            postprocessors = []
        else:
            postprocessors = [load_postprocessor(pp_config) for pp_config in config.pp]

        # Create the milvus client
        milvus_client = MilvusClient(
            config.db.uri,
            db_name=config.db.name,
            enable_sparse=True,
        )

        return cls(
            dense_model_name=config.dense_model_name,
            dense_model=dense_model,
            sparse_model_name=config.sparse_model_name,
            sparse_model=sparse_model,
            filters=filters,
            postprocessors=postprocessors,
            client=milvus_client,
            batch_size=config.batch_size
        )

    @classmethod
    def from_documents(
        cls,
        config: str | IndexerConfig,
        documents: List[MultimodalSample],
        collection_name: str = 'my_docs',
        partition_name: str = None,
        batch_size: int = 32,
    ):
        indexer = Indexer.from_config(config)
        indexer.index_documents(
            documents,
            collection_name=collection_name,
            partition_name=partition_name,
            batch_size=batch_size
        )
        return indexer
    
    def _print_plan(self):
        # Print Filtering pipeline
        logger.info("Filtering pipeline:")
        for filter in self.filters:
            logger.info(f"  - {filter}")

        # Print PP pipeline
        logger.info("PP pipeline:")
        for pp in self.post_processors:
            logger.info(f"  - {pp}")

    def _id_documents(self, documents: List[MultimodalSample]) -> List[MultimodalSample]:
        for doc in documents:
            doc.id = Indexer._compute_hash(doc)
        return documents
    
    def _filter_documents(self, documents: List[MultimodalSample]) -> List[MultimodalSample]:
        for filter in self.filters:
            documents = [documents[i] for i, flag in enumerate(filter.batch_filter(documents)) if flag]
        return documents
        
    def _pp_documents(self, documents: List[MultimodalSample]) -> List[MultimodalSample]:
        for pp in self.post_processors:
            documents = pp.batch_process(documents)
        return documents
    
    def _upsert_documents(self, 
            documents: List[MultimodalSample],
            collection_name: str = 'my_docs',
            partition_name: str = None,
            batch_size: int = 32
        ) -> List[int]:
        # Create collection
        if not self.client.has_collection(collection_name):
            logger.info(f"Creating collection {collection_name}")
            self._create_collection_with_schema(collection_name)

        # Process new documents in batches
        inserted = 0
        for i in tqdm(range(0, len(documents), batch_size), desc="[INDEXER] Indexing documents..."):
            batch = documents[i:i + batch_size]

            dense_embeddings, sparse_embeddings = self.compute_document_embeddings(batch)

            data = []
            for j, (sample, d, s) in enumerate(zip(batch, dense_embeddings, sparse_embeddings)):
                data.append({
                    "id": Indexer._compute_hash(sample),
                    "text": sample.text,
                    "dense_embedding": d,
                    "sparse_embedding": s.reshape(1, -1),
                    **sample.metadata
                })

            batch_inserted = self.client.insert(
                data=data,
                collection_name=collection_name,
                partition_name=partition_name,
            )
            inserted += list(batch_inserted.values())[0]

        logger.info(f"Updated {inserted} documents in collection {collection_name}")
        return inserted

    def compute_document_embeddings(self, documents: List[MultimodalSample]):
        """Compute both custom and SPLADE embeddings for documents."""
        if isinstance(self.dense_model, MultimodalEmbeddings):
            texts = [MultimodalEmbeddings._multimodal_to_text(doc) for doc in documents]
        else:
            texts = [doc.text.replace("<attachment>", "") for doc in documents]

        return self.dense_model.embed_documents(texts), self.sparse_model.embed_documents(texts)
    
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

    @staticmethod
    def _compute_hash(sample: MultimodalSample) -> str:
        return AutoID.hash_text(sample.text)

    def _create_index(self):
        """Create index on the embeddings fields."""
        index_params = self.client.prepare_index_params()

        logger.info(f"Creating index for dense embeddings with model {self.dense_model_name}")
        index_params.add_index(
            field_name="dense_embedding",
            model_name=self.dense_model_name,
            metric_type="COSINE",
            params={"nlist": 128},
        )

        logger.info(f"Creating index for sparse embeddings with model {self.sparse_model_name}")
        index_params.add_index(
            field_name="sparse_embedding",
            model_name=self.sparse_model_name,
            metric_type="IP",
            index_type="SPARSE_INVERTED_INDEX",
        )
        return index_params

    def index_documents(self, 
            documents: List[MultimodalSample],
            collection_name: str = 'my_docs',
            partition_name: str = None,
            batch_size: int = 32
        ) -> List[int]:
        # Print the pipeline plan
        self._print_plan()

        # ID documents
        documents = self._id_documents(documents)

        # Filter documents
        documents = self._filter_documents(documents)

        # Post-process documents
        documents = self._pp_documents(documents)

        # Index documents 
        inserted = self._upsert_documents(
            documents,
            collection_name=collection_name,
            partition_name=partition_name,
            batch_size=batch_size
        )

        logger.info(f"Collection stats:")
        for k,v in self.client.get_collection_stats(collection_name).items():
            logger.info(f"  - {k}: {v}")

        return inserted


def get_model_from_index(client: MilvusClient, index_name: str, collection_name: str = None):
    collection_name = collection_name or client.list_collections()[0]
    return client.describe_index(collection_name, index_name)['model_name']
