"""
Simple vector database indexer using Milvus for document storage.
Supports multimodal documents with chunking capabilities.
"""
from typing import List, Dict, Any, Literal
from dataclasses import dataclass, field

from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema

from mmore.type import MultimodalSample
from mmore.utils import load_config

from langchain_core.embeddings import Embeddings
from langchain_milvus.utils.sparse import BaseSparseEmbedding

from mmore.rag.model.dense.multimodal import MultimodalEmbeddings
from mmore.rag.model import DenseModel, SparseModel, DenseModelConfig, SparseModelConfig

from .postprocessor import load_postprocessor
from .postprocessor.base import BasePostProcessor, BasePostProcessorConfig
from .postprocessor.autoid import AutoID

from .filter import load_filter
from .filter.base import BaseFilter, BaseFilterConfig

from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)

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
                 dense_model_config: DenseModelConfig, 
                 sparse_model, 
                 sparse_model_config: SparseModelConfig, 
                 filters: List[BaseFilter],
                 postprocessors: List[BasePostProcessor], 
                 client: MilvusClient, 
                 batch_size: int = 8
            ):
        self.dense_model_config = dense_model_config
        self.dense_model = dense_model
        self.sparse_model_config = sparse_model_config
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
        dense_model = DenseModel.from_config(config.dense_model)
        sparse_model = SparseModel.from_config(config.sparse_model)

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
            dense_model_config=config.dense_model,
            dense_model=dense_model,
            sparse_model_config=config.sparse_model,
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
    
    def _print_plan(self, total_width: int = 100):
        # Define total line width
        section_title = " Filtering Pipeline "
        post_title = " Post-Processing Pipeline "
        
        # Function to create section headers with equal-length lines
        def _header_line(title):
            side_width = (total_width - len(title)) // 2
            return f"{'-' * side_width}{title}{'-' * (total_width - len(title) - side_width)}"

        # Print Filtering pipeline
        logger.info(_header_line(section_title))
        for i, filter in enumerate(self.filters):
            logger.info(f"  [{i+1}] {filter}")
        
        # Print PP pipeline
        logger.info(_header_line(post_title))
        for i, pp in enumerate(self.post_processors):
            logger.info(f"  [{i+1}] {pp}")

        # Print closing line
        logger.info('-' * total_width)

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
        for i in tqdm(range(0, len(documents), batch_size), desc="Indexing documents..."):
            batch = documents[i:i + batch_size]

            dense_embeddings = self.dense_model.embed_documents(Indexer._get_texts(batch, self.dense_model_config.is_multimodal))
            sparse_embeddings = self.sparse_model.embed_documents(Indexer._get_texts(batch, self.sparse_model_config.is_multimodal))

            data = []
            for _, (sample, d, s) in enumerate(zip(batch, dense_embeddings, sparse_embeddings)):
                data.append({
                    "id": Indexer._compute_hash(sample),
                    "doc_id": sample.id,
                    "text": sample.text,
                    "dense_embedding": d,
                    "sparse_embedding": s.reshape(1, -1),
                    **sample.metadata
                })

            batch_inserted = self.client.upsert(
                data=data,
                collection_name=collection_name,
                partition_name=partition_name,
            )
            inserted += list(batch_inserted.values())[0]

        logger.info(f"Updated {inserted} rows in collection {collection_name}")
        return inserted
    
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

    @staticmethod
    def _compute_hash(sample: MultimodalSample) -> str:
        return AutoID.hash_text(sample.text)

    def _create_index(self):
        """Create index on the embeddings fields."""
        index_params = self.client.prepare_index_params()

        logger.info(f"Creating index for dense embeddings with model {self.dense_model_config.model_name}")
        index_params.add_index(
            field_name="dense_embedding",
            model_name=self.dense_model_config.model_name,
            is_multimodal=self.dense_model_config.is_multimodal,
            metric_type="COSINE",
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

        logger.info(f"Filtering...")
        # Filter documents
        documents = self._filter_documents(documents)

        logger.info(f"Post-Processing...")
        # Post-process documents
        documents = self._pp_documents(documents)

        #logger.info(f"Cleaning pipelines done!\n")

        # Index documents 
        inserted = self._upsert_documents(
            documents,
            collection_name=collection_name,
            partition_name=partition_name,
            batch_size=batch_size
        )

        logger.debug(f"Collection stats:")
        for k,v in self.client.get_collection_stats(collection_name).items():
            logger.debug(f"  - {k}: {v}")

        return inserted


def get_model_from_index(client: MilvusClient, index_name: Literal['dense_embedding', 'sparse_embedding'], collection_name: str = None) -> DenseModelConfig | SparseModelConfig:
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
