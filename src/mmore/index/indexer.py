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
from src.mmore.index.chunker import ChunkerConfig, MultimodalChunker
from langchain_core.embeddings import Embeddings
from langchain_milvus.utils.sparse import BaseSparseEmbedding
from ..rag.models.multimodal_model import MultimodalEmbeddings
from tqdm import tqdm
import hashlib
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
    chunker: ChunkerConfig = field(default_factory=ChunkerConfig)
    batch_size: int = 8

    def __post_init__(self):
        if isinstance(self.chunker, dict):
            self.chunker = ChunkerConfig(**self.chunker)
        if isinstance(self.db, dict):
            self.db = DBConfig(**self.db)


class Indexer:
    """Handles document chunking, embedding computation, and Milvus storage."""
    dense_model: Embeddings
    sparse_model: BaseSparseEmbedding
    chunker: MultimodalChunker
    client: MilvusClient
    batch_size: int

    _DEFAULT_FIELDS = ['id', 'text', 'dense_embedding', 'sparse_embedding']

    def __init__(self, dense_model, dense_model_name, sparse_model, sparse_model_name, chunker, client, batch_size: int = 8):
        self.dense_model_name = dense_model_name
        self.dense_model = dense_model
        self.sparse_model_name = sparse_model_name
        self.sparse_model = sparse_model
        self.chunker = chunker
        self.client = client
        self.batch_size = batch_size

    @classmethod
    def from_config(cls, config: str | IndexerConfig):
        if isinstance(config, str):
            config = load_config(config, IndexerConfig)

        dense_model = load_dense_model(config.dense_model_name)
        sparse_model = load_sparse_model(config.sparse_model_name)

        chunker = MultimodalChunker.from_config(config.chunker)

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
            chunker=chunker,
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

    def index_documents(self, documents: List[MultimodalSample],
                        collection_name: str = 'my_docs',
                        partition_name: str = None,
                        metadata_tags: List[str] = None,
                        batch_size: int = 32) -> List[int]:

        # Create collection
        if not self.client.has_collection(collection_name):
            logger.info(f"Creating collection {collection_name}")
            if metadata_tags is None:
                metadata_tags = []
            self._create_collection_with_schema(
                collection_name=collection_name,
                metadata_tags=metadata_tags,
            )

        # Chunk and filter documents
        logger.info(f"Chunking {len(documents)} documents")
        chunked_documents = self.chunker.chunk_batch(documents)
        documents = [c for chunks in chunked_documents for c in chunks]

        # # Get existing document IDs
        # existing_docs = self._get_existing_doc_ids(collection_name)
        # logger.info(f"Found {len(existing_docs)} existing documents in collection {collection_name}")
        # new_documents = []
        # for doc in documents:
        #     doc_id = hashlib.md5(doc.text.encode()).hexdigest()
        #     if doc_id not in existing_docs:
        #         if doc.metadata is None:
        #             doc.metadata = {}
        #         doc.metadata["doc_id"] = doc_id  
        #         new_documents.append(doc)

        # if not new_documents:
        #     logger.warning("No new documents to index")
        #     return []

        # Process new documents in batches
        inserted = 0
        for i in tqdm(range(0, len(documents), batch_size), desc="Indexing new documents"):
            batch = documents[i:i + batch_size]

            dense_embeddings, sparse_embeddings = self.compute_document_embeddings(batch)

            data = []
            for j, (doc, d, s) in enumerate(zip(batch, dense_embeddings, sparse_embeddings)):
                data.append({
                    "id": hashlib.md5(doc.text.encode()).hexdigest(),
                    "text": doc.text,
                    "dense_embedding": d,
                    "sparse_embedding": s.reshape(1, -1)
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

    def _create_collection_with_schema(self, collection_name: str, metadata_tags: List[str]):
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
        schema = CollectionSchema(fields=fields)
        index_params = self._create_index()
        logger.info(f"Creating collection {collection_name} with schema {schema}")
        self.client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)

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

        index_params.add_index(
            field_name="sparse_embedding",
            model_name=self.sparse_model_name,
            metric_type="IP",
            index_type="SPARSE_INVERTED_INDEX",
        )

        return index_params


def get_model_from_index(client: MilvusClient, index_name: str, collection_name: str = None):
    collection_name = collection_name or client.list_collections()[0]
    return client.describe_index(collection_name, index_name)['model_name']
