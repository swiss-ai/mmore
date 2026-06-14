"""
Simple vector database indexer using Milvus for document storage.
Supports multimodal documents with chunking capabilities.
"""

import gc
import json
import logging
from dataclasses import dataclass, field
from typing import List, Literal, Optional, cast

import scipy
from langchain_core.embeddings import Embeddings
from langchain_milvus.utils.sparse import BaseSparseEmbedding
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient
from tqdm import tqdm

from ..rag.model import DenseModel, DenseModelConfig, SparseModel, SparseModelConfig
from ..rag.model.dense.multimodal import MultimodalEmbeddings, _release_device_memory
from ..type import MultimodalSample
from ..utils import load_config

logger = logging.getLogger(__name__)


@dataclass
class DBConfig:
    uri: str = "./proc_demo.db"
    name: str = "my_db"


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
    sparse_model: Optional[BaseSparseEmbedding]
    client: MilvusClient

    _DEFAULT_FIELDS = ["id", "text", "dense_embedding", "sparse_embedding"]

    def __init__(
        self,
        dense_model_config: DenseModelConfig,
        sparse_model_config: SparseModelConfig,
        client: MilvusClient,
    ):
        self.dense_model_config = dense_model_config
        self.dense_model = DenseModel.from_config(dense_model_config)
        self.sparse_model_config = sparse_model_config
        if dense_model_config.is_multimodal:
            # Defer SPLADE until dense (Qwen) is unloaded — avoids two large models on MPS.
            self.sparse_model = None
            logger.info(
                "Multimodal dense model loaded; SPLADE will load on CPU after dense pass"
            )
        else:
            self.sparse_model = SparseModel.from_config(sparse_model_config)

        self.client = client

    def _ensure_sparse_model(self) -> BaseSparseEmbedding:
        if self.sparse_model is None:
            device = self.sparse_model_config.device
            if device is None and self.dense_model_config.is_multimodal:
                device = "cpu"
            sparse_config = SparseModelConfig(
                model_name=self.sparse_model_config.model_name,
                is_multimodal=self.sparse_model_config.is_multimodal,
                device=device,
            )
            logger.info(
                "Loading sparse model %s on device %s",
                sparse_config.model_name,
                sparse_config.device or "auto",
            )
            self.sparse_model = SparseModel.from_config(sparse_config)
        return self.sparse_model

    def _unload_dense_model(self) -> None:
        if self.dense_model_config.is_multimodal and self.dense_model is not None:
            logger.info("Unloading multimodal dense model to free accelerator memory")
            del self.dense_model
            self.dense_model = None  # type: ignore[assignment]
            _release_device_memory()
            gc.collect()

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
        collection_name: str = "my_docs",
        partition_name: Optional[str] = None,
        batch_size: int = 64,
    ):
        indexer = Indexer.from_config(config)
        indexer.index_documents(
            documents,
            collection_name=collection_name,
            partition_name=partition_name,
            batch_size=batch_size,
        )
        return indexer

    @staticmethod
    def _get_texts(documents: List[MultimodalSample], is_multimodal: bool) -> List[str]:
        if is_multimodal:
            return [MultimodalEmbeddings._multimodal_to_text(doc) for doc in documents]
        else:
            return [doc.text.replace("<attachment>", "") for doc in documents]

    @staticmethod
    def _row_for_insert(
        sample: MultimodalSample,
        dense_embedding: list[float],
        sparse_embedding,
    ) -> dict:
        return {
            "id": sample.id,
            "document_id": sample.document_id,
            "text": sample.text,
            "dense_embedding": dense_embedding,
            "sparse_embedding": sparse_embedding,
            "image_paths": json.dumps(
                [m.value for m in sample.modalities if m.type == "image"]
            ),
            **sample.metadata.to_dict(),
        }

    def _create_collection_with_schema(self, collection_name: str):
        """Create Milvus collection with fields for both embeddings."""
        fields = [
            FieldSchema(
                name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=128
            ),
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=128),
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

        schema = CollectionSchema(fields, enable_dynamic_field=True)

        self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=self._create_index(),
        )

    def _create_index(self):
        """Create index on the embeddings fields."""
        index_params = self.client.prepare_index_params()

        logger.info(
            f"Creating index for dense embeddings with model {self.dense_model_config.model_name}"
        )
        index_params.add_index(
            field_name="dense_embedding",
            model_name=self.dense_model_config.model_name,
            is_multimodal=self.dense_model_config.is_multimodal,
            metric_type="COSINE",
            index_type="IVF_FLAT",
            params={"nlist": 128},
        )

        logger.info(
            f"Creating index for sparse embeddings with model {self.sparse_model_config.model_name}"
        )
        index_params.add_index(
            field_name="sparse_embedding",
            model_name=self.sparse_model_config.model_name,
            is_multimodal=self.sparse_model_config.is_multimodal,
            metric_type="IP",
            index_type="SPARSE_INVERTED_INDEX",
        )
        return index_params

    def _index_documents(
        self,
        documents: List[MultimodalSample],
        collection_name: str = "my_docs",
        partition_name: Optional[str] = None,
        batch_size: int = 64,
    ) -> int:
        if self.dense_model_config.is_multimodal:
            return self._index_documents_multimodal_dense(
                documents,
                collection_name=collection_name,
                partition_name=partition_name,
            )

        inserted = 0
        for i in tqdm(
            range(0, len(documents), batch_size), desc="Indexing documents..."
        ):
            batch = documents[i : i + batch_size]

            dense_embeddings = self.dense_model.embed_documents(
                Indexer._get_texts(batch, self.dense_model_config.is_multimodal)
            )
            sparse_embeddings: scipy.sparse.coo_array = (
                self._ensure_sparse_model().embed_documents(
                    Indexer._get_texts(batch, self.sparse_model_config.is_multimodal)
                )
            )

            data = [
                Indexer._row_for_insert(sample, d, s)
                for sample, d, s in zip(batch, dense_embeddings, sparse_embeddings)
            ]

            batch_inserted = self.client.insert(
                data=data,
                collection_name=collection_name,
                partition_name=partition_name,
            )

            inserted += batch_inserted["insert_count"]

        return inserted

    def _index_documents_multimodal_dense(
        self,
        documents: List[MultimodalSample],
        collection_name: str = "my_docs",
        partition_name: Optional[str] = None,
    ) -> int:
        """Two-pass indexing: all dense (Qwen) first, unload, then SPLADE on CPU."""
        dense_rows: list[tuple[MultimodalSample, list[float]]] = []
        for sample in tqdm(documents, desc="Dense embeddings (Qwen)..."):
            dense_text = MultimodalEmbeddings._multimodal_to_text(sample)
            dense_embedding = self.dense_model.embed_documents([dense_text])[0]
            dense_rows.append((sample, dense_embedding))

        self._unload_dense_model()
        sparse_model = self._ensure_sparse_model()

        inserted = 0
        for sample, dense_embedding in tqdm(
            dense_rows, desc="Sparse embeddings + insert..."
        ):
            sparse_text = Indexer._get_texts(
                [sample], self.sparse_model_config.is_multimodal
            )[0]
            sparse_embedding = sparse_model.embed_documents([sparse_text])[0]

            batch_inserted = self.client.insert(
                data=[
                    Indexer._row_for_insert(sample, dense_embedding, sparse_embedding)
                ],
                collection_name=collection_name,
                partition_name=partition_name,
            )
            inserted += batch_inserted["insert_count"]

        return inserted

    def _log_collection_stats(self, collection_name: str):
        logger.info("-" * 50)
        logger.info("Collection stats (before inserting):")
        for k, v in self.client.get_collection_stats(collection_name).items():
            logger.info(f"  - {k}: {v}")
        logger.info("-" * 50)

    def index_documents(
        self,
        documents: List[MultimodalSample],
        collection_name: str = "my_docs",
        partition_name: Optional[str] = None,
        batch_size: int = 64,
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
            batch_size=batch_size,
        )

        self._log_collection_stats(collection_name)

        return inserted


def get_model_from_index(
    client: MilvusClient,
    index_name: Literal["dense_embedding", "sparse_embedding"],
    collection_name: Optional[str] = None,
) -> DenseModelConfig | SparseModelConfig:
    collection_name = collection_name or cast(List[str], client.list_collections())[0]
    if index_name == "dense_embedding":
        index_config = client.describe_index(collection_name, index_name)
        return DenseModelConfig(
            model_name=index_config["model_name"],
            is_multimodal=index_config.get("is_multimodal", "False") == "True",
        )
    elif index_name == "sparse_embedding":
        index_config = client.describe_index(collection_name, index_name)
        return SparseModelConfig(
            model_name=index_config["model_name"],
            is_multimodal=index_config.get("is_multimodal", "False") == "True",
        )
    else:
        raise ValueError(
            f"Invalid index_name: {index_name}. Must be 'dense_embedding' or 'sparse_embedding'."
        )
