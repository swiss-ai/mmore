"""
Vector database retriever using Milvus for efficient similarity search.
Works in conjunction with the Indexer class for document retrieval.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Tuple, cast, get_args

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_milvus.utils.sparse import BaseSparseEmbedding
from pymilvus import AnnSearchRequest, MilvusClient, WeightedRanker

from ..index.indexer import DBConfig, get_model_from_index
from ..utils import load_config
from .model import DenseModel, DenseModelConfig, SparseModel, SparseModelConfig

logger = logging.getLogger(__name__)


@dataclass
class RetrieverConfig:
    db: DBConfig = field(default_factory=DBConfig)
    hybrid_search_weight: float = 0.5
    k: int = 1


class Retriever(BaseRetriever):
    """Handles similarity-based document retrieval from Milvus."""

    dense_model: Embeddings
    sparse_model: BaseSparseEmbedding
    client: MilvusClient
    hybrid_search_weight: float
    k: int

    _search_types = Literal["dense", "sparse", "hybrid"]

    _search_weights = {"dense": 0, "sparse": 1}

    @classmethod
    def from_config(cls, config: str | RetrieverConfig):
        if isinstance(config, str):
            config_obj: RetrieverConfig = load_config(config, RetrieverConfig)
        else:
            config_obj = config

        # Init the client
        client = MilvusClient(uri=config_obj.db.uri, db_name=config_obj.db.name)

        # Init models
        dense_model_config: DenseModelConfig = cast(
            DenseModelConfig, get_model_from_index(client, "dense_embedding")
        )
        dense_model = DenseModel.from_config(dense_model_config)
        logger.info(f"Loaded dense model: {dense_model_config}")

        sparse_model_config: SparseModelConfig = cast(
            SparseModelConfig, get_model_from_index(client, "sparse_embedding")
        )
        sparse_model = SparseModel.from_config(sparse_model_config)
        logger.info(f"Loaded sparse model: {sparse_model_config}")

        return cls(
            dense_model=dense_model,
            sparse_model=sparse_model,
            client=client,
            hybrid_search_weight=config_obj.hybrid_search_weight,
            k=config_obj.k,
        )

    def compute_query_embeddings(
        self, query: str
    ) -> Tuple[List[List[float]], List[Dict[int, float]]]:
        """Compute both dense and sparse embeddings for the query."""
        dense_embedding = [self.dense_model.embed_query(query)]
        sparse_embedding = [self.sparse_model.embed_query(query)]

        return dense_embedding, sparse_embedding

    def retrieve(
        self,
        query: str,
        collection_name: str = "my_docs",
        partition_names: List[str] = [],
        k: int = 1,
        search_type: str = "hybrid",  # Options: "dense", "sparse", "hybrid"
    ) -> List[List[Dict[str, Any]]]:
        """
        Retrieve top-k similar documents for a given query.

        Args:
            query: Search query string
            k: Number of documents to retrieve
            output_fields: Fields to return in results
            search_type: Type of search to perform ("dense", "sparse", or "hybrid")

        Returns:
            List of matching documents with specified output fields
        """
        if k == 0:
            return []

        assert search_type in get_args(self._search_types), (
            f"Invalid search_type: {search_type}. Must be 'dense', 'sparse', or 'hybrid'"
        )
        search_weight = self._search_weights.get(search_type, self.hybrid_search_weight)

        dense_embedding, sparse_embedding = self.compute_query_embeddings(query)

        search_param_1 = {
            "data": dense_embedding,  # Query vector
            "anns_field": "dense_embedding",  # Field to search in
            "param": {
                "metric_type": "COSINE",  # This parameter value must be identical to the one used in the collection schema
                "params": {"nprobe": 10},
            },
            "limit": k,
        }

        search_param_2 = {
            "data": sparse_embedding,  # Query vector
            "anns_field": "sparse_embedding",  # Field to search in
            "param": {
                "metric_type": "IP",  # This parameter value must be identical to the one used in the collection schema
                "params": {"nprobe": 10},
            },
            "limit": k,
        }

        request_1 = AnnSearchRequest(**search_param_1)
        request_2 = AnnSearchRequest(**search_param_2)

        return self.client.hybrid_search(
            reqs=[request_1, request_2],  # List of AnnSearchRequests
            ranker=WeightedRanker(
                search_weight, 1 - search_weight
            ),  # Reranking strategy
            limit=k,
            output_fields=["text"],
            collection_name=collection_name,
            partition_names=partition_names,
        )

    def batch_retrieve(
        self,
        queries: List[str],
        collection_name: str = "my_docs",
        partition_names: List[str] = [],
        k: int = 1,
        search_type: str = "hybrid",
    ) -> List[List[Dict[str, Any]]]:
        """
        Batch retrieve documents for multiple queries.

        Args:
            queries: List of search query strings
            k: Number of documents to retrieve per query
            search_type: Type of search to perform

        Returns:
            List of results for each query
        """
        all_results = []
        for query in queries:
            results = self.retrieve(
                query=query,
                collection_name=collection_name,
                partition_names=partition_names,
                k=k,
                # output_fields=output_fields,
                search_type=search_type,
            )
            all_results.append(results)
        return all_results

    def _get_relevant_documents(  # type: ignore
        self, query: Dict[str, Any], *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieve relevant documents from Milvus. This is necessary for compatibility with LangChain."""
        if self.k == 0:
            return []

        # For compatibility
        if isinstance(query.get("partition_name", None), str):
            query["partition_name"] = [query["partition_name"]]

        results = self.retrieve(
            query=query["input"],
            collection_name=query.get("collection_name", "my_docs"),
            partition_names=query.get("partition_name", []),
            k=self.k,
        )

        def parse_result(result: Dict[str, Any], i: int) -> Document:
            return Document(
                page_content=result["entity"]["text"],
                metadata={
                    "id": result["id"],
                    "rank": i + 1,
                    "similarity": result["distance"],
                },
            )

        # 0 because there is only one query
        return [parse_result(result, i) for i, result in enumerate(results[0])]
