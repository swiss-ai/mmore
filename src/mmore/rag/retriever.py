"""
Vector database retriever using Milvus for efficient similarity search.
Works in conjunction with the Indexer class for document retrieval.
"""

from typing import List, Dict, Any, Tuple, Literal, get_args
from dataclasses import dataclass, field

from mmore.rag.model.dense.base import DenseModel
from mmore.rag.model.sparse.base import SparseModel
from ..utils import load_config

from mmore.index.indexer import get_model_from_index
from mmore.index.indexer import DBConfig
from mmore.rag.model import DenseModel, SparseModel

from pymilvus import MilvusClient, WeightedRanker, AnnSearchRequest

from langchain_core.embeddings import Embeddings

from langchain_milvus.utils.sparse import BaseSparseEmbedding
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

import logging
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

    _search_weights = {
        "dense": 0,
        "sparse": 1
    }

    @classmethod
    def from_config(cls, config: str | RetrieverConfig):
        if isinstance(config, str):
            config = load_config(config, RetrieverConfig)

        # Init the client
        client = MilvusClient(uri=config.db.uri, db_name=config.db.name)

        # Init models
        dense_model_config = get_model_from_index(client, "dense_embedding")
        dense_model = DenseModel.from_config(dense_model_config)
        logger.info(f"Loaded dense model: {dense_model_config}")

        sparse_model_config = get_model_from_index(client, "sparse_embedding")
        sparse_model = SparseModel.from_config(sparse_model_config)
        logger.info(f"Loaded sparse model: {sparse_model_config}")

        return cls(
            dense_model=dense_model,
            sparse_model=sparse_model,
            client=client,
            hybrid_search_weight=config.hybrid_search_weight,
            k=config.k,
        )

    def compute_query_embeddings(self, query: str) -> Tuple[List[float], List[float]]:
        """Compute both dense and sparse embeddings for the query."""
        dense_embedding = [self.dense_model.embed_query(query)]
        sparse_embedding = [self.sparse_model.embed_query(query)]

        return dense_embedding, sparse_embedding

    # TODO: [FEATURE] minimal score for retrieval
    def retrieve(
            self,
            query: str,
            collection_name: str = 'my_docs',
            partition_names: List[str] = None,
            k: int = 1,
            search_type: str = "hybrid",  # Options: "dense", "sparse", "hybrid"
            doc_ids: List[str] = None
    ) -> List[Dict[str, Any]]:
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
    
        assert search_type in get_args(
            self._search_types), f"Invalid search_type: {search_type}. Must be 'dense', 'sparse', or 'hybrid'"
        search_weight = self._search_weights.get(search_type, self.hybrid_search_weight)

        dense_embedding, sparse_embedding = self.compute_query_embeddings(query)

        if doc_ids:
            ids_str = ",".join(f'"{d}"' for d in doc_ids)
            expr = f"id in [{ids_str}]"
        else:
            expr = None

        search_param_1 = {
            "data": dense_embedding,  # Query vector
            "anns_field": "dense_embedding",  # Field to search in
            "param": {
                "metric_type": "COSINE",  # This parameter value must be identical to the one used in the collection schema
                "params": {"nprobe": 10}
            },
            "limit": k,
        }

        if expr is not None:
            search_param_1["expr"] = expr

        search_param_2 = {
            "data": sparse_embedding,  # Query vector
            "anns_field": "sparse_embedding",  # Field to search in
            "param": {
                "metric_type": "IP",  # This parameter value must be identical to the one used in the collection schema
                "params": {"nprobe": 10}
            },
            "limit": k,
        }

        if expr is not None:
            search_param_2["expr"] = expr

        request_1 = AnnSearchRequest(**search_param_1)
        request_2 = AnnSearchRequest(**search_param_2)

        return self.client.hybrid_search(
            reqs=[request_1, request_2],  # List of AnnSearchRequests
            ranker=WeightedRanker(search_weight, 1 - search_weight),  # Reranking strategy
            limit=k,
            output_fields=["text"],
            collection_name=collection_name,
            partition_names=partition_names,
        )

    def batch_retrieve(
            self,
            queries: List[str],
            collection_name: str = 'my_docs',
            partition_names: List[str] = None,
            k: int = 1,
            output_fields: List[str] = ["text"],
            search_type: str = "hybrid"
    ) -> List[List[Dict[str, Any]]]:
        """
        Batch retrieve documents for multiple queries.
        
        Args:
            queries: List of search query strings
            k: Number of documents to retrieve per query
            output_fields: Fields to return in results
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
                output_fields=output_fields,
                search_type=search_type
            )
            all_results.append(results)
        return all_results

    def _get_relevant_documents(
            self,
            query: Dict[str, Any],
            *,
            run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieve relevant documents from Milvus. This is necessary for compatibility with LangChain."""
        if self.k == 0:
            return []

        # For compatibility
        if isinstance(query.get('partition_name', None), str):
            query['partition_name'] = [query['partition_name']]

        results = self.retrieve(
            query=query['input'],
            collection_name=query.get('collection_name', 'my_docs'),
            partition_names=query.get('partition_name', None),
            k=self.k
        )

        return [
            Document(
                page_content=result['entity']['text'],
                metadata={'id': result['id'], 'rank': i + 1, 'similarity': result['distance']}
            )
            for i, result in enumerate(results[0])
        ]
    
    def get_documents_by_ids(self, doc_ids: list[str], collection_name: str = 'my_docs') -> list[Document]:
        """
        Fetch documents with the specified IDs from Milvus (if they exist).
        """

        # If no document IDs are provided, return empty list
        if not doc_ids:
            return []
        
        # Build a comma-seperated string of document IDs, each wrapped in double quotes
        # example: if doc_ids = ["doc1", "doc2"],
        # ids_str becomes '"doc1", "doc2"'
        # This format is required for querying the "id" PK field in Milvus which is a VARCHAR
        ids_str = ",".join(f'"{d}"' for d in doc_ids)
        expr = f"id in [{ids_str}]"

        logger.info(f"Querying Milvus by expr: {expr}")

        results = self.client.query(collection_name, expr, ["id", "text"])

        # If the query returned no results, log a warning
        if not results:
            logger.warning(f"Warning: No documents found for the given IDs: {doc_ids}")
        else:
            # Extract IDs of retrieved documents.
            found_ids = {row["id"] for row in results}
            # Determine which requested IDs were not found (if any)
            missing_ids = [doc_id for doc_id in doc_ids if doc_id not in found_ids]
            if missing_ids:
                logger.warning(f"Warning: The following IDs were not found: {missing_ids}")

        # Convert the returned rows into Document objects
        # Each document contains the text and metadata
        return [
            Document(
                page_content=row["text"],
                metadata={"id": row["id"]}
            ) for row in results
        ]
