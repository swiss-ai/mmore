"""
Vector database retriever using Milvus for efficient similarity search.
Works in conjunction with the Indexer class for document retrieval.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, cast, get_args

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_milvus.utils.sparse import BaseSparseEmbedding
from pymilvus import AnnSearchRequest, MilvusClient, WeightedRanker

from ..index.indexer import DBConfig, get_model_from_index
from ..utils import load_config
from .model.dense.base import DenseModel, DenseModelConfig
from .model.sparse.base import SparseModel, SparseModelConfig

logger = logging.getLogger(__name__)


@dataclass
class RetrieverConfig:
    db: DBConfig = field(default_factory=DBConfig)
    hybrid_search_weight: float = 0.5
    k: int = 1
    collection_name: str = "my_docs"


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
            config = load_config(config, RetrieverConfig)

        # Init the client
        client = MilvusClient(uri=config.db.uri, db_name=config.db.name)

        if not client.has_collection(config.collection_name):
            raise ValueError(
                f"The Milvus database has not been initialized yet / does not have a collection {config.collection_name}. "
                "Ensure the path is valid with a database that was already populated with the indexer."
            )

        # Init models
        dense_model_config = cast(
            DenseModelConfig,
            get_model_from_index(client, "dense_embedding", config.collection_name),
        )
        dense_model = DenseModel.from_config(dense_model_config)
        logger.info(f"Loaded dense model: {dense_model_config}")

        sparse_model_config = cast(
            SparseModelConfig,
            get_model_from_index(client, "sparse_embedding", config.collection_name),
        )
        sparse_model = SparseModel.from_config(sparse_model_config)
        logger.info(f"Loaded sparse model: {sparse_model_config}")

        return cls(
            dense_model=dense_model,
            sparse_model=sparse_model,
            client=client,
            hybrid_search_weight=config.hybrid_search_weight,
            k=config.k,
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
        partition_names: Optional[List[str]] = None,
        min_score: float = -1.0,  # -1.0 is the minimum possible score anyway
        k: int = 1,
        output_fields: List[str] = ["text"],
        search_type: str = "hybrid",  # Options: "dense", "sparse", "hybrid"
        document_ids: List[str] = [],  # Optional: candidate doc IDs to restrict search
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k similar documents for a given query.

        This method computes dense and sparse query embeddings and builds two seperate search requests (one for dense and one for sparse). If candidate document IDs are provided, a filter expression is attacked to both search requests to restrict the search to those documents only.

        Args:
            query: Search query string
            collection_name: the Milvus collection to search in
            partition_names: Specific partitions within the collection
            min_score: Minimal similarity score
            k: Number of documents to retrieve
            output_fields: Fields to return in results
            search_type: Type of search to perform ("dense", "sparse", or "hybrid")
            document_ids: Candidate document Ids to filter the search

        Returns:
            The raw search results (a nested list of dictionaries) returned by Milvus
        """
        if k == 0:
            return []

        assert search_type in get_args(self._search_types), (
            f"Invalid search_type: {search_type}. Must be 'dense', 'sparse', or 'hybrid'"
        )
        search_weight = self._search_weights.get(search_type, self.hybrid_search_weight)

        # Combute both dense and sparse embeddings for the query
        dense_embedding, sparse_embedding = self.compute_query_embeddings(query)

        # Build a filter expression if candidate document IDs are provided.
        # The expression will restrict the search to documents with Ids in the given list
        if document_ids:
            # Create a comme-seperated string of quoted document IDs
            ids_str = ",".join(f'"{d}"' for d in document_ids)
            expr = f"document_id in [{ids_str}]"
        else:
            # No filtering if doc_ids is not provided
            expr = None

        # Prepare the search request for the dense embeddings
        # This request searches within the "dense_embedding" field using cosine similarity
        search_param_1 = {
            "data": dense_embedding,  # Query vector
            "anns_field": "dense_embedding",  # Field to search in
            "param": {
                "metric_type": "COSINE",  # This parameter value must be identical to the one used in the collection schema
                "params": {"nprobe": 10},
            },
            "limit": k,
            "expr": expr,
        }

        # Attach the filtering expression if available
        if expr is not None:
            search_param_1["expr"] = expr

        # Prepare the search request for the sparse embeddings.
        # This request searches within the "sparse_embedding" field using the inner product (IP)
        search_param_2 = {
            "data": sparse_embedding,  # Query vector
            "anns_field": "sparse_embedding",  # Field to search in
            "param": {
                "metric_type": "IP",  # This parameter value must be identical to the one used in the collection schema
                "params": {"nprobe": 10},
            },
            "limit": k,
            "expr": expr,
        }

        # Attach the filtering expression if available
        if expr is not None:
            search_param_2["expr"] = expr

        # Create AnnSearchRequest objects from the parameter dictionaries
        request_1 = AnnSearchRequest(**search_param_1)
        request_2 = AnnSearchRequest(**search_param_2)

        # Call the Milvus hybrid_search to perform both searches and then rerank results.
        # The WeightedRanker combined the scores from the dense and sparse searches
        res = self.client.hybrid_search(
            reqs=[request_1, request_2],  # List of AnnSearchRequests
            ranker=WeightedRanker(
                search_weight, 1 - search_weight
            ),  # Reranking strategy
            limit=k,
            output_fields=output_fields,
            collection_name=collection_name,
            partition_names=partition_names,
        )

        # Apply the threshold of min_score
        return list(
            filter(lambda x: x["distance"] >= min_score, res[0])
        )  # 0 because there is only one query

    def batch_retrieve(
        self,
        queries: List[str],
        collection_name: str = "my_docs",
        partition_names: List[str] = [],
        min_score: float = -1.0,  # -1.0 is the minimum possible score anyway
        k: int = 1,
        output_fields: List[str] = ["text"],
        search_type: str = "hybrid",
    ) -> List[List[Dict[str, Any]]]:
        """
        Batch retrieve documents for multiple queries.

        Args:
            queries: List of search query strings
            collection_name: Name of the collection in which the research has to be done
            partition_names: Names of the partitions in which the research has to be done
            min_score: Minimal score of the documents to retrieve
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
                min_score=min_score,
                k=k,
                output_fields=output_fields,
                search_type=search_type,
            )
            all_results.append(results)
        return all_results

    def _get_relevant_documents(
        self,
        query: str | Dict[str, Any],
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        """Retrieve relevant documents from Milvus. This is necessary for compatibility with LangChain."""

        if isinstance(query, str):
            query_input: str = query
            collection_name: str = kwargs.get("collection_name", "my_docs")
            document_ids: List[str] = kwargs.get("document_ids", [])
        else:
            if "input" not in query:
                raise ValueError("Missing query input")

            query_input: str = query["input"]
            collection_name: str = query.get("collection_name", "my_docs")
            document_ids: List[str] = query.get("document_ids", [])

        partition_names: Optional[List[str]] = kwargs.get("partition_names", None)
        min_score: float = kwargs.get("min_score", -1.0)
        k: int = kwargs.get("k", self.k)

        if k == 0:
            return []

        results = self.retrieve(
            query=query_input,
            collection_name=collection_name,
            partition_names=partition_names,
            min_score=min_score,
            k=k,
            document_ids=document_ids,
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
        return [parse_result(result, i) for i, result in enumerate(results)]

    def get_documents_by_ids(
        self, doc_ids: list[str], collection_name: str = "my_docs"
    ) -> list[Document]:
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
                logger.warning(
                    f"Warning: The following IDs were not found: {missing_ids}"
                )

        # Convert the returned rows into Document objects
        # Each document contains the text and metadata
        return [
            Document(page_content=row["text"], metadata={"id": row["id"]})
            for row in results
        ]
