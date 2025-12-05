import concurrent.futures
import logging
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from pymilvus import DataType, MilvusClient
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[MilvusColpaliManager 🧠 -- %(asctime)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


class MilvusColpaliManager:
    """
    Handles all Milvus operations for ColPali embeddings (both indexing and retrieval).
    Uses a local Milvus storage by default.
    """

    def __init__(
        self,
        db_path: Union[str, Path] = "./milvus_data",
        collection_name: str = "pdf_pages",
        dim: int = 128,
        metric_type: str = "IP",
        create_collection: bool = False,
    ):
        """
        Initialize a local Milvus database connection and collection handler.
        """
        self.uri = str(Path(db_path).resolve())
        self.collection_name = collection_name
        self.dim = dim
        self.metric_type = metric_type

        logger.info(f"Connecting to Milvus local instance at {self.uri}")
        self.client = MilvusClient(uri=self.uri)

        if create_collection:
            self.create_collection(overwrite=True)
            logger.info(f"Created new collection '{self.collection_name}'")
        elif self.client.has_collection(self.collection_name):
            self.client.load_collection(self.collection_name)
            logger.info(f"Loaded existing collection '{self.collection_name}'")
        else:
            raise ValueError(
                f"Collection '{self.collection_name}' does not exist. "
                "Use create_collection=True to create it."
            )

    def create_collection(self, overwrite: bool = False):
        """
        Create a new Milvus collection with ColPali embedding schema.
        """
        if overwrite and self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
            logger.info(f"Dropped existing collection '{self.collection_name}'")

        schema = self.client.create_schema(auto_id=True)
        schema.add_field("pk", DataType.INT64, is_primary=True)
        schema.add_field("pdf_path", DataType.VARCHAR, max_length=1024)
        schema.add_field("page_number", DataType.INT64)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.dim)

        self.client.create_collection(
            collection_name=self.collection_name, schema=schema
        )
        logger.info(f"Created collection schema for '{self.collection_name}'")

    def create_index(self):
        """
        Create a vector index on the embedding field.
        Should be called AFTER data insertion for better performance.
        """
        # Check if index already exists
        try:
            index_info = self.client.list_indexes(self.collection_name)
            if index_info:
                logger.info(f"Index already exists: {index_info}")
                return True
        except Exception as e:
            logger.warning(f"Could not check existing indexes: {e}")

        logger.info("Creating vector index on 'embedding' field...")

        # Create index parameters
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_name="embedding_index",
            index_type="FLAT",
            metric_type=self.metric_type,
        )

        # Create index
        try:
            self.client.create_index(
                collection_name=self.collection_name,
                index_params=index_params,
            )
            logger.info("Created vector index on 'embedding' field")
            return True
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            return False

    def insert_from_dataframe(self, df: pd.DataFrame, batch_size: int = 500):
        """
        Insert all subvectors from each row of the DataFrame into Milvus in batches.
        """
        required_cols = {"pdf_path", "page_number", "embedding"}
        if not required_cols <= set(df.columns):
            raise ValueError(
                f"DataFrame missing required columns: {required_cols - set(df.columns)}"
            )

        logger.info(f"Preparing {len(df)} pages...")

        data = []
        for _, row in tqdm(
            df.iterrows(), total=len(df), desc="Preparing vectors", ncols=100
        ):
            emb = row["embedding"]

            if isinstance(emb, np.ndarray) and emb.dtype == object:
                emb = np.stack(emb)
            elif isinstance(emb, list) and isinstance(emb[0], np.ndarray):
                emb = np.stack(emb)
            elif isinstance(emb, list):
                emb = np.array(emb, dtype=np.float32)
            if emb.ndim == 1:
                emb = emb[np.newaxis, :]

            for vec in emb:
                data.append(
                    {
                        "pdf_path": row["pdf_path"],
                        "page_number": int(row["page_number"]),
                        "embedding": np.asarray(vec, dtype=np.float32).tolist(),
                    }
                )

        # Validate embedding dimensions
        for i, row in enumerate(data[:5]):
            vlen = len(row["embedding"])
            if vlen != self.dim:
                logger.error(f"Row {i}: got embedding len={vlen}, expected {self.dim}")
            else:
                logger.debug(f"Row {i}: embedding dimension ok ({vlen})")

        total_vecs = len(data)
        logger.info(f"Inserting {total_vecs} vectors in batches of {batch_size}...")

        for i in tqdm(
            range(0, total_vecs, batch_size), desc="Inserting into Milvus", ncols=100
        ):
            batch = data[i : i + batch_size]
            try:
                self.client.insert(self.collection_name, batch)
            except Exception as e:
                logger.error(f"Failed to insert batch {i // batch_size}: {e}")
                raise

        logger.info(f"✅ Insert complete — {total_vecs} vectors inserted.")

    def search_embeddings(self, query_embeddings, top_k=3, max_workers=4):
        """
        Search for similar embeddings using vector similarity search.
        """
        # Handle query shape
        arr = np.asarray(query_embeddings, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr.squeeze(0)
        elif arr.ndim == 1:
            arr = arr[np.newaxis, :]

        # Perform initial vector search
        results = self.client.search(
            collection_name=self.collection_name,
            data=arr,
            anns_field="embedding",
            limit=top_k * 5,
            # As each page is embedded as a multi vector, each page is represented multiple times inside Milvus, each with one column of the multi vector
            # Thus there can be duplicates in the ids, and top_k * 5 allows us to ensure the retrieval of top_k distinct pages
            output_fields=["pdf_path", "page_number"],
            search_params={"metric_type": self.metric_type, "params": {}},
        )

        # Collect unique candidate pages
        candidates = set()
        for hits in results:
            for hit in hits:
                entity_dict = hit.get("entity", {})
                key = (entity_dict.get("pdf_path"), entity_dict.get("page_number"))
                candidates.add(key)

        logger.info(f"Found {len(candidates)} candidate pages for reranking.")

        def rerank_page(pdf_path, page_number, query_vecs):
            # Get all subvectors for this page
            docs = self.client.query(
                collection_name=self.collection_name,
                filter="pdf_path == $pdf_path and page_number == $page_number",
                output_fields=["embedding", "pdf_path"],
                limit=10000,
                params={"pdf_path": pdf_path, "page_number": page_number},
            )
            if not docs:
                return (None, pdf_path, page_number)

            doc_vecs = np.vstack([d["embedding"] for d in docs]).astype(np.float32)
            score = np.dot(query_vecs, doc_vecs.T).max(1).sum()
            return (score, pdf_path, page_number)

        reranked = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(rerank_page, pdf_path, page_number, arr)
                for (pdf_path, page_number) in candidates
            ]
            for f in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="🔁 Reranking",
            ):
                try:
                    score, pdf_path, page_number = f.result()
                    if score is not None:
                        reranked.append(
                            {
                                "pdf_path": pdf_path,
                                "page_number": page_number,
                                "score": score,
                            }
                        )
                except Exception as e:
                    logger.error(f"Rerank failed: {e}")

        # Sort and return top-k pages
        reranked.sort(key=lambda x: x["score"], reverse=True)
        for i, item in enumerate(reranked):
            item["rank"] = i + 1
        return reranked[:top_k]

    def drop_collection(self):
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
            logger.info(f"Dropped collection '{self.collection_name}'")
        else:
            logger.warning(f"Collection '{self.collection_name}' not found.")

    def close(self):
        if hasattr(self, "client"):
            self.client.close()
            logger.info("Closed Milvus client connection")
