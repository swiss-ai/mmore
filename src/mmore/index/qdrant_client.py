"""
QdrantMilvusClient — a MilvusClient-shaped adapter backed by qdrant-client
local mode.

Used as a drop-in replacement for ``pymilvus.MilvusClient`` so that mmore's
Indexer / Retriever / index-API code paths work unchanged when
``db.backend: qdrant`` is set in the YAML config.

Why
---
``milvus-lite`` ships x86_64-only wheels; on ARM64 (NVIDIA GH200, Apple
Silicon, …) the embedded mode that mmore relies on by default cannot run.
Qdrant local mode works on any architecture.

Implemented surface
-------------------
Only the subset of MilvusClient that mmore actually calls:

* ``has_collection`` / ``list_collections`` / ``get_collection_stats``
* ``prepare_index_params`` (returns a small capture object)
* ``create_collection`` (introspects the supplied CollectionSchema for the
  dense vector dimension and persists the index params for later
  ``describe_index`` calls)
* ``describe_index``
* ``insert`` / ``delete`` / ``flush`` / ``query``
* ``hybrid_search`` (two AnnSearchRequest legs fused with RRF)

Caveats
-------
* Hybrid fusion uses Qdrant's RRF (Reciprocal Rank Fusion). The
  ``WeightedRanker`` passed by mmore is accepted but its weights are
  ignored — rankings will differ slightly from Milvus's WeightedRanker.
  In practice top-k overlap is high; if you need exact weight semantics,
  use the Milvus backend.
* Point IDs must be UUIDs in Qdrant; mmore's string chunk IDs are mapped
  via :func:`uuid.uuid5` deterministically and the original is stored in
  the payload under ``_str_id``.
* Logical partitions (``partition_name=``) are emulated via a payload
  field — Qdrant has no native partition concept.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Namespace for deterministic UUID generation from arbitrary string IDs.
_UUID_NS = uuid.NAMESPACE_URL

# Reserved payload keys used by the adapter for bookkeeping.
_STR_ID_KEY = "_str_id"  # original mmore string chunk id
_PARTITION_KEY = "_partition"  # logical partition emulation
_META_KEY = "_is_meta"  # marks the model-config sentinel point
_INDEX_PARAMS_KEY = "_index_params"


def _str_to_uuid(s: str) -> str:
    return str(uuid.uuid5(_UUID_NS, s))


def _meta_point_uuid(collection_name: str) -> str:
    return _str_to_uuid(f"_mmore_meta_{collection_name}")


def _coerce_sparse(value: Any) -> Dict[int, float]:
    """Normalise a sparse vector to a ``{index: value}`` dict.

    Accepts dicts (from BaseSparseEmbedding.embed_query), 1-row scipy sparse
    arrays (from SpladeSparseEmbedding.embed_documents), and anything with
    ``indices`` / ``data`` attributes.
    """
    if isinstance(value, dict):
        return {int(k): float(v) for k, v in value.items()}
    if hasattr(value, "tocoo"):
        coo = value.tocoo()
        return {int(c): float(v) for c, v in zip(coo.col, coo.data)}
    if hasattr(value, "indices") and hasattr(value, "data"):
        return {int(c): float(v) for c, v in zip(value.indices, value.data)}
    raise TypeError(f"Cannot convert {type(value).__name__} to sparse vector dict")


class _IndexParamsCapture:
    """Captures ``add_index`` calls so the adapter can persist the metadata.

    The returned object mimics the ``IndexParams`` returned by
    ``MilvusClient.prepare_index_params()`` — mmore only ever calls
    ``add_index(...)`` on it.
    """

    def __init__(self) -> None:
        self.fields: Dict[str, Dict[str, Any]] = {}

    def add_index(
        self,
        field_name: str,
        model_name: str = "",
        is_multimodal: bool = False,
        metric_type: str = "",
        index_type: str = "",
        params: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> None:
        # is_multimodal is stored as the string "True"/"False" because that's
        # what MilvusClient.describe_index returns and the upstream Indexer
        # parses (see get_model_from_index in indexer.py).
        self.fields[field_name] = {
            "model_name": str(model_name),
            "is_multimodal": "True" if is_multimodal else "False",
            "metric_type": metric_type,
            "index_type": index_type,
            "params": params or {},
        }


def _milvus_filter_to_qdrant(
    expr: Optional[str],
    partition_names: Optional[List[str]] = None,
):
    """Convert a Milvus filter expression to a ``qdrant_client.models.Filter``.

    Supported patterns (the only ones mmore uses):

    * ``field in ["a", "b", ...]``
    * ``field == 'value'``
    * ``field != 'value'`` (including ``field != ""``)

    Anything else raises ``ValueError`` so unsupported patterns surface
    loudly instead of silently returning the wrong rows.
    """
    from qdrant_client.models import (
        FieldCondition,
        Filter,
        IsEmptyCondition,
        MatchAny,
        MatchValue,
        PayloadField,
    )

    must: List[Any] = []
    must_not: List[Any] = []

    if expr:
        e = expr.strip()
        m = re.fullmatch(r"(\w+)\s+in\s+\[(.+)\]", e, re.DOTALL)
        if m:
            values = re.findall(r'"([^"]*)"', m.group(2)) or re.findall(
                r"'([^']*)'", m.group(2)
            )
            must.append(FieldCondition(key=m.group(1), match=MatchAny(any=values)))
        else:
            m = re.fullmatch(r"(\w+)\s*==\s*[\"']([^\"']*)[\"']", e)
            if m:
                must.append(
                    FieldCondition(key=m.group(1), match=MatchValue(value=m.group(2)))
                )
            else:
                m = re.fullmatch(r"(\w+)\s*!=\s*[\"']([^\"']*)[\"']", e)
                if m:
                    field, value = m.group(1), m.group(2)
                    if value == "":
                        must_not.append(
                            IsEmptyCondition(is_empty=PayloadField(key=field))
                        )
                        must_not.append(
                            FieldCondition(key=field, match=MatchValue(value=""))
                        )
                    else:
                        must_not.append(
                            FieldCondition(key=field, match=MatchValue(value=value))
                        )
                else:
                    raise ValueError(
                        f"QdrantMilvusClient cannot translate filter expression "
                        f"{expr!r}. Supported: 'field in [...]', "
                        "'field == X', 'field != X'."
                    )

    # Always exclude the sentinel meta point from user-facing reads.
    must_not.append(FieldCondition(key=_META_KEY, match=MatchValue(value=True)))

    if partition_names:
        must.append(
            FieldCondition(
                key=_PARTITION_KEY, match=MatchAny(any=list(partition_names))
            )
        )

    return Filter(must=must or None, must_not=must_not or None)


class QdrantMilvusClient:
    """MilvusClient-compatible facade backed by qdrant-client local mode."""

    def __init__(self, uri: str, db_name: Optional[str] = None, **_: Any) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as e:
            raise ImportError(
                "qdrant-client is not installed. Install the Qdrant backend with: "
                "pip install mmore[qdrant]"
            ) from e

        # `db_name` and any other extra kwargs (e.g. enable_sparse) are
        # accepted for MilvusClient signature compatibility; they have no
        # equivalent on the Qdrant side.
        self._db_name = db_name
        if uri.startswith("http://") or uri.startswith("https://"):
            logger.info("Connecting to Qdrant server at %s", uri)
            self._qdrant = QdrantClient(url=uri)
        else:
            logger.info("Initialising Qdrant local backend at %s", uri)
            self._qdrant = QdrantClient(path=uri)

    # ── Collection lifecycle ──────────────────────────────────────────────

    def has_collection(self, collection_name: str) -> bool:
        return self._qdrant.collection_exists(collection_name)

    def list_collections(self) -> List[str]:
        return [c.name for c in self._qdrant.get_collections().collections]

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        info = self._qdrant.get_collection(collection_name)
        return {"row_count": info.points_count or 0}

    def prepare_index_params(self) -> _IndexParamsCapture:
        return _IndexParamsCapture()

    def create_collection(
        self,
        collection_name: str,
        schema: Any = None,
        index_params: Any = None,
        **_: Any,
    ) -> None:
        from qdrant_client.models import (
            Distance,
            SparseIndexParams,
            SparseVectorParams,
            VectorParams,
        )

        dense_dim = self._extract_dense_dim(schema)

        self._qdrant.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": VectorParams(size=dense_dim, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=SparseIndexParams())
            },
        )

        if isinstance(index_params, _IndexParamsCapture):
            self._upsert_meta(collection_name, index_params, dense_dim)

    @staticmethod
    def _extract_dense_dim(schema: Any) -> int:
        """Pull the dense vector dimension from a ``CollectionSchema``."""
        if schema is None:
            raise ValueError("create_collection requires a CollectionSchema")
        for f in getattr(schema, "fields", []):
            if getattr(f, "name", "") == "dense_embedding":
                dim = getattr(f, "dim", None)
                if dim is None:
                    params = getattr(f, "params", {}) or {}
                    dim = params.get("dim")
                if dim is None:
                    raise ValueError("dense_embedding field has no `dim`")
                return int(dim)
        raise ValueError("CollectionSchema does not contain a 'dense_embedding' field")

    # ── Index metadata round-trip ─────────────────────────────────────────

    def describe_index(self, collection_name: str, field_name: str) -> Dict[str, Any]:
        """Return the model_name / is_multimodal stored at create time."""
        meta_id = _meta_point_uuid(collection_name)
        results = self._qdrant.retrieve(
            collection_name=collection_name,
            ids=[meta_id],
            with_payload=True,
        )
        if not results:
            raise RuntimeError(
                f"No index metadata found for collection '{collection_name}'. "
                "The collection may have been created outside of mmore."
            )
        payload = results[0].payload or {}
        per_field = payload.get(_INDEX_PARAMS_KEY, {}).get(field_name)
        if not per_field:
            raise RuntimeError(
                f"No metadata for field '{field_name}' in collection "
                f"'{collection_name}'."
            )
        return per_field

    # ── Writes ────────────────────────────────────────────────────────────

    def insert(
        self,
        data: List[Dict[str, Any]],
        collection_name: str,
        partition_name: Optional[str] = None,
        **_: Any,
    ) -> Dict[str, int]:
        from qdrant_client.models import PointStruct, SparseVector

        points: List[PointStruct] = []
        for row in data:
            str_id = str(row["id"])
            sparse_dict = _coerce_sparse(row["sparse_embedding"])
            sparse_vec = SparseVector(
                indices=list(sparse_dict.keys()),
                values=list(sparse_dict.values()),
            )

            payload: Dict[str, Any] = {_STR_ID_KEY: str_id, _META_KEY: False}
            for k, v in row.items():
                if k in ("id", "dense_embedding", "sparse_embedding"):
                    continue
                payload[k] = v
            if partition_name is not None:
                payload[_PARTITION_KEY] = partition_name

            points.append(
                PointStruct(
                    id=_str_to_uuid(str_id),
                    vector={
                        "dense": list(row["dense_embedding"]),
                        "sparse": sparse_vec,
                    },
                    payload=payload,
                )
            )

        if points:
            self._qdrant.upsert(collection_name=collection_name, points=points)
        return {"insert_count": len(points)}

    def delete(
        self,
        collection_name: str,
        filter: Optional[str] = None,
        **_: Any,
    ) -> Dict[str, int]:
        from qdrant_client.models import FilterSelector

        if filter is None:
            return {"delete_count": 0}
        qf = _milvus_filter_to_qdrant(filter)
        self._qdrant.delete(
            collection_name=collection_name,
            points_selector=FilterSelector(filter=qf),
        )
        return {"delete_count": 0}

    def flush(self, collection_name: str) -> None:
        # Qdrant local mode writes are synchronous — nothing buffered.
        pass

    # ── Reads ─────────────────────────────────────────────────────────────

    def query(
        self,
        collection_name: str,
        filter: str,
        output_fields: Optional[List[str]] = None,
        limit: int = 16000,
        **_: Any,
    ) -> List[Dict[str, Any]]:
        qf = _milvus_filter_to_qdrant(filter)
        results, _next = self._qdrant.scroll(
            collection_name=collection_name,
            scroll_filter=qf,
            limit=limit,
            with_payload=True,
        )
        wanted = output_fields or []
        rows: List[Dict[str, Any]] = []
        for point in results:
            payload = dict(point.payload or {})
            str_id = payload.pop(_STR_ID_KEY, str(point.id))
            payload.pop(_META_KEY, None)
            payload.pop(_PARTITION_KEY, None)
            if wanted:
                row: Dict[str, Any] = {}
                for f in wanted:
                    row["id"] = str_id if f == "id" else row.get("id")
                    if f != "id":
                        row[f] = payload.get(f)
                rows.append(row)
            else:
                payload["id"] = str_id
                rows.append(payload)
        return rows

    def hybrid_search(
        self,
        reqs: List[Any],
        ranker: Any = None,
        limit: int = 10,
        output_fields: Optional[List[str]] = None,
        collection_name: str = "",
        partition_names: Optional[List[str]] = None,
        **_: Any,
    ) -> List[List[Dict[str, Any]]]:
        """Run a hybrid dense+sparse search and return results in Milvus shape.

        The mmore Retriever passes two ``AnnSearchRequest`` objects in fixed
        order ``[dense, sparse]``. We honour that convention but also accept
        the reverse order via the ``anns_field`` attribute.
        """
        from qdrant_client.models import (
            Fusion,
            FusionQuery,
            Prefetch,
            SparseVector,
        )

        dense_req, sparse_req = self._classify_reqs(reqs)
        dense_query = self._extract_data(dense_req)
        sparse_query = _coerce_sparse(self._extract_data(sparse_req))
        expr = getattr(dense_req, "expr", None) or getattr(sparse_req, "expr", None)
        qf = _milvus_filter_to_qdrant(expr, partition_names)

        leg_limit = max(int(limit), 1) * 2
        prefetch = [
            Prefetch(
                query=list(dense_query), using="dense", limit=leg_limit, filter=qf
            ),
            Prefetch(
                query=SparseVector(
                    indices=list(sparse_query.keys()),
                    values=list(sparse_query.values()),
                ),
                using="sparse",
                limit=leg_limit,
                filter=qf,
            ),
        ]

        results = self._qdrant.query_points(
            collection_name=collection_name,
            prefetch=prefetch,
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
            with_payload=True,
        )

        wanted = output_fields or []
        hits: List[Dict[str, Any]] = []
        for r in results.points:
            payload = dict(r.payload or {})
            str_id = payload.pop(_STR_ID_KEY, str(r.id))
            payload.pop(_META_KEY, None)
            payload.pop(_PARTITION_KEY, None)
            entity = (
                {f: payload.get(f) for f in wanted if f != "id"} if wanted else payload
            )
            hits.append({"id": str_id, "distance": float(r.score), "entity": entity})
        return [hits]

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _classify_reqs(reqs: List[Any]) -> Tuple[Any, Any]:
        if len(reqs) != 2:
            raise ValueError(
                f"QdrantMilvusClient.hybrid_search expects exactly 2 "
                f"AnnSearchRequests (one dense, one sparse); got {len(reqs)}."
            )
        a, b = reqs
        if getattr(a, "anns_field", "dense_embedding") == "dense_embedding":
            return a, b
        return b, a

    @staticmethod
    def _extract_data(req: Any) -> Any:
        data = getattr(req, "data", None)
        if data is None:
            data = getattr(req, "_data", None)
        if data is None:
            raise ValueError("AnnSearchRequest has no `data` attribute")
        if isinstance(data, (list, tuple)) and len(data) >= 1:
            return data[0]
        return data

    def _upsert_meta(
        self,
        collection_name: str,
        index_params: _IndexParamsCapture,
        dense_dim: int,
    ) -> None:
        from qdrant_client.models import PointStruct, SparseVector

        meta_id = _meta_point_uuid(collection_name)
        payload = {
            _META_KEY: True,
            _INDEX_PARAMS_KEY: index_params.fields,
        }
        # The meta point still needs both vectors to satisfy the schema.
        self._qdrant.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=meta_id,
                    vector={
                        "dense": [0.0] * dense_dim,
                        "sparse": SparseVector(indices=[0], values=[0.0]),
                    },
                    payload=payload,
                )
            ],
        )

    def close(self) -> None:
        """Release the file lock held by Qdrant local mode."""
        self._qdrant.close()
