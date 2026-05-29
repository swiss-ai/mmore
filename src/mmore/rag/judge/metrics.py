"""Retrieval metrics, threshold checks, and document merging."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

_CORRECTION_COMPARE_KEYS = (
    "num_docs",
    "mean_similarity",
    "max_similarity",
    "mean_rerank_score",
    "max_rerank_score",
)


def compute_retrieval_metrics(docs: List[Document]) -> Dict[str, float]:
    """Metrics from Milvus similarity, BGE rerank scores, and chunk count."""
    similarities = [
        float(doc.metadata["similarity"])
        for doc in docs
        if doc.metadata.get("similarity") is not None
    ]
    rerank_scores = [
        float(doc.metadata["rerank_score"])
        for doc in docs
        if doc.metadata.get("rerank_score") is not None
    ]

    metrics: Dict[str, float] = {"num_docs": float(len(docs))}

    if similarities:
        metrics["max_similarity"] = max(similarities)
        metrics["mean_similarity"] = sum(similarities) / len(similarities)
    else:
        metrics["max_similarity"] = 0.0
        metrics["mean_similarity"] = 0.0

    if rerank_scores:
        metrics["max_rerank_score"] = max(rerank_scores)
        metrics["mean_rerank_score"] = sum(rerank_scores) / len(rerank_scores)
    else:
        metrics["max_rerank_score"] = 0.0
        metrics["mean_rerank_score"] = 0.0

    return metrics


def _check_thresholds(
    metrics: Dict[str, float], thresholds: Dict[str, float]
) -> Tuple[bool, str]:
    if not thresholds:
        return False, "No thresholds configured."

    lines: List[str] = []
    all_pass = True
    for key, bound in thresholds.items():
        if not key.startswith("min_"):
            continue
        metric_key = key[4:]
        value = metrics.get(metric_key, 0.0)
        passed = value >= bound
        if not passed:
            all_pass = False
        status = "PASS" if passed else "FAIL"
        lines.append(f"- {metric_key}: {value:.4f} (need {key}={bound}) -> {status}")

    status_text = "\n".join(lines) if lines else "No applicable threshold keys."
    return all_pass, status_text


def evaluate_metrics(
    docs: List[Document],
    thresholds: Dict[str, float],
    context_relevance_score: Optional[float] = None,
) -> Tuple[Dict[str, float], bool, str]:
    """Compute metrics, check thresholds, and format status in one pass."""
    metrics = compute_retrieval_metrics(docs)
    if context_relevance_score is not None:
        metrics["context_relevance_score"] = context_relevance_score
    passed, status = _check_thresholds(metrics, thresholds)
    return metrics, passed, status


def metrics_for_output(
    docs: List[Document],
    thresholds: Dict[str, float],
    context_relevance_score: Optional[float] = None,
) -> Dict[str, float]:
    metrics, passed, _ = evaluate_metrics(docs, thresholds, context_relevance_score)
    metrics["thresholds_met"] = float(passed)
    return metrics


def _metrics_snapshot(metrics: Dict[str, float]) -> Dict[str, float]:
    return {k: float(metrics[k]) for k in _CORRECTION_COMPARE_KEYS if k in metrics}


def record_correction_metrics(
    action: str,
    docs_before: List[Document],
    docs_after: List[Document],
    thresholds: Dict[str, float],
    context_relevance_score: Optional[float],
) -> Dict[str, Any]:
    """Before/after retrieval metrics for one corrective action (e.g. RE_RETRIEVE)."""
    before_full, tm_before, _ = evaluate_metrics(
        docs_before, thresholds, context_relevance_score
    )
    after_full, tm_after, _ = evaluate_metrics(
        docs_after, thresholds, context_relevance_score
    )
    before = _metrics_snapshot(before_full)
    after = _metrics_snapshot(after_full)
    delta = {f"delta_{k}": after.get(k, 0.0) - before.get(k, 0.0) for k in after}
    return {
        "action": action,
        "before": before,
        "after": after,
        "delta": delta,
        "thresholds_met_before": float(tm_before),
        "thresholds_met_after": float(tm_after),
    }


def log_correction_metrics(query: str, record: Dict[str, Any]) -> None:
    b, a, d = record["before"], record["after"], record["delta"]
    logger.info(
        "Judge corrective %s | query=%r | num_docs %.0f→%.0f (Δ%+.0f) | "
        "mean_sim %.4f→%.4f (Δ%+.4f) | max_rerank %.4f→%.4f (Δ%+.4f) | "
        "context_rel %s→%s | thresholds_met %.0f→%.0f",
        record["action"],
        query[:120],
        b.get("num_docs", 0),
        a.get("num_docs", 0),
        d.get("delta_num_docs", 0),
        b.get("mean_similarity", 0),
        a.get("mean_similarity", 0),
        d.get("delta_mean_similarity", 0),
        b.get("max_rerank_score", 0),
        a.get("max_rerank_score", 0),
        d.get("delta_max_rerank_score", 0),
        b.get("context_relevance_score", "—"),
        a.get("context_relevance_score", "—"),
        record["thresholds_met_before"],
        record["thresholds_met_after"],
    )


def _dedupe_key(doc: Document) -> str:
    doc_id = doc.metadata.get("id")
    if doc_id is not None:
        return str(doc_id)
    return doc.page_content


def merge_documents(
    existing: List[Document], new_docs: List[Document]
) -> List[Document]:
    """Merge document lists, dedupe by id or content, re-assign ranks."""
    seen: set[str] = set()
    merged: List[Document] = []

    for doc in existing + new_docs:
        key = _dedupe_key(doc)
        if key in seen:
            continue
        seen.add(key)
        merged.append(doc)

    for i, doc in enumerate(merged):
        doc.metadata["rank"] = i + 1
    return merged
