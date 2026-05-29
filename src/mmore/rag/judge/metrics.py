"""Retrieval metrics, threshold checks, and document merging."""

import logging
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document

from .types import CorrectionRecord, RetrievalMetrics

logger = logging.getLogger(__name__)


def compute_retrieval_metrics(docs: List[Document]) -> RetrievalMetrics:
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

    if similarities:
        max_similarity = max(similarities)
        mean_similarity = sum(similarities) / len(similarities)
    else:
        max_similarity = 0.0
        mean_similarity = 0.0

    if rerank_scores:
        max_rerank_score = max(rerank_scores)
        mean_rerank_score = sum(rerank_scores) / len(rerank_scores)
    else:
        max_rerank_score = 0.0
        mean_rerank_score = 0.0

    return RetrievalMetrics(
        num_docs=float(len(docs)),
        mean_similarity=mean_similarity,
        max_similarity=max_similarity,
        mean_rerank_score=mean_rerank_score,
        max_rerank_score=max_rerank_score,
    )


def _check_thresholds(
    metrics: RetrievalMetrics, thresholds: Dict[str, float]
) -> Tuple[bool, str]:
    if not thresholds:
        return False, "No thresholds configured."

    lines: List[str] = []
    all_pass = True
    for metric_key, value in metrics.threshold_items().items():
        threshold_key = f"min_{metric_key}"
        if threshold_key not in thresholds:
            continue
        bound = thresholds[threshold_key]
        passed = value >= bound
        if not passed:
            all_pass = False
        status = "PASS" if passed else "FAIL"
        lines.append(
            f"- {metric_key}: {value:.4f} (need {threshold_key}={bound}) -> {status}"
        )

    status_text = "\n".join(lines) if lines else "No applicable threshold keys."
    return all_pass, status_text


def evaluate_metrics(
    docs: List[Document],
    thresholds: Dict[str, float],
    context_relevance_score: Optional[float] = None,
) -> Tuple[RetrievalMetrics, bool, str]:
    """Compute metrics, check thresholds, and format status in one pass."""
    metrics = compute_retrieval_metrics(docs)
    if context_relevance_score is not None:
        metrics = replace(metrics, context_relevance_score=context_relevance_score)
    passed, status = _check_thresholds(metrics, thresholds)
    return metrics, passed, status


def metrics_for_output(
    docs: List[Document],
    thresholds: Dict[str, float],
    context_relevance_score: Optional[float] = None,
) -> RetrievalMetrics:
    metrics, passed, _ = evaluate_metrics(docs, thresholds, context_relevance_score)
    return replace(metrics, thresholds_met=float(passed))


def record_correction_metrics(
    action: str,
    docs_before: List[Document],
    docs_after: List[Document],
    thresholds: Dict[str, float],
    context_relevance_score: Optional[float],
) -> CorrectionRecord:
    """Before/after retrieval metrics for one corrective action (e.g. RE_RETRIEVE)."""
    before_full, tm_before, _ = evaluate_metrics(
        docs_before, thresholds, context_relevance_score
    )
    after_full, tm_after, _ = evaluate_metrics(
        docs_after, thresholds, context_relevance_score
    )
    return CorrectionRecord(
        action=action,
        before=before_full,
        after=after_full,
        thresholds_met_before=float(tm_before),
        thresholds_met_after=float(tm_after),
    )


def log_correction_metrics(query: str, record: CorrectionRecord) -> None:
    b, a, d = record.before, record.after, record.delta_dict()
    logger.info(
        "Judge corrective %s | query=%r | num_docs %.0f→%.0f (Δ%+.0f) | "
        "mean_sim %.4f→%.4f (Δ%+.4f) | max_rerank %.4f→%.4f (Δ%+.4f) | "
        "context_rel %s→%s | thresholds_met %.0f→%.0f",
        record.action,
        query[:120],
        b.num_docs,
        a.num_docs,
        d["delta_num_docs"],
        b.mean_similarity,
        a.mean_similarity,
        d["delta_mean_similarity"],
        b.max_rerank_score,
        a.max_rerank_score,
        d["delta_max_rerank_score"],
        "—",
        "—",
        record.thresholds_met_before,
        record.thresholds_met_after,
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
