"""
Corrective RAG judge.
Evaluates retrieval quality after the retriever (and optional reranker) and may trigger corrective actions before generation.
"""

from .corrective import retrieve_with_judge
from .decisions import coerce_decision, step_record
from .evaluator import LLMJudge
from .metrics import (
    compute_retrieval_metrics,
    evaluate_metrics,
    format_metrics_status,
    merge_documents,
    metrics_meet_thresholds,
    record_correction_metrics,
)
from .parsing import parse_json_response
from .types import JudgeConfig, JudgeDecision, JudgeResult

__all__ = [
    "JudgeConfig",
    "JudgeDecision",
    "JudgeResult",
    "LLMJudge",
    "compute_retrieval_metrics",
    "coerce_decision",
    "evaluate_metrics",
    "format_metrics_status",
    "merge_documents",
    "metrics_meet_thresholds",
    "parse_json_response",
    "record_correction_metrics",
    "retrieve_with_judge",
    "step_record",
]
