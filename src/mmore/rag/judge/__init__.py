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
    merge_documents,
    record_correction_metrics,
)
from .parsing import parse_json_response
from .types import (
    JUDGE_OUTPUT_KEYS,
    CorrectionRecord,
    JudgeConfig,
    JudgeDecision,
    JudgeResult,
    RetrievalMetrics,
    extract_judge_output,
)

__all__ = [
    "JUDGE_OUTPUT_KEYS",
    "CorrectionRecord",
    "JudgeConfig",
    "JudgeDecision",
    "JudgeResult",
    "RetrievalMetrics",
    "LLMJudge",
    "compute_retrieval_metrics",
    "coerce_decision",
    "evaluate_metrics",
    "extract_judge_output",
    "merge_documents",
    "parse_json_response",
    "record_correction_metrics",
    "retrieve_with_judge",
    "step_record",
]
