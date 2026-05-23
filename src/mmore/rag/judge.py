"""
Corrective RAG judge.
Evaluates retrieval quality after the retriever (and optional reranker) and may trigger corrective actions before generation.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .llm import LLMConfig
from .retriever import Retriever

logger = logging.getLogger(__name__)


class JudgeDecision(str, Enum):
    PROCEED = "PROCEED"
    ADD_QUESTIONS = "ADD_QUESTIONS"
    ADD_CONTEXT = "ADD_CONTEXT"
    RE_RETRIEVE = "RE_RETRIEVE"


@dataclass
class JudgeResult:
    decision: JudgeDecision
    reason: str = ""
    extra_questions: List[str] = field(default_factory=list)
    web_query: Optional[str] = None
    retrieve_params: Optional[Dict[str, Any]] = None


@dataclass
class JudgeConfig:
    """LLM judge config. Only LLMJudge is implemented."""

    llm: LLMConfig
    system_prompt: str
    user_prompt: str
    metric_thresholds: Dict[str, float] = field(default_factory=dict)
    skip_llm_judge: bool = False
    max_corrective_steps: int = 1
    allow_add_questions: bool = True
    allow_add_context: bool = True
    allow_re_retrieve: bool = True
    max_web_results: int = 5
    max_chunk_chars: int = 400


_THRESHOLD_CHECKS: Dict[str, str] = {
    "min_mean_similarity": "mean_similarity",
    "min_max_similarity": "max_similarity",
    "min_num_docs": "num_docs",
    "min_max_rerank_score": "max_rerank_score",
    "min_mean_rerank_score": "mean_rerank_score",
    "min_context_relevance": "context_relevance_score",
}


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
        metrics["has_rerank_scores"] = 1.0
    else:
        metrics["max_rerank_score"] = 0.0
        metrics["mean_rerank_score"] = 0.0
        metrics["has_rerank_scores"] = 0.0

    return metrics


def _check_thresholds(
    metrics: Dict[str, float], thresholds: Dict[str, float]
) -> Tuple[bool, str]:
    if not thresholds:
        return False, "No thresholds configured."

    lines: List[str] = []
    all_pass = True
    for key, metric_key in _THRESHOLD_CHECKS.items():
        if key not in thresholds:
            continue
        value = metrics.get(metric_key, 0.0)
        bound = thresholds[key]
        passed = value >= bound
        if not passed:
            all_pass = False
        status = "PASS" if passed else "FAIL"
        lines.append(f"- {metric_key}: {value:.4f} (need {key}={bound}) -> {status}")

    status_text = "\n".join(lines) if lines else "No applicable threshold keys."
    return all_pass, status_text


def metrics_meet_thresholds(
    metrics: Dict[str, float], thresholds: Dict[str, float]
) -> bool:
    return _check_thresholds(metrics, thresholds)[0]


def format_metrics_status(
    metrics: Dict[str, float], thresholds: Dict[str, float]
) -> str:
    return _check_thresholds(metrics, thresholds)[1]


def _metrics_for_output(
    docs: List[Document], thresholds: Dict[str, float]
) -> Dict[str, float]:
    metrics = compute_retrieval_metrics(docs)
    metrics["thresholds_met"] = float(metrics_meet_thresholds(metrics, thresholds))
    return metrics


def merge_documents(
    existing: List[Document], new_docs: List[Document]
) -> List[Document]:
    """Merge document lists, dedupe by id or content, re-assign ranks."""
    seen: set[str] = set()
    merged: List[Document] = []

    for doc in existing + new_docs:
        key = doc.metadata.get("id") or doc.page_content
        if key in seen:
            continue
        seen.add(key)
        merged.append(doc)

    for i, doc in enumerate(merged):
        doc.metadata["rank"] = i + 1
    return merged


def _parse_json_response(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


def _extract_llm_text(content: Any) -> str:
    if isinstance(content, list):
        content = content[-1] if content else ""
    if isinstance(content, dict):
        content = content.get("content", "")
    return str(content)


def _format_chunks_for_prompt(docs: List[Document], max_chunk_chars: int) -> str:
    lines = []
    for doc in docs:
        content = doc.page_content
        if len(content) > max_chunk_chars:
            content = content[:max_chunk_chars] + "..."
        sim = doc.metadata.get("similarity", "n/a")
        rerank = doc.metadata.get("rerank_score", "n/a")
        rank = doc.metadata.get("rank", "?")
        lines.append(
            f"[rank={rank}, similarity={sim}, rerank_score={rerank}] {content}"
        )
    return "\n".join(lines) if lines else "(no chunks retrieved)"


def _allowed_actions(config: JudgeConfig) -> List[str]:
    allowed = [JudgeDecision.PROCEED.value]
    if config.allow_add_questions:
        allowed.append(JudgeDecision.ADD_QUESTIONS.value)
    if config.allow_add_context:
        allowed.append(JudgeDecision.ADD_CONTEXT.value)
    if config.allow_re_retrieve:
        allowed.append(JudgeDecision.RE_RETRIEVE.value)
    return allowed


def _coerce_decision(raw: str, config: JudgeConfig) -> JudgeDecision:
    try:
        decision = JudgeDecision(raw)
    except ValueError:
        logger.warning("Unknown judge decision '%s', defaulting to PROCEED", raw)
        return JudgeDecision.PROCEED

    allowed = _allowed_actions(config)
    if decision.value not in allowed:
        logger.warning(
            "Judge chose disallowed action %s, defaulting to PROCEED", decision.value
        )
        return JudgeDecision.PROCEED
    return decision


class LLMJudge:
    """LLM-as-a-Judge: structured JSON decision on retrieval quality."""

    def __init__(self, llm: BaseChatModel, config: JudgeConfig):
        self.llm = llm
        self.config = config

    def evaluate(self, query: str, docs: List[Document]) -> JudgeResult:
        thresholds = self.config.metric_thresholds
        metrics = compute_retrieval_metrics(docs)

        if self.config.skip_llm_judge:
            return JudgeResult(
                decision=JudgeDecision.PROCEED,
                reason="skip_llm_judge",
            )

        if "min_context_relevance" not in thresholds and metrics_meet_thresholds(
            metrics, thresholds
        ):
            return JudgeResult(
                decision=JudgeDecision.PROCEED,
                reason="metrics_above_thresholds",
            )

        user_prompt = self.config.user_prompt.format(
            query=query,
            metrics=metrics,
            metrics_status=format_metrics_status(metrics, thresholds),
            thresholds=thresholds,
            allowed_actions=_allowed_actions(self.config),
            chunks=_format_chunks_for_prompt(docs, self.config.max_chunk_chars),
        )
        response = self.llm.invoke(
            [
                SystemMessage(content=self.config.system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )

        try:
            parsed = _parse_json_response(_extract_llm_text(response.content))
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("Failed to parse judge JSON: %s", e)
            return JudgeResult(
                decision=JudgeDecision.PROCEED,
                reason="parse_error_fallback",
            )

        decision = _coerce_decision(
            str(parsed.get("decision", JudgeDecision.PROCEED.value)), self.config
        )
        extra_questions = parsed.get("extra_questions") or []
        if not isinstance(extra_questions, list):
            extra_questions = []

        return JudgeResult(
            decision=decision,
            reason=str(parsed.get("reason", "")),
            extra_questions=[str(q) for q in extra_questions],
            web_query=parsed.get("web_query"),
            retrieve_params=parsed.get("retrieve_params"),
        )


def _apply_corrective_action(
    retriever: Retriever,
    config: JudgeConfig,
    state: Dict[str, Any],
    docs: List[Document],
    result: JudgeResult,
) -> List[Document]:
    query = state["input"]

    if result.decision == JudgeDecision.ADD_QUESTIONS:
        for sub_query in result.extra_questions[:3]:
            sub_state = {**state, "input": sub_query}
            sub_docs = retriever.invoke(sub_state)
            docs = merge_documents(docs, sub_docs)

    elif result.decision == JudgeDecision.ADD_CONTEXT:
        web_query = result.web_query or query
        web_docs = retriever._get_web_documents(
            web_query, max_results=config.max_web_results
        )
        docs = merge_documents(docs, web_docs)

    elif result.decision == JudgeDecision.RE_RETRIEVE:
        params = result.retrieve_params or {}
        new_state = {**state}
        if params.get("input"):
            new_state["input"] = params["input"]
        retrieve_kwargs: Dict[str, Any] = {}
        if params.get("k") is not None:
            retrieve_kwargs["k"] = int(params["k"])
        else:
            retrieve_kwargs["k"] = max(retriever.k * 2, retriever.k + 3)
        new_docs = retriever.invoke(new_state, **retrieve_kwargs)
        docs = merge_documents(docs, new_docs)

    return docs


def retrieve_with_judge(
    retriever: Retriever,
    judge: LLMJudge,
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run retrieval with optional corrective loop (retrieve -> judge -> action).

    Returns state enriched with docs, retrieval_metrics, judge_decision, judge_actions.
    """
    docs = retriever.invoke(state)
    judge_actions: List[str] = []
    config = judge.config
    query = state["input"]
    thresholds = config.metric_thresholds
    final_result = JudgeResult(decision=JudgeDecision.PROCEED, reason="")

    for step in range(config.max_corrective_steps + 1):
        final_result = judge.evaluate(query, docs)

        if final_result.decision == JudgeDecision.PROCEED:
            break

        if step >= config.max_corrective_steps:
            logger.info(
                "Max corrective steps (%s) reached; proceeding with current docs",
                config.max_corrective_steps,
            )
            final_result = JudgeResult(
                decision=JudgeDecision.PROCEED,
                reason="max_corrective_steps",
            )
            break

        judge_actions.append(final_result.decision.value)
        docs = _apply_corrective_action(retriever, config, state, docs, final_result)
        if metrics_meet_thresholds(compute_retrieval_metrics(docs), thresholds):
            final_result = JudgeResult(
                decision=JudgeDecision.PROCEED,
                reason="metrics_after_correction",
            )
            break

    return {
        **state,
        "docs": docs,
        "retrieval_metrics": _metrics_for_output(docs, thresholds),
        "judge_decision": final_result.decision.value,
        "judge_actions": judge_actions,
    }
