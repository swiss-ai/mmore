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
    context_relevance_score: Optional[float] = None
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
    rerank_after_merge: bool = False
    max_web_results: int = 5
    max_chunk_chars: int = 400


_THRESHOLD_CHECKS: Dict[str, str] = {
    "min_mean_similarity": "mean_similarity",
    "min_max_similarity": "max_similarity",
    "min_num_docs": "num_docs",
    "min_max_rerank_score": "max_rerank_score",
    "min_mean_rerank_score": "mean_rerank_score",
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


def _parse_context_relevance_score(parsed: Dict[str, Any]) -> Optional[float]:
    raw = parsed.get("context_relevance_score")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid context_relevance_score %r", raw)
        return None


def _metrics_with_context_relevance(
    docs: List[Document], context_relevance_score: Optional[float]
) -> Dict[str, float]:
    metrics = compute_retrieval_metrics(docs)
    if context_relevance_score is not None:
        metrics["context_relevance_score"] = context_relevance_score
    return metrics


def _metrics_for_output(
    docs: List[Document],
    thresholds: Dict[str, float],
    context_relevance_score: Optional[float] = None,
) -> Dict[str, float]:
    metrics = _metrics_with_context_relevance(docs, context_relevance_score)
    metrics["thresholds_met"] = float(metrics_meet_thresholds(metrics, thresholds))
    return metrics


_CORRECTION_COMPARE_KEYS = (
    "num_docs",
    "mean_similarity",
    "max_similarity",
    "mean_rerank_score",
    "max_rerank_score",
    "context_relevance_score",
)


def _metrics_snapshot(metrics: Dict[str, float]) -> Dict[str, float]:
    return {k: float(metrics[k]) for k in _CORRECTION_COMPARE_KEYS if k in metrics}


def _record_correction_metrics(
    action: str,
    docs_before: List[Document],
    docs_after: List[Document],
    thresholds: Dict[str, float],
    context_relevance_score: Optional[float],
) -> Dict[str, Any]:
    """Before/after retrieval metrics for one corrective action (e.g. RE_RETRIEVE)."""
    before_full = _metrics_with_context_relevance(docs_before, context_relevance_score)
    after_full = _metrics_with_context_relevance(docs_after, context_relevance_score)
    before = _metrics_snapshot(before_full)
    after = _metrics_snapshot(after_full)
    delta = {f"delta_{k}": after.get(k, 0.0) - before.get(k, 0.0) for k in after}
    tm_before = metrics_meet_thresholds(before_full, thresholds)
    tm_after = metrics_meet_thresholds(after_full, thresholds)
    return {
        "action": action,
        "before": before,
        "after": after,
        "delta": delta,
        "thresholds_met_before": float(tm_before),
        "thresholds_met_after": float(tm_after),
    }


def _log_correction_metrics(query: str, record: Dict[str, Any]) -> None:
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


def _repair_json_text(text: str) -> str:
    """Fix common invalid JSON from instruct models (trailing commas, Python literals)."""
    text = re.sub(r",\s*([}\]])", r"\1", text)
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    text = re.sub(r"\bNone\b", "null", text)
    return text


def _extract_judge_fields_loose(text: str) -> Dict[str, Any]:
    """Best-effort field extraction when strict JSON parsing fails."""
    obj: Dict[str, Any] = {}
    if m := re.search(r'"decision"\s*:\s*"([A-Z_]+)"', text, re.IGNORECASE):
        obj["decision"] = m.group(1).upper()
    if m := re.search(r'"context_relevance_score"\s*:\s*([\d.]+)', text):
        obj["context_relevance_score"] = float(m.group(1))
    if m := re.search(r'"reason"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL):
        obj["reason"] = m.group(1)
    if m := re.search(r'"sufficient"\s*:\s*(true|false)', text, re.IGNORECASE):
        obj["sufficient"] = m.group(1).lower() == "true"
    if "decision" not in obj:
        raise json.JSONDecodeError("No decision field found", text, 0)
    return obj


def _parse_json_response(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```.*$", "", text, flags=re.DOTALL).strip()

    start = text.find("{")
    if start == -1:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise json.JSONDecodeError("No JSON object found", text, 0)
        start = match.start()

    snippet = text[start:]
    last_error: Optional[Exception] = None
    for candidate in (snippet, _repair_json_text(snippet)):
        try:
            obj, _ = json.JSONDecoder().raw_decode(candidate)
            if not isinstance(obj, dict):
                raise TypeError(f"Expected JSON object, got {type(obj).__name__}")
            return obj
        except (json.JSONDecodeError, TypeError) as e:
            last_error = e
            continue

    try:
        return _extract_judge_fields_loose(snippet)
    except json.JSONDecodeError:
        if last_error is not None:
            raise last_error
        raise json.JSONDecodeError("Could not parse judge JSON", snippet, 0)


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
    if decision.value in allowed:
        return decision

    if decision != JudgeDecision.PROCEED:
        if decision == JudgeDecision.RE_RETRIEVE and config.allow_add_questions:
            logger.warning(
                "Judge chose disallowed action %s, falling back to ADD_QUESTIONS",
                decision.value,
            )
            return JudgeDecision.ADD_QUESTIONS
        if decision != JudgeDecision.RE_RETRIEVE and config.allow_re_retrieve:
            logger.warning(
                "Judge chose disallowed action %s, falling back to RE_RETRIEVE",
                decision.value,
            )
            return JudgeDecision.RE_RETRIEVE

    logger.warning(
        "Judge chose disallowed action %s, defaulting to PROCEED", decision.value
    )
    return JudgeDecision.PROCEED


class LLMJudge:
    """LLM-as-a-Judge: structured JSON decision on retrieval quality."""

    def __init__(self, llm: BaseChatModel, config: JudgeConfig):
        self.llm = llm
        self.config = config

    def evaluate(
        self,
        query: str,
        docs: List[Document],
        *,
        after_correction: bool = False,
        corrective_actions_used: int = 0,
    ) -> JudgeResult:
        thresholds = self.config.metric_thresholds
        metrics = compute_retrieval_metrics(docs)

        if self.config.skip_llm_judge:
            return JudgeResult(
                decision=JudgeDecision.PROCEED,
                reason="skip_llm_judge",
            )

        if metrics_meet_thresholds(metrics, thresholds):
            reason = (
                "metrics_after_correction"
                if after_correction
                else "metrics_above_thresholds"
            )
            return JudgeResult(
                decision=JudgeDecision.PROCEED,
                reason=reason,
            )

        max_steps = self.config.max_corrective_steps
        user_prompt = self.config.user_prompt.format(
            query=query,
            metrics=metrics,
            metrics_status=format_metrics_status(metrics, thresholds),
            thresholds=thresholds,
            allowed_actions=_allowed_actions(self.config),
            chunks=_format_chunks_for_prompt(docs, self.config.max_chunk_chars),
            correction_step=corrective_actions_used + 1,
            corrective_actions_used=corrective_actions_used,
            max_corrective_steps=max_steps,
            remaining_corrective_steps=max(0, max_steps - corrective_actions_used),
            after_correction=after_correction,
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
            raw = _extract_llm_text(response.content)
            logger.warning(
                "Failed to parse judge JSON: %s | raw=%r",
                e,
                raw[:500],
            )
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

        context_relevance_score = _parse_context_relevance_score(parsed)
        reason = str(parsed.get("reason", ""))
        retrieve_params = parsed.get("retrieve_params")

        return JudgeResult(
            decision=decision,
            reason=reason,
            context_relevance_score=context_relevance_score,
            extra_questions=[str(q) for q in extra_questions],
            web_query=parsed.get("web_query"),
            retrieve_params=retrieve_params,
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

    if config.rerank_after_merge and docs:
        # Re-rank the final merged list with the retriever's existing reranker.
        if getattr(retriever, "reranker_model", None):
            docs = retriever.rerank(query, docs)
        else:
            logger.info(
                "Judge rerank_after_merge requested but retriever has no reranker_model"
            )

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
    retrieval_corrections: List[Dict[str, Any]] = []
    config = judge.config
    query = state["input"]
    thresholds = config.metric_thresholds
    final_result = JudgeResult(decision=JudgeDecision.PROCEED, reason="")

    for step in range(config.max_corrective_steps + 1):
        final_result = judge.evaluate(
            query,
            docs,
            after_correction=bool(judge_actions),
            corrective_actions_used=len(judge_actions),
        )
        logger.info(
            "Judge step %s | query=%r | decision=%s | context_relevance_score=%s | reason=%s",
            step,
            query[:120],
            final_result.decision.value,
            final_result.context_relevance_score,
            final_result.reason,
        )

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
                context_relevance_score=final_result.context_relevance_score,
            )
            break

        action = final_result.decision.value
        judge_actions.append(action)
        docs_before = docs
        docs = _apply_corrective_action(
            retriever, config, state, docs_before, final_result
        )
        correction = _record_correction_metrics(
            action,
            docs_before,
            docs,
            thresholds,
            final_result.context_relevance_score,
        )
        retrieval_corrections.append(correction)
        _log_correction_metrics(query, correction)

    retrieval_metrics = _metrics_for_output(
        docs, thresholds, final_result.context_relevance_score
    )
    logger.info(
        "Judge done | query=%r | decision=%s | actions=%s | thresholds_met=%s | "
        "context_relevance_score=%s | reason=%s",
        query[:120],
        final_result.decision.value,
        judge_actions,
        retrieval_metrics.get("thresholds_met"),
        retrieval_metrics.get("context_relevance_score"),
        final_result.reason,
    )

    out: Dict[str, Any] = {
        **state,
        "docs": docs,
        "retrieval_metrics": retrieval_metrics,
        "judge_decision": final_result.decision.value,
        "judge_actions": judge_actions,
    }
    if retrieval_corrections:
        out["retrieval_corrections"] = retrieval_corrections
    return out
