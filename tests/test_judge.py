"""Unit tests for mmore.rag.judge package (config: examples/rag/config_judge.yaml)."""

from dataclasses import replace
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from mmore.rag.judge import (
    JUDGE_OUTPUT_KEYS,
    JudgeDecision,
    JudgeResult,
    LLMJudge,
    coerce_decision,
    compute_retrieval_metrics,
    evaluate_metrics,
    extract_judge_output,
    merge_documents,
    metrics_meet_thresholds,
    parse_json_response,
    record_correction_metrics,
    retrieve_with_judge,
)
from mmore.run_rag import RAGInferenceConfig
from mmore.utils import load_config

_CFG = load_config("examples/rag/config_judge.yaml", RAGInferenceConfig).rag.judge
_THRESH = _CFG.metric_thresholds


def _cfg(**kw):
    return replace(_CFG, **kw)


def _doc(sim=0.5, id_="1", rerank=None):
    meta = {"rank": 1, "similarity": sim, "id": id_}
    if rerank is not None:
        meta["rerank_score"] = rerank
    return Document(page_content=id_, metadata=meta)


def test_judge_output_keys_match_retrieve_with_judge():
    retriever, judge = MagicMock(), MagicMock()
    retriever.invoke.return_value = [_doc(0.9)]
    judge.config = _cfg()
    judge.evaluate.return_value = JudgeResult(decision=JudgeDecision.PROCEED)
    out = retrieve_with_judge(retriever, judge, {"input": "q?"})
    for key in JUDGE_OUTPUT_KEYS:
        if key == "retrieval_corrections":
            continue  # only set when a corrective action ran
        assert key in out

    public = extract_judge_output(out)
    assert public["judge_decision"] == "PROCEED"
    assert "retrieval_corrections" not in public


def test_judge_result_proceed():
    r = JudgeResult.proceed("metrics_above_thresholds")
    assert r.decision == JudgeDecision.PROCEED
    assert r.reason == "metrics_above_thresholds"
    assert r.exit_reason == "metrics_above_thresholds"
    assert r.llm_invoked is False

    r2 = JudgeResult.proceed(
        "parse_error_fallback",
        llm_invoked=True,
        context_relevance_score=5.0,
    )
    assert r2.llm_invoked is True
    assert r2.context_relevance_score == 5.0


def test_evaluate_metrics_unifies_compute_threshold_and_status():
    docs = [_doc(0.9, "1"), _doc(0.5, "2")]
    thresholds = {"min_mean_similarity": 0.35, "min_num_docs": 2}
    metrics, passed, status = evaluate_metrics(docs, thresholds)
    assert metrics["mean_similarity"] == pytest.approx(0.7)
    assert passed is True
    assert "mean_similarity" in status
    assert "PASS" in status

    metrics2, passed2, _ = evaluate_metrics(
        docs, thresholds, context_relevance_score=3.0
    )
    assert metrics2["context_relevance_score"] == 3.0
    assert passed2 is True


def test_threshold_key_derived_from_min_prefix():
    docs = [_doc(0.9, "1")]
    thresholds = {"min_max_rerank_score": 0.5}
    assert metrics_meet_thresholds(compute_retrieval_metrics(docs), thresholds) is False
    docs_reranked = [_doc(0.9, "1", rerank=0.8)]
    assert metrics_meet_thresholds(compute_retrieval_metrics(docs_reranked), thresholds)


def test_correction_delta_omits_context_relevance_score():
    before = [_doc(0.2, "1")]
    after = [_doc(0.8, "2", rerank=2.0)]
    record = record_correction_metrics(
        "RE_RETRIEVE", before, after, _THRESH, context_relevance_score=7.0
    )
    assert "context_relevance_score" not in record["delta"]
    assert "context_relevance_score" not in record["before"]
    assert "context_relevance_score" not in record["after"]
    assert record["delta"]["delta_mean_similarity"] == pytest.approx(0.6)


@pytest.mark.parametrize(
    "cfg_kw,raw,expected",
    [
        (
            {
                "allow_re_retrieve": True,
                "allow_add_questions": False,
                "allow_add_context": False,
            },
            "ADD_QUESTIONS",
            JudgeDecision.RE_RETRIEVE,
        ),
        (
            {
                "allow_re_retrieve": False,
                "allow_add_questions": True,
                "allow_add_context": False,
            },
            "RE_RETRIEVE",
            JudgeDecision.ADD_QUESTIONS,
        ),
        (
            {
                "allow_re_retrieve": True,
                "allow_add_questions": False,
                "allow_add_context": False,
            },
            "ADD_CONTEXT",
            JudgeDecision.RE_RETRIEVE,
        ),
        (
            {
                "allow_re_retrieve": False,
                "allow_add_questions": False,
                "allow_add_context": True,
            },
            "ADD_QUESTIONS",
            JudgeDecision.PROCEED,
        ),
        (
            {
                "allow_re_retrieve": False,
                "allow_add_questions": False,
                "allow_add_context": False,
            },
            "RE_RETRIEVE",
            JudgeDecision.PROCEED,
        ),
        (
            {
                "allow_re_retrieve": True,
                "allow_add_questions": True,
                "allow_add_context": False,
            },
            "ADD_QUESTIONS",
            JudgeDecision.ADD_QUESTIONS,
        ),
    ],
)
def test_coerce_decision_fallback(cfg_kw, raw, expected):
    decision, coerced, raw_decision = coerce_decision(raw, _cfg(**cfg_kw))
    assert decision == expected
    assert raw_decision == raw


def test_parse_json_repairs_trailing_comma_and_python_literals():
    p = parse_json_response('{"decision":"RE_RETRIEVE","context_relevance_score":4,}')
    assert p["decision"] == "RE_RETRIEVE" and p["context_relevance_score"] == 4
    p = parse_json_response('{"decision":"PROCEED","sufficient":True}')
    assert p["sufficient"] is True


def test_metrics_merge_and_thresholds():
    m = compute_retrieval_metrics([_doc(0.9, "1"), _doc(0.5, "2")])
    assert m["mean_similarity"] == pytest.approx(0.7)
    th = {"min_mean_similarity": 0.35, "min_num_docs": 2}
    assert metrics_meet_thresholds(m, th)
    m["context_relevance_score"] = 3.0
    assert metrics_meet_thresholds(m, th)
    merged = merge_documents([_doc(0.8, "1")], [_doc(0.8, "1"), _doc(0.6, "2")])
    assert len(merged) == 2 and merged[1].metadata["rank"] == 2


@pytest.mark.parametrize(
    "cfg_kw,docs,llm_json,invoke,decision,reason_sub",
    [
        (
            {"force_corrective_action": "PROCEED", "metric_thresholds": _THRESH},
            [_doc(0.2)],
            None,
            False,
            "PROCEED",
            "force_corrective",
        ),
        (
            {"metric_thresholds": {"min_mean_similarity": 0.3, "min_num_docs": 1}},
            [_doc(0.9)],
            None,
            False,
            "PROCEED",
            "metrics_above",
        ),
        (
            {},
            [_doc(0.2)],
            '{"decision":"RE_RETRIEVE","reason":"weak"}',
            True,
            "RE_RETRIEVE",
            "weak",
        ),
        (
            {"metric_thresholds": _THRESH},
            [_doc(0.2)],
            '{"decision":"PROCEED","reason":"ok","context_relevance_score":8}',
            True,
            "PROCEED",
            "ok",
        ),
        (
            {"metric_thresholds": {"min_num_docs": 1}},
            [],
            "not json",
            True,
            "PROCEED",
            "parse_error",
        ),
    ],
)
def test_llm_judge_evaluate(cfg_kw, docs, llm_json, invoke, decision, reason_sub):
    llm = MagicMock()
    if llm_json:
        llm.invoke.return_value = MagicMock(content=llm_json)
    r = LLMJudge(llm, _cfg(**cfg_kw)).evaluate("q?", docs)
    assert llm.invoke.called == invoke
    assert r.decision == JudgeDecision(decision)
    assert reason_sub in r.reason


def test_llm_judge_early_exit_on_thresholds():
    llm = MagicMock()
    judge = LLMJudge(
        llm, _cfg(metric_thresholds={"min_mean_similarity": 0.3, "min_num_docs": 1})
    )
    result = judge._early_exit([_doc(0.9)], after_correction=False)
    assert result is not None
    assert result.decision == JudgeDecision.PROCEED
    assert result.reason == "metrics_above_thresholds"
    llm.invoke.assert_not_called()


def test_llm_judge_result_from_parsed():
    judge = LLMJudge(MagicMock(), _cfg())
    parsed = {
        "decision": "RE_RETRIEVE",
        "reason": "weak retrieval",
        "context_relevance_score": 4,
        "retrieve_params": {"k": 10},
    }
    result = judge._result_from_parsed(parsed)
    assert result.decision == JudgeDecision.RE_RETRIEVE
    assert result.reason == "weak retrieval"
    assert result.exit_reason == "llm_corrective"
    assert result.context_relevance_score == 4.0
    assert result.retrieve_params == {"k": 10}


def test_retrieve_with_judge_proceed():
    retriever, judge = MagicMock(), MagicMock()
    retriever.invoke.return_value = [_doc(0.9)]
    judge.config = _cfg()
    judge.evaluate.return_value = JudgeResult(decision=JudgeDecision.PROCEED)
    out = retrieve_with_judge(retriever, judge, {"input": "q?"})
    assert out["judge_decision"] == "PROCEED" and out["judge_actions"] == []
    judge.evaluate.assert_called_once()


def test_retrieve_with_judge_re_retrieve_and_correction_log():
    retriever, judge = MagicMock(), MagicMock()
    retriever.k = 5
    retriever.invoke.side_effect = [[_doc(0.2)], [_doc(0.8, "2", rerank=2.0)]]
    judge.config = _cfg(max_corrective_steps=2)
    judge.evaluate.side_effect = [
        JudgeResult(
            decision=JudgeDecision.RE_RETRIEVE,
            exit_reason="llm_corrective",
            llm_invoked=True,
            retrieve_params={"k": 10},
        ),
        JudgeResult(
            decision=JudgeDecision.PROCEED,
            exit_reason="metrics_above_thresholds",
        ),
    ]
    out = retrieve_with_judge(retriever, judge, {"input": "q?"})
    assert out["judge_actions"] == ["RE_RETRIEVE"]
    assert out["retrieval_corrections"][0]["action"] == "RE_RETRIEVE"
    assert out["judge_llm_calls"] == 1
    assert judge.evaluate.call_count == 2


def test_retrieve_with_judge_exports_trace_and_max_steps():
    retriever, judge = MagicMock(), MagicMock()
    retriever.k = 5
    retriever.invoke.side_effect = [[_doc(0.2)], [_doc(0.8, "2", rerank=2.0)]]
    judge.config = _cfg(max_corrective_steps=1)
    judge.evaluate.side_effect = [
        JudgeResult(
            decision=JudgeDecision.RE_RETRIEVE,
            exit_reason="llm_corrective",
            llm_invoked=True,
            retrieve_params={"k": 10},
        ),
        JudgeResult(
            decision=JudgeDecision.RE_RETRIEVE,
            exit_reason="llm_corrective",
            llm_invoked=True,
        ),
    ]
    out = retrieve_with_judge(retriever, judge, {"input": "q?"})
    assert out["judge_reason"] == "max_corrective_steps"
    assert out["hit_max_corrective_steps"] == 1.0
    assert out["judge_llm_calls"] == 2
    assert out["judge_actions"] == ["RE_RETRIEVE"]


def test_retrieve_stops_on_metrics_without_second_llm_call():
    retriever, judge = MagicMock(), MagicMock()
    retriever.invoke.side_effect = [
        [_doc(0.2)],
        [
            _doc(0.9, "2", rerank=2.0),
            _doc(0.85, "3", rerank=2.0),
            _doc(0.88, "4", rerank=2.0),
        ],
    ]
    judge.config = _cfg(max_corrective_steps=2, metric_thresholds=_THRESH)
    judge.evaluate.side_effect = [
        JudgeResult(decision=JudgeDecision.RE_RETRIEVE, retrieve_params={"k": 10}),
        JudgeResult(decision=JudgeDecision.PROCEED, reason="metrics_above_thresholds"),
    ]
    out = retrieve_with_judge(retriever, judge, {"input": "q?"})
    assert out["judge_decision"] == "PROCEED"
    assert out["retrieval_metrics"]["thresholds_met"] == 1.0
    assert judge.evaluate.call_count == 2
