"""Tests for judge.py — aligned with examples/rag/config_judge.yaml."""

from dataclasses import replace
from unittest.mock import MagicMock

import pytest
from dacite.exceptions import MissingValueError
from langchain_core.documents import Document

from mmore.rag.judge import (
    JudgeDecision,
    JudgeResult,
    LLMJudge,
    _metrics_for_output,
    _record_correction_metrics,
    compute_retrieval_metrics,
    format_metrics_status,
    merge_documents,
    metrics_meet_thresholds,
    retrieve_with_judge,
)
from mmore.rag.pipeline import RAGConfig
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


def _strong(n=3):
    return [_doc(0.9, str(i), 2.0) for i in range(1, n + 1)]


def test_config_judge_yaml():
    assert _CFG.skip_llm_judge is False and _CFG.max_corrective_steps == 1
    assert _CFG.metric_thresholds == _THRESH
    assert "{metrics_status}" in _CFG.user_prompt
    assert "failed_thresholds" not in _CFG.user_prompt
    with pytest.raises(MissingValueError):
        load_config(
            {
                "retriever": {"db": {"uri": "x.db", "name": "db"}, "k": 3},
                "llm": {"llm_name": "gpt2"},
                "judge": {
                    "llm": {"llm_name": "gpt-4o-mini"},
                    "max_corrective_steps": 1,
                },
            },
            RAGConfig,
        )


def test_metrics_and_thresholds():
    m = compute_retrieval_metrics([_doc(0.9, "1"), _doc(0.5, "2")])
    assert m["mean_similarity"] == 0.7 and m["has_rerank_scores"] == 0.0
    th = {"min_mean_similarity": 0.35, "min_num_docs": 2}
    assert metrics_meet_thresholds({"mean_similarity": 0.5, "num_docs": 3}, th)
    assert "FAIL" in format_metrics_status({"mean_similarity": 0.2, "num_docs": 1}, th)
    merged = merge_documents([_doc(0.8, "1")], [_doc(0.8, "1"), _doc(0.6, "2")])
    assert len(merged) == 2 and [d.metadata["rank"] for d in merged] == [1, 2]

    th_full = {k: v for k, v in _THRESH.items() if k != "min_context_relevance"}
    th_full["min_context_relevance"] = 7.0
    out_low = _metrics_for_output(_strong(), th_full, context_relevance_score=5.0)
    assert (
        out_low["context_relevance_score"] == 5.0 and out_low["thresholds_met"] == 0.0
    )
    out_high = _metrics_for_output(_strong(), th_full, context_relevance_score=8.0)
    assert out_high["thresholds_met"] == 1.0

    rec = _record_correction_metrics(
        "RE_RETRIEVE",
        [_doc(0.2)],
        _strong(2),
        {"min_mean_similarity": 0.3, "min_num_docs": 2},
        context_relevance_score=5.0,
    )
    assert rec["delta"]["delta_mean_similarity"] > 0
    assert rec["after"]["num_docs"] == 2.0


@pytest.mark.parametrize(
    "cfg_kw,docs,llm_json,expect_invoke,expect_decision,expect_reason",
    [
        (
            {"skip_llm_judge": True},
            [_doc(0.2)],
            None,
            False,
            "PROCEED",
            "skip_llm_judge",
        ),
        (
            {"metric_thresholds": {"min_mean_similarity": 0.3, "min_num_docs": 1}},
            [_doc(0.9)],
            None,
            False,
            "PROCEED",
            "metrics_above_thresholds",
        ),
        (
            {},
            [_doc(0.2)],
            '{"decision":"RE_RETRIEVE","reason":"x"}',
            True,
            "RE_RETRIEVE",
            "x",
        ),
        (
            {"metric_thresholds": _THRESH},
            _strong(),
            '{"decision":"PROCEED","reason":"ok","context_relevance_score":8}',
            True,
            "PROCEED",
            "ok",
        ),
        (
            {"metric_thresholds": {"min_num_docs": 1}, "allow_re_retrieve": False},
            [],
            '{"decision":"RE_RETRIEVE"}',
            True,
            "PROCEED",
            None,
        ),
        (
            {"metric_thresholds": {"min_num_docs": 1}},
            [],
            "garbage",
            True,
            "PROCEED",
            "parse_error_fallback",
        ),
        (
            {"metric_thresholds": {"min_num_docs": 1}},
            [],
            '{"decision":"RE_RETRIEVE","reason":"x"}\n\nNote: reformulating query.',
            True,
            "RE_RETRIEVE",
            "x",
        ),
        (
            {"metric_thresholds": {"min_num_docs": 1}},
            [],
            '```json\n{"decision":"PROCEED","reason":"ok"}\n```\nDone.',
            True,
            "RE_RETRIEVE",
            "thresholds_not_met",
        ),
        (
            {"metric_thresholds": _THRESH},
            [_doc(0.2)],
            '{"decision":"PROCEED","reason":"ok","context_relevance_score":8}',
            True,
            "RE_RETRIEVE",
            "thresholds_not_met",
        ),
    ],
    ids=[
        "skip",
        "numeric_ok",
        "llm_weak",
        "llm_context_rel",
        "coerce",
        "parse_err",
        "extra_data",
        "markdown_fence",
        "proceed_overridden",
    ],
)
def test_evaluate(
    cfg_kw, docs, llm_json, expect_invoke, expect_decision, expect_reason
):
    llm = MagicMock()
    if llm_json:
        llm.invoke.return_value = MagicMock(content=llm_json)
    r = LLMJudge(llm, _cfg(**cfg_kw)).evaluate("q?", docs)
    assert llm.invoke.called == expect_invoke
    assert r.decision == JudgeDecision(expect_decision)
    if expect_reason:
        assert expect_reason in r.reason
    if llm_json and "context_relevance_score" in llm_json:
        assert r.context_relevance_score == 8.0


@pytest.mark.parametrize(
    "loop",
    ["proceed", "re_retrieve", "metrics_after_correction", "max_steps"],
)
def test_retrieve_with_judge(loop):
    retriever, judge = MagicMock(), MagicMock()
    retriever.k = 5
    state = {"input": "q?", "collection_name": "my_docs"}

    if loop == "proceed":
        retriever.invoke.return_value = [_doc(0.9)]
        judge.config = _cfg()
        judge.evaluate.return_value = JudgeResult(
            decision=JudgeDecision.PROCEED,
            context_relevance_score=8.0,
        )
        want_actions, want_evals = [], 1
    elif loop == "re_retrieve":
        retriever.invoke.side_effect = [[_doc(0.2)], [_doc(0.8, "2", rerank=2.0)]]
        judge.config = _cfg(max_corrective_steps=2)
        judge.evaluate.side_effect = [
            JudgeResult(
                decision=JudgeDecision.RE_RETRIEVE,
                retrieve_params={"k": 10},
                context_relevance_score=4.0,
            ),
            JudgeResult(decision=JudgeDecision.PROCEED, context_relevance_score=8.0),
        ]
        want_actions, want_evals = ["RE_RETRIEVE"], 2
    elif loop == "metrics_after_correction":
        retriever.invoke.side_effect = [[_doc(0.2)], _strong()]
        judge.config = _cfg(
            max_corrective_steps=2,
            metric_thresholds={
                k: v for k, v in _THRESH.items() if k != "min_context_relevance"
            },
        )
        judge.evaluate.return_value = JudgeResult(
            decision=JudgeDecision.RE_RETRIEVE, retrieve_params={"k": 10}
        )
        want_actions, want_evals = ["RE_RETRIEVE"], 1
    else:
        retriever.invoke.return_value = [_doc(0.2)]
        judge.config = _cfg(max_corrective_steps=0)
        judge.evaluate.return_value = JudgeResult(decision=JudgeDecision.RE_RETRIEVE)
        want_actions, want_evals = [], 1

    out = retrieve_with_judge(retriever, judge, state)

    assert out["judge_decision"] == "PROCEED"
    assert out["judge_actions"] == want_actions
    assert judge.evaluate.call_count == want_evals
    if loop == "metrics_after_correction":
        assert out["retrieval_metrics"]["thresholds_met"] == 1.0
    if loop == "proceed":
        assert out["retrieval_metrics"]["context_relevance_score"] == 8.0
    if loop == "re_retrieve":
        assert len(out["retrieval_corrections"]) == 1
        corr = out["retrieval_corrections"][0]
        assert corr["action"] == "RE_RETRIEVE"
        assert corr["delta"]["delta_max_rerank_score"] > 0
