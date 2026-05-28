"""Unit tests for judge.py (config: examples/rag/config_judge.yaml)."""

from dataclasses import replace
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from mmore.rag.judge import (
    JudgeDecision,
    JudgeResult,
    LLMJudge,
    _parse_json_response,
    compute_retrieval_metrics,
    merge_documents,
    metrics_meet_thresholds,
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


def test_parse_json_repairs_trailing_comma_and_python_literals():
    p = _parse_json_response(
        '{"decision":"RE_RETRIEVE","context_relevance_score":4,}'
    )
    assert p["decision"] == "RE_RETRIEVE" and p["context_relevance_score"] == 4
    p = _parse_json_response('{"decision":"PROCEED","sufficient":True}')
    assert p["sufficient"] is True


def test_metrics_merge_and_thresholds():
    m = compute_retrieval_metrics([_doc(0.9, "1"), _doc(0.5, "2")])
    assert m["mean_similarity"] == pytest.approx(0.7)
    th = {"min_mean_similarity": 0.35, "min_num_docs": 2}
    assert metrics_meet_thresholds(m, th)
    merged = merge_documents([_doc(0.8, "1")], [_doc(0.8, "1"), _doc(0.6, "2")])
    assert len(merged) == 2 and merged[1].metadata["rank"] == 2


@pytest.mark.parametrize(
    "cfg_kw,docs,llm_json,invoke,decision,reason_sub",
    [
        ({"skip_llm_judge": True}, [_doc(0.2)], None, False, "PROCEED", "skip_llm"),
        (
            {"metric_thresholds": {"min_mean_similarity": 0.3, "min_num_docs": 1}},
            [_doc(0.9)],
            None,
            False,
            "PROCEED",
            "metrics_above",
        ),
        ({}, [_doc(0.2)], '{"decision":"RE_RETRIEVE","reason":"weak"}', True, "RE_RETRIEVE", "weak"),
        (
            {"metric_thresholds": _THRESH},
            [_doc(0.2)],
            '{"decision":"PROCEED","reason":"ok","context_relevance_score":8}',
            True,
            "RE_RETRIEVE",
            "thresholds_not_met",
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
        JudgeResult(decision=JudgeDecision.RE_RETRIEVE, retrieve_params={"k": 10}),
        JudgeResult(decision=JudgeDecision.PROCEED),
    ]
    out = retrieve_with_judge(retriever, judge, {"input": "q?"})
    assert out["judge_actions"] == ["RE_RETRIEVE"]
    assert out["retrieval_corrections"][0]["action"] == "RE_RETRIEVE"
    assert judge.evaluate.call_count == 2


def test_retrieve_stops_on_metrics_without_second_llm_call():
    retriever, judge = MagicMock(), MagicMock()
    retriever.invoke.side_effect = [
        [_doc(0.2)],
        [_doc(0.9, "2", rerank=2.0), _doc(0.85, "3", rerank=2.0), _doc(0.88, "4", rerank=2.0)],
    ]
    judge.config = _cfg(
        max_corrective_steps=2,
        metric_thresholds={k: v for k, v in _THRESH.items() if k != "min_context_relevance"},
    )
    judge.evaluate.return_value = JudgeResult(
        decision=JudgeDecision.RE_RETRIEVE, retrieve_params={"k": 10}
    )
    out = retrieve_with_judge(retriever, judge, {"input": "q?"})
    assert out["judge_decision"] == "PROCEED"
    assert out["retrieval_metrics"]["thresholds_met"] == 1.0
    judge.evaluate.assert_called_once()
