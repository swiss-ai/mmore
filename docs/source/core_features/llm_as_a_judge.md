# LLM as a judge

## Overview

With a `judge:` block in your RAG config, MMORE checks whether retrieved chunks are good enough before generation. If not, it can search again, ask follow-up questions, or pull web context and then merge everything into one deduplicated list.

## TL;DR

**Once:** search the index (Milvus + optional BGE rerank) → initial chunk list.

**Then a loop** (at most `max_corrective_steps` corrective actions, default `1`). Each iteration runs **step 1** on the current chunks:

1. `skip_llm_judge: true` → `PROCEED`, leave the loop.
2. All **index** metrics ≥ `metric_thresholds` → `PROCEED`, leave the loop (no judge LLM call).
3. Otherwise → call `judge.llm` → `decision` (+ `context_relevance_score` 1–10 for logs).
4. Disallowed corrective `decision` (`allow_*`) → fallback: any non-`RE_RETRIEVE` choice becomes `RE_RETRIEVE` if allowed; a disallowed `RE_RETRIEVE` becomes `ADD_QUESTIONS` if allowed; otherwise `PROCEED`. Invalid JSON / unknown decision → `PROCEED`.
5. LLM says `PROCEED` → `PROCEED`, leave the loop.
6. LLM says `RE_RETRIEVE`, `ADD_QUESTIONS`, or `ADD_CONTEXT` → run the action, merge chunks (dedupe + optional re-rank).
7. After a corrective action → go back to **step 1** on the merged chunks (metrics may now pass without an LLM call; final `reason` is then `metrics_after_correction`).
8. No corrective steps left → `PROCEED` anyway and generate with current chunks.

**After the loop:** the answer LLM runs on the final context. Output also includes `retrieval_metrics`, `judge_decision`, `judge_actions`, and `retrieval_corrections` (per-action `before` / `after` / `delta` and `thresholds_met_before` → `thresholds_met_after` to measure whether a correction helped).

## What the judge can decide


| Decision        | Meaning                                                  |
| --------------- | -------------------------------------------------------- |
| `PROCEED`       | Chunks are good enough; continue to the answer LLM       |
| `RE_RETRIEVE`   | Search the index again (new wording and/or more results) |
| `ADD_QUESTIONS` | Up to 3 extra searches from sub-questions, then merge    |
| `ADD_CONTEXT`   | DuckDuckGo web snippets, then merge                      |


## Config file

`config_judge.yaml` does not load on top of `config.yaml`. Either:

```bash
python3 -m mmore rag --config-file examples/rag/config_judge.yaml
```

or copy the `judge:` block into your config and pass your file path.

### `rag.judge` settings

- `skip_llm_judge: true` → step 1 only; no judge LLM and no corrective actions.
- `metric_thresholds` — index mins (`min_mean_similarity`, `min_max_rerank_score`, `min_num_docs`, …). Used in step 2 as an entry filter; not applied again when the LLM chooses `PROCEED`.
- `system_prompt` / `user_prompt` — used in step 3 when step 2 fails. The user prompt may use `{correction_step}`, `{corrective_actions_used}`, `{max_corrective_steps}`, `{remaining_corrective_steps}`, `{after_correction}`, plus `{query}`, `{metrics}`, `{metrics_status}`, `{allowed_actions}`, `{chunks}`.
- `max_corrective_steps` — how many corrective actions (step 6) after the first retrieval.
- `allow_re_retrieve` / `allow_add_questions` / `allow_add_context` — which decisions the LLM may return (step 4).
- For `ADD_CONTEXT`: `pip install "mmore[rag,websearch]"`.

## See also

- [RAG](../getting_started/rag.md)
- [Websearch](websearch.md)
