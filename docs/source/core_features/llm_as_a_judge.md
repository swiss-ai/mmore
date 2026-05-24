# LLM as a judge

## Overview

With a `judge:` block in your RAG config, MMORE checks whether retrieved chunks are good enough before generation. If not, it can search again, ask follow-up questions, or pull web context—then merge everything into one deduplicated list.

## TL;DR

**Once:** search the index (Milvus + optional BGE rerank) → initial chunk list.

**Then a loop** (at most `max_corrective_steps` corrective actions, default `1`):

1. Compute metrics on the current chunks.
2. Judge → `PROCEED` or a corrective action (how this is chosen is explained in Config file paragraph).
3. If `PROCEED` → leave the loop and generate the final answer.
4. Otherwise run one corrective action (`RE_RETRIEVE`, `ADD_QUESTIONS`, or `ADD_CONTEXT`), merge new chunks with the old ones (dedupe + re-rank).
5. If metrics pass after that merge → stop as `PROCEED` without calling the judge LLM again.
6. If metrics still fail and you have corrective steps left → go back to step 1 with the merged chunks.
7. If no steps left → stop anyway and generate with current chunks.

**After the loop:** the answer LLM runs on the final context. Output also includes `retrieval_metrics`, `judge_decision`, and `judge_actions`.

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

Keep your usual `retriever`, `llm`, `mode`, and `mode_args` sections. Add `judge` for this feature.

### How step 2 decides between `PROCEED` and Use the Judge

This is driven by the `rag.judge` settings you configure:

- `skip_llm_judge: true` → always `PROCEED` (no judge LLM; useful for threshold-only experiments).
- `metric_thresholds` — compare computed metrics to your mins (`min_mean_similarity`, `min_max_rerank_score`, `min_num_docs`, …). If every **numeric** min is met and you did **not** set `min_context_relevance` → `PROCEED` without calling the judge LLM.
- **Otherwise** → one call to `judge.llm` using your `system_prompt` / `user_prompt` (question, metrics, PASS/FAIL per threshold, chunk excerpts). The model must return JSON with a relevance score (1–10) and a `decision`:
  - `PROCEED` if it judges the context sufficient.
  - `RE_RETRIEVE`, `ADD_QUESTIONS`, or `ADD_CONTEXT` if not (only if the matching `allow_*` flag is `true`).

`min_context_relevance` needs the judge LLM (it sets `context_relevance_score` in the JSON). Omit it if you want step 2 to skip the LLM when numeric metrics alone are enough.

### Other useful `rag.judge` fields

- `max_corrective_steps` — how many corrective actions after the first retrieval (default `1`).
- `allow_re_retrieve` / `allow_add_questions` / `allow_add_context` — which decisions the LLM may return.
- For `ADD_CONTEXT`: `pip install "mmore[rag,websearch]"`.

## See also

- [RAG](../getting_started/rag.md)
- [Websearch](websearch.md)

