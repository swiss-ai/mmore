# LLM as a judge

Add a `judge:` block to your RAG config to check retrieval quality before generation. When chunks are not good enough, the judge can trigger a corrective action such as re-search, sub-questions, or web context and then merge everything into one deduplicated list.

## How it works

1. Retrieve chunks from the index (Milvus + optional BGE rerank).
2. Evaluate them in a loop (at most `max_corrective_steps` corrective actions, default `1`):
  - Proceed without calling the judge LLM if index metrics meet `metric_thresholds`.
  - Otherwise, call `judge.llm` (or apply `force_corrective_action`) and run the chosen corrective action.
  - Repeat on the merged chunks until the judge says `PROCEED` or the step budget is exhausted.
3. Generate the answer from the final context.

Disallowed decisions are coerced to a fallback action (`RE_RETRIEVE`, `ADD_QUESTIONS`, or `PROCEED`). Invalid JSON defaults to `PROCEED`.

## Decisions


| Decision        | What it does                                                    |
| --------------- | --------------------------------------------------------------- |
| `PROCEED`       | Chunks are good enough; continue to the answer LLM              |
| `RE_RETRIEVE`   | Search the index again (reformulated query and/or more results) |
| `ADD_QUESTIONS` | Up to 3 extra searches from sub-questions, then merge           |
| `ADD_CONTEXT`   | DuckDuckGo web snippets, then merge                             |


## Configuration

`examples/rag/config_judge.yaml` is a standalone config — it does not load on top of `config.yaml`.

```bash
python3 -m mmore rag --config-file examples/rag/config_judge.yaml
```

Or copy the `judge:` block into your own config.

Key settings under `rag.judge`:

- `metric_thresholds` — index minimums (`min_mean_similarity`, `min_max_rerank_score`, `min_num_docs`, …)
- `max_corrective_steps` — how many corrective actions after the first retrieval
- `allow_re_retrieve` / `allow_add_questions` / `allow_add_context` — allowed LLM decisions
- `force_corrective_action` — skip the judge LLM and force an action when thresholds fail (benchmarks); use `PROCEED` for a no-correction baseline
- `system_prompt` / `user_prompt` — judge prompts; user prompt supports `{query}`, `{metrics}`, `{chunks}`, `{allowed_actions}`, and correction-step placeholders

For `ADD_CONTEXT`, install web search support:

```bash
pip install "mmore[rag,websearch]"
```

## See also

- [RAG](../getting_started/rag.md)
- [Websearch](websearch.md)

