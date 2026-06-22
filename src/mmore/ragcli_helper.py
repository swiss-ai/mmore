"""Presentation helpers for the RAG CLI: timing/token metrics collection."""

import time
from typing import Dict, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

# ----------------------------- Timing metrics ------------------------------ #


class TimingHandler(BaseCallbackHandler):
    """Collects retrieval/generation wall times and token usage from callbacks."""

    def __init__(self):
        self.retrieval_time: Optional[float] = None
        self.generation_time: Optional[float] = None
        self.completion_tokens: Optional[int] = None
        self._starts: Dict[UUID, float] = {}

    def on_retriever_start(self, serialized, query, *, run_id, **kwargs):
        self._starts[run_id] = time.perf_counter()

    def on_retriever_end(self, documents, *, run_id, **kwargs):
        if run_id in self._starts:
            self.retrieval_time = time.perf_counter() - self._starts.pop(run_id)

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        self._starts[run_id] = time.perf_counter()

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self._starts[run_id] = time.perf_counter()

    def on_llm_end(self, response, *, run_id, **kwargs):
        if run_id in self._starts:
            self.generation_time = time.perf_counter() - self._starts.pop(run_id)
        self.completion_tokens = _output_tokens(response)


def _output_tokens(response) -> Optional[int]:
    """Generated-token count if the provider reported it (specific to API models)."""
    try:
        usage = response.generations[0][0].message.usage_metadata
        if usage and usage.get("output_tokens"):
            return usage["output_tokens"]
    except (AttributeError, IndexError, TypeError):
        pass
    usage = (response.llm_output or {}).get("token_usage", {})
    return usage.get("completion_tokens") or usage.get("output_tokens")
