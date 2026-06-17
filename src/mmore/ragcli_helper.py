"""Presentation helpers for the RAG CLI: colored output, spinner, noise control,
and timing/token metrics collection."""

import itertools
import logging
import random
import sys
import threading
import time
import warnings
from typing import Dict, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

# ----------------------------- Colored output ------------------------------ #


def str_in_color(to_print: str | int, color: str, bold: bool = False) -> str:
    colors = {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "gray": "\033[90m",
    }
    style = colors.get(color, colors["reset"])
    if bold:
        style = colors["bold"] + style
    return f"{style}{to_print}{colors['reset']}"


def print_in_color(to_print: str | int, color: str, bold: bool = False) -> None:
    print(str_in_color(to_print, color, bold))


def str_green(text, bold=False):
    return str_in_color(text, "green", bold=bold)


# ------------------------------ Noise control ------------------------------ #


def quiet_noisy_libs():
    """Hide INFO logs, warnings and progress bars so the CLI stays clean."""
    logging.disable(logging.INFO)
    warnings.filterwarnings("ignore")
    try:
        from transformers.utils import logging as hf_logging
    except ImportError:
        return
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()


# --------------------------------- Spinner --------------------------------- #

SPINNER_WORDS = [
    "Thinking",
    "Pondering",
    "Discombobulating",
    "Cooking",
    "Brewing",
    "Ruminating",
    "Rummaging",
    "Noodling",
]


class Spinner:
    """Animated status line shown while work happens in the calling thread."""

    def __init__(self):
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self):
        self._stop.clear()
        if sys.stdout.isatty():
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()

    def _spin(self):
        frames = itertools.cycle("|/-\\")
        word = random.choice(SPINNER_WORDS)
        start = word_start = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            if now - word_start > 3:
                word = random.choice(SPINNER_WORDS)
                word_start = now
            status = f"{next(frames)} {word}... ({int(now - start)}s)"
            sys.stdout.write(f"\r\033[K{str_in_color(status, 'blue')}")
            sys.stdout.flush()
            time.sleep(0.1)


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
