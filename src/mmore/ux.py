"""Shared UX helpers for all mmore pipelines: standard logging setup, colored
output, a spinner for indeterminate waits, and a single-line progress bar."""

import itertools
import logging
import random
import sys
import threading
import time
import warnings
from typing import TYPE_CHECKING, Iterable, Optional

if TYPE_CHECKING:
    from tqdm import tqdm

DATEFMT = "%Y-%m-%d %H:%M:%S"


# ------------------------------ Logging setup ------------------------------ #


def setup_logging(name: str, emoji: str, level: int = logging.INFO) -> logging.Logger:
    """Configure logging with the standard mmore header `[name emoji time] msg`
    and return a logger for the step. Call once from a pipeline entry point."""
    logging.basicConfig(
        format=f"[{name} {emoji} %(asctime)s] %(message)s",
        datefmt=DATEFMT,
        level=level,
        force=True,
    )
    return logging.getLogger(name)


# ----------------------------- Colored output ------------------------------ #

_COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "gray": "\033[90m",
}


def str_in_color(to_print: "str | int", color: str, bold: bool = False) -> str:
    style = _COLORS.get(color, _COLORS["reset"])
    if bold:
        style = _COLORS["bold"] + style
    return f"{style}{to_print}{_COLORS['reset']}"


def print_in_color(to_print: "str | int", color: str, bold: bool = False) -> None:
    print(str_in_color(to_print, color, bold))


def str_green(text, bold=False):
    return str_in_color(text, "green", bold=bold)


# ------------------------------ Noise control ------------------------------ #


def quiet_noisy_libs():
    """Hide INFO logs, warnings and progress bars so an interactive CLI stays
    clean. Opt-in: batch pipelines that want their own logs should not call it."""
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

    def __init__(self, label: Optional[str] = None):
        self._label = label
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
        word = self._label or random.choice(SPINNER_WORDS)
        start = word_start = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            if self._label is None and now - word_start > 3:
                word = random.choice(SPINNER_WORDS)
                word_start = now
            status = f"{next(frames)} {word}... ({int(now - start)}s)"
            sys.stdout.write(f"\r\033[K{str_in_color(status, 'blue')}")
            sys.stdout.flush()
            time.sleep(0.1)


# ------------------------------ Progress bar ------------------------------- #


def progress(
    iterable: Optional[Iterable] = None,
    *,
    total: Optional[int] = None,
    desc: str = "",
    unit: str = "doc",
    **kwargs,
) -> "tqdm":
    """Standard single-line progress bar for per-item pipeline loops. Shows the
    step, count done/total and ETA. Call `bar.set_postfix_str(name)` inside the
    loop to display the current item.

        for doc in (bar := progress(docs, desc="Processing", unit="file")):
            bar.set_postfix_str(doc.name)
    """
    from tqdm import tqdm

    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        bar_format="{desc}: {n_fmt}/{total_fmt} {bar} {percentage:3.0f}%{postfix}",
        **kwargs,
    )
