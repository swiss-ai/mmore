"""Shared UX helpers for all mmore pipelines: standard logging setup, colored
output, a spinner, and a progress bar."""

import itertools
import logging
import os
import random
import sys
import threading
import time
import warnings
from typing import TYPE_CHECKING, Iterable, Optional

if TYPE_CHECKING:
    from tqdm import tqdm

DATEFMT = "%Y-%m-%d %H:%M:%S"

# Set MMORE_VERBOSE=1 to unquiet everything
VERBOSE_ENV = "MMORE_VERBOSE"

# Third-party loggers names
_NOISY_LIBS = [
    "surya",
    "marker",
    "transformers",
    "moviepy",
    "datasets",
    "sentence_transformers",
    "PIL",
    "httpx",
    "urllib3",
    "primp",
    "reqwest",
    "hyper_util",
    "h2",
    "hickory_net",
    "hickory_resolver",
    "rustls",
    "cookie_store",
]


def is_verbose() -> bool:
    """True when MMORE_VERBOSE is set."""
    return bool(os.environ.get(VERBOSE_ENV))


# ------------------------------ Logging setup ------------------------------ #


def setup_logging(name: str, emoji: str, level: int = logging.INFO) -> logging.Logger:
    """Configure logging with the standard mmore header `[name emoji time] msg`
    and return a logger for the step. Call once from a pipeline entry point."""
    logging.basicConfig(
        format=f"[{name} {emoji} %(asctime)s] %(message)s",
        datefmt=DATEFMT,
        level=logging.DEBUG if is_verbose() else level,
        force=True,
    )
    return logging.getLogger(name)


def quiet_noisy_libs(hide_info: bool = False) -> None:
    """Silence noisy third-party libraries and their progress bars so output
    stays readable. With hide_info=True, also hide mmore's own INFO logs."""
    if is_verbose():
        return
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    warnings.filterwarnings("ignore")
    for name in _NOISY_LIBS:
        logging.getLogger(name).setLevel(logging.ERROR)
    if hide_info:
        logging.disable(logging.INFO)
    try:
        from transformers.utils import logging as hf_logging
    except ImportError:
        return
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()


def init_worker(name: str = "Process", emoji: str = "🚀") -> None:
    """Reconfigure logging inside a worker process so its logging config
    is the same as the parent one."""
    setup_logging(name, emoji)
    quiet_noisy_libs()


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


def str_green(text: str | int, bold: bool = False) -> str:
    return str_in_color(text, "green", bold=bold)


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

    def __init__(self, label: Optional[str] = None) -> None:
        self._label = label
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "Spinner":
        self._stop.clear()
        if sys.stdout.isatty():
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()

    def _spin(self) -> None:
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
    """Standard progress bar for per-item pipeline loops. Shows the
    step, count done/total and ETA.

    Example:
        ```for doc in (bar := progress(docs, desc="Processing", unit="file")):
            bar.set_postfix_str(doc.name)
        ```
    """
    from tqdm import tqdm

    # disable=False keeps our bars visible even when TQDM_DISABLE silences
    # third-party library bars (see quiet_noisy_libs).
    kwargs.setdefault("disable", False)
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        bar_format="{desc}: {n_fmt}/{total_fmt} {bar} {percentage:3.0f}%{postfix}",
        **kwargs,
    )
