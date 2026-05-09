"""Locate bundled example configs regardless of CWD or install layout.

Strategy:
- If `examples/` exists relative to CWD (source checkout), use it.
- Else, walk up from CWD looking for a repo root that contains `examples/`.
- Else, fall back to `importlib.resources` to read examples shipped with the
  package (only available if the wheel actually bundles them).
- If nothing is found, return the original repo-relative path so error
  messages stay readable; callers handle "missing" gracefully.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional


@lru_cache(maxsize=1)
def repo_root() -> Optional[Path]:
    """Return a directory that contains an `examples/` folder, if any."""
    cwd = Path.cwd()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "examples").is_dir():
            return candidate
    return None


def resolve_example(rel: str) -> str:
    """Resolve an `examples/...` relative path to an absolute one.

    Falls back to the original string if no source checkout is found, so the
    UI can still display it (and the validator will surface a clear error).
    """
    root = repo_root()
    if root is not None:
        candidate = root / rel
        if candidate.exists():
            return str(candidate)
    return rel


def resolve_glob(pattern: str) -> tuple[Path, str]:
    """Split a relative glob into (root, remaining-pattern) for Path.glob."""
    root = repo_root() or Path.cwd()
    return root, pattern


def cwd_default(rel: str) -> str:
    """A safe default path rooted at CWD (e.g. `./data` instead of `examples/...`)."""
    return os.path.join(".", rel)
