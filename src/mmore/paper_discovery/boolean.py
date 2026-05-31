"""Stage 1: build boolean queries from a synonym table + category map.

Offline, deterministic - no netowrk calls.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Union

from .schema import CategoryQuery, SynonymEntry

logger = logging.getLogger(__name__)


def load_synonyms(path: Union[str, Path]) -> List[SynonymEntry]:
    """Load synonyms from a JSON file.

    Expected format:
        [{"word": "Foundation model", "synonyms": ["LLM", "GPT", ...]}]
    """

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = []
    for row in data:
        word = row.get("word") or row.get("WORD")
        raw = row.get("synonyms") or row.get("SYNONYMS AND NEAR SYNONYMS") or ""
        if isinstance(raw, str):
            synonyms = [s.strip() for s in re.split(r"[,;]", raw) if s.strip()]
        else:
            synonyms = [s.strip() for s in raw if s and s.strip()]
        if word:
            entries.append(SynonymEntry(word=word.strip(), synonyms=synonyms))
    return entries


def _or_group(entry: SynonymEntry) -> str:
    """Turn a SynonymEntry into an OR-group of quoted strings."""
    terms = {entry.word, *entry.synonyms}
    quoted = sorted(f'"{t}"' for t in terms if t)
    return "(" + " OR ".join(quoted) + ")"


def build_boolean_queries(
    synonyms: List[SynonymEntry],
    categories: Dict[str, List[str]],
) -> List[CategoryQuery]:
    """Compose category-level AND-of-OR-groups boolean queries.
    Missing keys are logged and skipped - never fail the whole run."""

    by_word = {e.word: e for e in synonyms}
    queries: List[CategoryQuery] = []

    for cat_name, words in categories.items():
        groups: List[str] = []
        for w in words:
            entry = by_word.get(w)
            if entry is None:
                logger.warning(
                    "Category %r references unknown word %r - skipping",
                    cat_name,
                    w,
                )
                continue
            groups.append(_or_group(entry))

        if not groups:
            continue

        queries.append(
            CategoryQuery(
                combination_title=cat_name,
                boolean_combination=" AND ".join(groups),
            )
        )

    return queries
