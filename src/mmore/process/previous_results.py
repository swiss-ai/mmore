import json
import os
from datetime import datetime
from typing import Dict, List


def load_previous_results(path: str) -> Dict[str, List[dict]]:
    """Load a JSONL file and index samples by ``metadata.file_path``.

    Args:
        path: Absolute path to the JSONL file produced by a previous run.

    Returns:
        A dict mapping each ``file_path`` string to the list of sample dicts
        whose ``metadata.file_path`` equals that key.

    Raises:
        FileNotFoundError: If *path* does not exist on disk.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Previous results file not found: {path}")

    index: Dict[str, List[dict]] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            file_path = sample.get("metadata", {}).get("file_path", "__unknown__")
            index.setdefault(file_path, []).append(sample)
    return index


def is_reusable_process(
    file_path: str, previous_results: Dict[str, List[dict]]
) -> bool:
    """Decide whether the process-stage cache for *file_path* is still valid.

    A cached result is considered valid when the source file has not been
    modified since it was last processed, i.e.::

        current_mtime <= max(processed_at for each cached sample)

    Args:
        file_path: Path to the source file on disk.
        previous_results: Index returned by :func:`load_previous_results`.

    Returns:
        ``True`` if the cache is usable; ``False`` otherwise.
    """
    if file_path not in previous_results:
        return False

    samples = previous_results[file_path]
    if not samples:
        return False

    timestamps = [
        s["metadata"]["processed_at"]
        for s in samples
        if s.get("metadata", {}).get("processed_at")
    ]
    if not timestamps:
        return False

    max_processed_at = max(datetime.fromisoformat(t) for t in timestamps)
    mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
    return mtime <= max_processed_at


def is_reusable_postprocess(
    file_path: str,
    input_processed_at: str,
    previous_results: Dict[str, List[dict]],
) -> bool:
    """Decide whether the post-process-stage cache for *file_path* is still valid.

    A cached result is considered valid when the post-processor ran *after* the
    process stage produced its output, i.e.::

        input_processed_at <= max(processed_at for each cached sample)

    Args:
        file_path: The source-document path used as the grouping key.
        input_processed_at: ISO 8601 timestamp of when the process stage last
            produced output for this file.
        previous_results: Index returned by :func:`load_previous_results`.

    Returns:
        ``True`` if the cache is usable; ``False`` otherwise.
    """
    if file_path not in previous_results:
        return False

    samples = previous_results[file_path]
    if not samples:
        return False

    timestamps = [
        s["metadata"]["processed_at"]
        for s in samples
        if s.get("metadata", {}).get("processed_at")
    ]
    if not timestamps:
        return False

    max_cached_at = max(datetime.fromisoformat(t) for t in timestamps)
    input_dt = datetime.fromisoformat(input_processed_at)
    return input_dt <= max_cached_at


def merge_results(
    reused: Dict[str, List[dict]],
    new_results: List[dict],
    current_file_paths: set,
) -> List[dict]:
    """Combine reused samples and newly processed samples into a flat list.

    Entries whose ``metadata.file_path`` is **not** in *current_file_paths* are
    silently dropped so that output for deleted files does not accumulate.

    Args:
        reused: Mapping of ``file_path`` → list of cached sample dicts to keep.
        new_results: Flat list of freshly processed sample dicts.
        current_file_paths: Set of file paths that currently exist (i.e. were
            present in the crawl that triggered this run).

    Returns:
        A flat list of sample dicts containing valid cached entries followed by
        newly processed entries, with deleted-file entries removed.
    """
    merged: List[dict] = []

    for file_path, samples in reused.items():
        if file_path in current_file_paths:
            merged.extend(samples)

    for sample in new_results:
        fp = sample.get("metadata", {}).get("file_path", "__unknown__")
        if fp in current_file_paths:
            merged.append(sample)

    return merged
