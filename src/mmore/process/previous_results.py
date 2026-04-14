import json
import os
from datetime import datetime
from typing import Dict, List

from ..type import MultimodalSample


def load_previous_results(path: str) -> Dict[str, List[MultimodalSample]]:
    """Load a JSONL file and index samples by ``metadata.file_path``.

    Args:
        path: Absolute path to the JSONL file produced by a previous run.

    Returns:
        A dict mapping each ``file_path`` string to the list of samples
        whose ``metadata.file_path`` equals that key.

    Raises:
        FileNotFoundError: If path does not exist on disk.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Previous results file not found: {path}")

    index: Dict[str, List[MultimodalSample]] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = MultimodalSample.from_dict(json.loads(line))
            index.setdefault(sample.metadata["file_path"], []).append(sample)
    return index


def is_reusable_process(
    file_path: str, previous_results: Dict[str, List[MultimodalSample]]
) -> bool:
    """Decide whether the process-stage cache for ``file_path`` is still valid.

    A cached result is considered valid when the source file has not been
    modified since it was last processed, i.e.::

        file_mtime <= max(processed_at for each cached sample)

    Args:
        file_path: Path to the source file on disk.
        previous_results: Index returned by :func:`load_previous_results`.

    Returns:
        ``True`` if the cache is usable, ``False`` otherwise.
    """
    if file_path not in previous_results:
        return False

    samples = previous_results[file_path]
    if not samples:
        return False

    timestamps = [
        s.metadata["processed_at"] for s in samples if s.metadata.get("processed_at")
    ]
    if not timestamps:
        return False

    max_processed_at = max(datetime.fromisoformat(t) for t in timestamps)
    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
    return file_mtime <= max_processed_at


def is_reusable_postprocess(
    file_path: str,
    input_processed_at: str,
    previous_results: Dict[str, List[MultimodalSample]],
) -> bool:
    """Decide whether the post-process-stage cache for ``file_path`` is still valid.

    A cached result is considered valid when the post-processor ran after the
    process stage produced its output, i.e.::

        input_processed_at <= max(processed_at for each cached sample)

    Args:
        file_path: The source-document path used as the grouping key.
        input_processed_at: ISO timestamp of when the process stage last produced an
            output for this file.
        previous_results: Index returned by :func:`load_previous_results`.

    Returns:
        ``True`` if the cache is usable, ``False`` otherwise.
    """
    if file_path not in previous_results:
        return False

    samples = previous_results[file_path]
    if not samples:
        return False

    timestamps = [
        s.metadata["processed_at"] for s in samples if s.metadata.get("processed_at")
    ]
    if not timestamps:
        return False

    max_processed_at = max(datetime.fromisoformat(t) for t in timestamps)
    return datetime.fromisoformat(input_processed_at) <= max_processed_at


def merge_results(
    reused: Dict[str, List[MultimodalSample]],
    new_results: List[MultimodalSample],
    current_file_paths: set,
) -> List[MultimodalSample]:
    """Combine reused samples and newly processed samples into a list.

    Args:
        reused: Dictionary mapping ``file_path`` to a list of saved samples to keep.
        new_results: List of newly processed samples.
        current_file_paths: Set of source file paths present in the current run.

    Returns:
        A list of samples containing reused entries followed by newly processed
        ones, with deleted-file entries removed.
    """
    merged: List[MultimodalSample] = []

    for file_path, samples in reused.items():
        if file_path in current_file_paths:
            merged.extend(samples)

    for sample in new_results:
        if sample.metadata["file_path"] in current_file_paths:
            merged.append(sample)

    return merged
