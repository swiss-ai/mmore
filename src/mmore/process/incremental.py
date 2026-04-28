import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Set

from ..type import MultimodalSample

logger = logging.getLogger(__name__)


def _iter_samples_jsonl(path: str):
    """Helper function to stream line by line a JSONL to avoid loading it fully in memory."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Previous results file not found: {path}")
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield MultimodalSample.from_dict(json.loads(line))


def load_previous_process_results(path: str) -> Dict[str, MultimodalSample]:
    """Index samples by ``metadata.file_path`` for the processing pipeline,
    keeping the latest ``processed_at`` if there are any duplicates."""
    samples_by_file_path: Dict[str, List[MultimodalSample]] = {}
    for sample in _iter_samples_jsonl(path):
        filepath = str(sample.metadata["file_path"])
        samples_by_file_path.setdefault(filepath, []).append(sample)

    index: Dict[str, MultimodalSample] = {}
    for file_path, samples in samples_by_file_path.items():
        if len(samples) > 1:
            logger.warning(
                "Duplicate samples for file_path %s: keeping latest processed_at, "
                "dropping %d samples",
                file_path,
                len(samples) - 1,
            )

        index[file_path] = max(
            samples,
            key=lambda s: datetime.fromisoformat(str(s.metadata["processed_at"]))
            if s.metadata.get("processed_at") is not None
            else datetime.min,
        )
    return index


def load_previous_postprocess_results(
    path: str,
) -> Dict[str, List[MultimodalSample]]:
    """Index samples by ``metadata.file_path`` for the post-processing pipeline."""
    index: Dict[str, List[MultimodalSample]] = {}
    for sample in _iter_samples_jsonl(path):
        index.setdefault(str(sample.metadata["file_path"]), []).append(sample)
    return index


def is_reusable_process(file_path: str, previous: Dict[str, MultimodalSample]) -> bool:
    """Check whether the previous processed sample the given file can be reused.

    Conditions (all required):
    - ``file_path`` is present in ``previous``
    - the cached sample has a ``processed_at`` timestamp
    - the source file has not been modified since (``file_mtime <= processed_at``)
    """
    sample = previous.get(file_path)
    if sample is None:
        return False

    processed_at_value = sample.metadata.get("processed_at")
    if processed_at_value is None:
        return False

    processed_at_str = str(processed_at_value)
    processed_at = datetime.fromisoformat(processed_at_str)
    if not os.path.exists(file_path):
        return False
    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
    return file_mtime <= processed_at


def is_reusable_postprocess(
    file_path: str,
    input_processed_at: str,
    previous: Dict[str, List[MultimodalSample]],
) -> bool:
    """Check whether the previous post-processed samples of the given file can be reused.

    Conditions (all required):
    - ``file_path`` has at least one cached sample in ``previous``
    - every cached sample has a ``processed_at`` timestamp
    - ``input_processed_at <= min(cached processed_at)``
    """
    samples = previous.get(file_path)
    if not samples:
        return False

    timestamps: List[datetime] = []
    for s in samples:
        timestamp_value = s.metadata.get("processed_at")
        if timestamp_value is None:
            return False
        timestamp_str = str(timestamp_value)
        timestamps.append(datetime.fromisoformat(timestamp_str))

    return datetime.fromisoformat(input_processed_at) <= min(timestamps)


def merge_results(
    reused: Dict[str, List[MultimodalSample]],
    new_results: List[MultimodalSample],
    current_file_paths: Set[str],
) -> List[MultimodalSample]:
    """Combine reused and newly processed/post-processed samples."""
    merged: List[MultimodalSample] = []
    for file_path, samples in reused.items():
        if file_path in current_file_paths:
            merged.extend(samples)
    for sample in new_results:
        if sample.metadata["file_path"] in current_file_paths:
            merged.append(sample)
    return merged
