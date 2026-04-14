import json
import os
from datetime import datetime, timedelta

import pytest

from mmore.process.previous_results import (
    is_reusable_postprocess,
    is_reusable_process,
    load_previous_results,
    merge_results,
)
from mmore.type import MultimodalSample

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sample_dict(file_path: str, processed_at: str) -> dict:
    """Build a minimal sample dict as stored in a JSONL file."""
    return {
        "text": f"content of {file_path}",
        "modalities": [],
        "metadata": {
            "file_path": file_path,
            "processed_at": processed_at,
        },
    }


def _make_sample(file_path: str, processed_at: str) -> MultimodalSample:
    """Build a minimal MultimodalSample for tests."""
    return MultimodalSample.from_dict(_make_sample_dict(file_path, processed_at))


def _write_jsonl(path: str, samples: list[dict]) -> None:
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


# ---------------------------------------------------------------------------
# load_previous_results
# ---------------------------------------------------------------------------


class TestLoadPreviousResults:
    def test_loads_and_indexes_by_file_path(self, tmp_path):
        jsonl = str(tmp_path / "results.jsonl")
        samples = [
            _make_sample_dict("/data/a.pdf", "2026-01-01T10:00:00"),
            _make_sample_dict("/data/a.pdf", "2026-01-01T10:00:01"),
            _make_sample_dict("/data/b.txt", "2026-01-01T11:00:00"),
        ]
        _write_jsonl(jsonl, samples)

        result = load_previous_results(jsonl)
        assert set(result.keys()) == {"/data/a.pdf", "/data/b.txt"}
        assert len(result["/data/a.pdf"]) == 2
        assert len(result["/data/b.txt"]) == 1

    def test_samples_preserved(self, tmp_path):
        jsonl = str(tmp_path / "results.jsonl")
        sample = _make_sample_dict("/data/a.pdf", "2026-01-01T10:00:00")
        _write_jsonl(jsonl, [sample])

        result = load_previous_results(jsonl)
        assert isinstance(result["/data/a.pdf"][0], MultimodalSample)
        assert result["/data/a.pdf"][0].text == "content of /data/a.pdf"
        assert (
            result["/data/a.pdf"][0].metadata["processed_at"] == "2026-01-01T10:00:00"
        )

    def test_empty_file_returns_empty_dict(self, tmp_path):
        jsonl = str(tmp_path / "empty.jsonl")
        jsonl_path = str(jsonl)
        open(jsonl_path, "w").close()

        result = load_previous_results(jsonl_path)
        assert result == {}

    def test_raises_file_not_found_when_missing(self, tmp_path):
        missing = str(tmp_path / "nonexistent.jsonl")
        with pytest.raises(FileNotFoundError):
            load_previous_results(missing)

    def test_single_sample_per_file(self, tmp_path):
        jsonl = str(tmp_path / "results.jsonl")
        samples = [
            _make_sample_dict("/data/x.docx", "2026-03-01T08:00:00"),
        ]
        _write_jsonl(jsonl, samples)

        result = load_previous_results(jsonl)
        assert len(result) == 1
        assert "/data/x.docx" in result
        assert len(result["/data/x.docx"]) == 1


# ---------------------------------------------------------------------------
# is_reusable_process
# ---------------------------------------------------------------------------


class TestIsReusableProcess:
    def test_true_when_file_unchanged(self, tmp_path):
        """mtime is older than processed_at → reusable."""
        real_file = str(tmp_path / "doc.pdf")
        with open(real_file, "w") as f:
            f.write("content")

        # processed_at is in the future relative to mtime → reusable
        future = (datetime.now() + timedelta(hours=1)).isoformat()
        previous = {real_file: [_make_sample(real_file, future)]}

        assert is_reusable_process(real_file, previous) is True

    def test_false_when_file_modified_after_processing(self, tmp_path):
        """mtime is newer than processed_at → not reusable."""
        real_file = str(tmp_path / "doc.pdf")
        with open(real_file, "w") as f:
            f.write("content")

        # processed_at is in the past relative to mtime → not reusable
        past = (datetime.now() - timedelta(hours=1)).isoformat()
        previous = {real_file: [_make_sample(real_file, past)]}

        assert is_reusable_process(real_file, previous) is False

    def test_false_when_not_in_previous_results(self, tmp_path):
        real_file = str(tmp_path / "doc.pdf")
        with open(real_file, "w") as f:
            f.write("content")

        assert is_reusable_process(real_file, {}) is False

    def test_uses_max_processed_at_across_samples(self, tmp_path):
        """When a file has multiple cached samples, uses the max processed_at."""
        real_file = str(tmp_path / "doc.pdf")
        with open(real_file, "w") as f:
            f.write("content")

        # One sample has past processed_at, another has future → max is future → reusable
        past = (datetime.now() - timedelta(hours=2)).isoformat()
        future = (datetime.now() + timedelta(hours=2)).isoformat()
        previous = {
            real_file: [
                _make_sample(real_file, past),
                _make_sample(real_file, future),
            ]
        }

        assert is_reusable_process(real_file, previous) is True

    def test_uses_max_processed_at_all_past(self, tmp_path):
        """All cached samples have past processed_at → not reusable."""
        real_file = str(tmp_path / "doc.pdf")
        with open(real_file, "w") as f:
            f.write("content")

        past1 = (datetime.now() - timedelta(hours=3)).isoformat()
        past2 = (datetime.now() - timedelta(hours=1)).isoformat()
        previous = {
            real_file: [
                _make_sample(real_file, past1),
                _make_sample(real_file, past2),
            ]
        }

        assert is_reusable_process(real_file, previous) is False

    def test_mtime_equal_to_processed_at_is_reusable(self, tmp_path):
        """When mtime == processed_at, should be considered reusable (<=)."""
        real_file = str(tmp_path / "doc.pdf")
        with open(real_file, "w") as f:
            f.write("content")

        # Get the actual mtime and use it as processed_at
        mtime = os.path.getmtime(real_file)
        processed_at = datetime.fromtimestamp(mtime).isoformat()
        previous = {real_file: [_make_sample(real_file, processed_at)]}

        assert is_reusable_process(real_file, previous) is True


# ---------------------------------------------------------------------------
# is_reusable_postprocess
# ---------------------------------------------------------------------------


class TestIsReusablePostprocess:
    def test_true_when_input_older_than_cached(self):
        """input_processed_at <= cached processed_at → reusable."""
        file_path = "/data/doc.pdf"
        cached_time = "2026-03-01T12:00:00"
        input_time = "2026-03-01T11:00:00"  # older than cached
        previous = {file_path: [_make_sample(file_path, cached_time)]}

        assert is_reusable_postprocess(file_path, input_time, previous) is True

    def test_true_when_input_equal_to_cached(self):
        """input_processed_at == cached processed_at → reusable."""
        file_path = "/data/doc.pdf"
        same_time = "2026-03-01T12:00:00"
        previous = {file_path: [_make_sample(file_path, same_time)]}

        assert is_reusable_postprocess(file_path, same_time, previous) is True

    def test_false_when_input_newer_than_cached(self):
        """input_processed_at > cached processed_at → not reusable."""
        file_path = "/data/doc.pdf"
        cached_time = "2026-03-01T10:00:00"
        input_time = "2026-03-01T12:00:00"  # newer than cached
        previous = {file_path: [_make_sample(file_path, cached_time)]}

        assert is_reusable_postprocess(file_path, input_time, previous) is False

    def test_false_when_not_in_previous_results(self):
        file_path = "/data/doc.pdf"
        assert is_reusable_postprocess(file_path, "2026-03-01T12:00:00", {}) is False

    def test_uses_max_processed_at_across_cached_samples(self):
        """Uses the max processed_at from all cached samples for the file."""
        file_path = "/data/doc.pdf"
        early = "2026-03-01T08:00:00"
        late = "2026-03-01T14:00:00"
        previous = {
            file_path: [
                _make_sample(file_path, early),
                _make_sample(file_path, late),
            ]
        }

        # input processed at noon, max cached is 14:00 → noon <= 14:00 → reusable
        assert (
            is_reusable_postprocess(file_path, "2026-03-01T12:00:00", previous) is True
        )

    def test_uses_max_processed_at_all_older_than_input(self):
        """Even taking the max of cached, it's still older than input → not reusable."""
        file_path = "/data/doc.pdf"
        t1 = "2026-03-01T08:00:00"
        t2 = "2026-03-01T10:00:00"
        previous = {
            file_path: [
                _make_sample(file_path, t1),
                _make_sample(file_path, t2),
            ]
        }
        # input at noon, max cached is 10:00 → noon > 10:00 → not reusable
        assert (
            is_reusable_postprocess(file_path, "2026-03-01T12:00:00", previous) is False
        )


# ---------------------------------------------------------------------------
# merge_results
# ---------------------------------------------------------------------------


class TestMergeResults:
    def test_combines_reused_and_new(self):
        current_files = {"/data/a.pdf", "/data/b.txt"}
        reused = {"/data/a.pdf": [_make_sample("/data/a.pdf", "2026-01-01T10:00:00")]}
        new_results = [_make_sample("/data/b.txt", "2026-01-02T10:00:00")]

        result = merge_results(reused, new_results, current_files)
        assert len(result) == 2
        file_paths = {r.metadata["file_path"] for r in result}
        assert file_paths == {"/data/a.pdf", "/data/b.txt"}

    def test_drops_deleted_files(self):
        """Entries whose file_path is not in current_file_paths are excluded."""
        current_files = {"/data/b.txt"}
        reused = {"/data/a.pdf": [_make_sample("/data/a.pdf", "2026-01-01T10:00:00")]}
        new_results = [_make_sample("/data/b.txt", "2026-01-02T10:00:00")]

        result = merge_results(reused, new_results, current_files)
        assert len(result) == 1
        assert result[0].metadata["file_path"] == "/data/b.txt"

    def test_empty_reused_returns_only_new(self):
        current_files = {"/data/b.txt"}
        new_results = [_make_sample("/data/b.txt", "2026-01-02T10:00:00")]

        result = merge_results({}, new_results, current_files)
        assert len(result) == 1
        assert result[0].metadata["file_path"] == "/data/b.txt"

    def test_empty_new_returns_only_reused(self):
        current_files = {"/data/a.pdf"}
        reused = {"/data/a.pdf": [_make_sample("/data/a.pdf", "2026-01-01T10:00:00")]}

        result = merge_results(reused, [], current_files)
        assert len(result) == 1
        assert result[0].metadata["file_path"] == "/data/a.pdf"

    def test_both_empty_returns_empty_list(self):
        result = merge_results({}, [], set())
        assert result == []

    def test_multiple_samples_per_file_included(self):
        current_files = {"/data/a.pdf"}
        reused = {
            "/data/a.pdf": [
                _make_sample("/data/a.pdf", "2026-01-01T10:00:00"),
                _make_sample("/data/a.pdf", "2026-01-01T10:00:01"),
            ]
        }

        result = merge_results(reused, [], current_files)
        assert len(result) == 2

    def test_new_results_with_deleted_file_path_dropped(self):
        """new_results entries whose file_path is not in current_file_paths are also dropped."""
        current_files = {"/data/a.pdf"}
        # new_results contains a file that is no longer current
        new_results = [
            _make_sample("/data/a.pdf", "2026-01-02T10:00:00"),
            _make_sample("/data/deleted.pdf", "2026-01-02T11:00:00"),
        ]

        result = merge_results({}, new_results, current_files)
        assert len(result) == 1
        assert result[0].metadata["file_path"] == "/data/a.pdf"

    def test_returns_flat_list(self):
        """merge_results always returns a flat list of MultimodalSample."""
        current_files = {"/data/a.pdf", "/data/b.txt"}
        reused = {
            "/data/a.pdf": [
                _make_sample("/data/a.pdf", "2026-01-01T10:00:00"),
                _make_sample("/data/a.pdf", "2026-01-01T10:00:01"),
            ]
        }
        new_results = [
            _make_sample("/data/b.txt", "2026-01-02T10:00:00"),
        ]

        result = merge_results(reused, new_results, current_files)
        assert isinstance(result, list)
        assert all(isinstance(r, MultimodalSample) for r in result)
        assert len(result) == 3
