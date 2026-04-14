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
# load_previous_results
# ---------------------------------------------------------------------------


class TestLoadPreviousResults:
    def test_loads_and_indexes_by_file_path(self, tmp_path, make_sample, write_jsonl):
        jsonl = str(tmp_path / "results.jsonl")
        samples = [
            make_sample("/data/a.pdf", processed_at="2026-01-01T10:00:00"),
            make_sample("/data/a.pdf", processed_at="2026-01-01T10:00:01"),
            make_sample("/data/b.txt", processed_at="2026-01-01T11:00:00"),
        ]
        write_jsonl(jsonl, samples)

        result = load_previous_results(jsonl)
        assert set(result.keys()) == {"/data/a.pdf", "/data/b.txt"}
        assert len(result["/data/a.pdf"]) == 2
        assert len(result["/data/b.txt"]) == 1

    def test_samples_preserved(self, tmp_path, make_sample, write_jsonl):
        jsonl = str(tmp_path / "results.jsonl")
        sample = make_sample(
            "/data/a.pdf", text="custom content", processed_at="2026-01-01T10:00:00"
        )
        write_jsonl(jsonl, [sample])

        result = load_previous_results(jsonl)
        assert isinstance(result["/data/a.pdf"][0], MultimodalSample)
        assert result["/data/a.pdf"][0].text == "custom content"
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


# ---------------------------------------------------------------------------
# is_reusable_process
# ---------------------------------------------------------------------------


class TestIsReusableProcess:
    def test_true_when_file_unchanged(self, tmp_path, make_sample):
        """mtime is older than processed_at yields reusable."""
        real_file = str(tmp_path / "doc.pdf")
        with open(real_file, "w") as f:
            f.write("content")

        future = (datetime.now() + timedelta(hours=1)).isoformat()
        previous = {real_file: [make_sample(real_file, processed_at=future)]}

        assert is_reusable_process(real_file, previous)

    def test_false_when_file_modified_after_processing(self, tmp_path, make_sample):
        """mtime is newer than processed_at yields not reusable."""
        real_file = str(tmp_path / "doc.pdf")
        with open(real_file, "w") as f:
            f.write("content")

        past = (datetime.now() - timedelta(hours=1)).isoformat()
        previous = {real_file: [make_sample(real_file, processed_at=past)]}

        assert not is_reusable_process(real_file, previous)

    def test_false_when_not_in_previous_results(self, tmp_path):
        real_file = str(tmp_path / "doc.pdf")
        with open(real_file, "w") as f:
            f.write("content")

        assert not is_reusable_process(real_file, {})

    def test_uses_max_processed_at_across_samples(self, tmp_path, make_sample):
        """When a file has multiple cached samples, uses the max processed_at."""
        real_file = str(tmp_path / "doc.pdf")
        with open(real_file, "w") as f:
            f.write("content")

        # One sample has past processed_at, another has future → max is future → reusable
        past = (datetime.now() - timedelta(hours=2)).isoformat()
        future = (datetime.now() + timedelta(hours=2)).isoformat()
        previous = {
            real_file: [
                make_sample(real_file, processed_at=past),
                make_sample(real_file, processed_at=future),
            ]
        }

        assert is_reusable_process(real_file, previous)

    def test_uses_max_processed_at_all_past(self, tmp_path, make_sample):
        """All cached samples have past processed_at → not reusable."""
        real_file = str(tmp_path / "doc.pdf")
        with open(real_file, "w") as f:
            f.write("content")

        past1 = (datetime.now() - timedelta(hours=3)).isoformat()
        past2 = (datetime.now() - timedelta(hours=1)).isoformat()
        previous = {
            real_file: [
                make_sample(real_file, processed_at=past1),
                make_sample(real_file, processed_at=past2),
            ]
        }

        assert not is_reusable_process(real_file, previous)

    def test_mtime_equal_to_processed_at_is_reusable(self, tmp_path, make_sample):
        """When mtime == processed_at, should be considered reusable (<=)."""
        real_file = str(tmp_path / "doc.pdf")
        with open(real_file, "w") as f:
            f.write("content")

        # Get the actual mtime and use it as processed_at
        mtime = os.path.getmtime(real_file)
        processed_at = datetime.fromtimestamp(mtime).isoformat()
        previous = {real_file: [make_sample(real_file, processed_at=processed_at)]}

        assert is_reusable_process(real_file, previous)


# ---------------------------------------------------------------------------
# is_reusable_postprocess
# ---------------------------------------------------------------------------


class TestIsReusablePostprocess:
    def test_true_when_input_older_than_cached(self, make_sample):
        """input_processed_at <= cached processed_at → reusable."""
        file_path = "/data/doc.pdf"
        cached_time = "2026-03-01T12:00:00"
        input_time = "2026-03-01T11:00:00"  # older than cached
        previous = {file_path: [make_sample(file_path, processed_at=cached_time)]}

        assert is_reusable_postprocess(file_path, input_time, previous)

    def test_true_when_input_equal_to_cached(self, make_sample):
        """input_processed_at == cached processed_at → reusable."""
        file_path = "/data/doc.pdf"
        same_time = "2026-03-01T12:00:00"
        previous = {file_path: [make_sample(file_path, processed_at=same_time)]}

        assert is_reusable_postprocess(file_path, same_time, previous)

    def test_false_when_input_newer_than_cached(self, make_sample):
        """input_processed_at > cached processed_at → not reusable."""
        file_path = "/data/doc.pdf"
        cached_time = "2026-03-01T10:00:00"
        input_time = "2026-03-01T12:00:00"  # newer than cached
        previous = {file_path: [make_sample(file_path, processed_at=cached_time)]}

        assert not is_reusable_postprocess(file_path, input_time, previous)

    def test_false_when_not_in_previous_results(self):
        file_path = "/data/doc.pdf"
        assert not is_reusable_postprocess(file_path, "2026-03-01T12:00:00", {})

    def test_uses_max_processed_at_across_cached_samples(self, make_sample):
        """Uses the max processed_at from all cached samples for the file."""
        file_path = "/data/doc.pdf"
        early = "2026-03-01T08:00:00"
        late = "2026-03-01T14:00:00"
        previous = {
            file_path: [
                make_sample(file_path, processed_at=early),
                make_sample(file_path, processed_at=late),
            ]
        }

        # input processed at noon, max cached is 14:00 → noon <= 14:00 → reusable
        assert is_reusable_postprocess(file_path, "2026-03-01T12:00:00", previous)

    def test_uses_max_processed_at_all_older_than_input(self, make_sample):
        """Even taking the max of cached, it's still older than input → not reusable."""
        file_path = "/data/doc.pdf"
        t1 = "2026-03-01T08:00:00"
        t2 = "2026-03-01T10:00:00"
        previous = {
            file_path: [
                make_sample(file_path, processed_at=t1),
                make_sample(file_path, processed_at=t2),
            ]
        }
        # input at noon, max cached is 10:00 → noon > 10:00 → not reusable
        assert not is_reusable_postprocess(file_path, "2026-03-01T12:00:00", previous)


# ---------------------------------------------------------------------------
# merge_results
# ---------------------------------------------------------------------------


class TestMergeResults:
    def test_combines_reused_and_new(self, make_sample):
        current_files = {"/data/a.pdf", "/data/b.txt"}
        reused = {
            "/data/a.pdf": [
                make_sample("/data/a.pdf", processed_at="2026-01-01T10:00:00")
            ]
        }
        new_results = [make_sample("/data/b.txt", processed_at="2026-01-02T10:00:00")]

        result = merge_results(reused, new_results, current_files)
        assert len(result) == 2
        file_paths = {r.metadata["file_path"] for r in result}
        assert file_paths == {"/data/a.pdf", "/data/b.txt"}

    def test_drops_deleted_files(self, make_sample):
        """Entries whose file_path is not in current_file_paths are excluded."""
        current_files = {"/data/b.txt"}
        reused = {
            "/data/a.pdf": [
                make_sample("/data/a.pdf", processed_at="2026-01-01T10:00:00")
            ]
        }
        new_results = [make_sample("/data/b.txt", processed_at="2026-01-02T10:00:00")]

        result = merge_results(reused, new_results, current_files)
        assert len(result) == 1
        assert result[0].metadata["file_path"] == "/data/b.txt"

    def test_empty_reused_returns_only_new(self, make_sample):
        current_files = {"/data/b.txt"}
        new_results = [make_sample("/data/b.txt", processed_at="2026-01-02T10:00:00")]

        result = merge_results({}, new_results, current_files)
        assert len(result) == 1
        assert result[0].metadata["file_path"] == "/data/b.txt"

    def test_empty_new_returns_only_reused(self, make_sample):
        current_files = {"/data/a.pdf"}
        reused = {
            "/data/a.pdf": [
                make_sample("/data/a.pdf", processed_at="2026-01-01T10:00:00")
            ]
        }

        result = merge_results(reused, [], current_files)
        assert len(result) == 1
        assert result[0].metadata["file_path"] == "/data/a.pdf"

    def test_both_empty_returns_empty_list(self):
        result = merge_results({}, [], set())
        assert result == []

    def test_multiple_samples_per_file_included(self, make_sample):
        current_files = {"/data/a.pdf"}
        reused = {
            "/data/a.pdf": [
                make_sample("/data/a.pdf", processed_at="2026-01-01T10:00:00"),
                make_sample("/data/a.pdf", processed_at="2026-01-01T10:00:01"),
            ]
        }

        result = merge_results(reused, [], current_files)
        assert len(result) == 2

    def test_new_results_with_deleted_file_path_dropped(self, make_sample):
        """new_results entries whose file_path is not in current_file_paths are also dropped."""
        current_files = {"/data/a.pdf"}
        # new_results contains a file that is no longer current
        new_results = [
            make_sample("/data/a.pdf", processed_at="2026-01-02T10:00:00"),
            make_sample("/data/deleted.pdf", processed_at="2026-01-02T11:00:00"),
        ]

        result = merge_results({}, new_results, current_files)
        assert len(result) == 1
        assert result[0].metadata["file_path"] == "/data/a.pdf"

    def test_returns_flat_list(self, make_sample):
        """merge_results always returns a flat list of MultimodalSample."""
        current_files = {"/data/a.pdf", "/data/b.txt"}
        reused = {
            "/data/a.pdf": [
                make_sample("/data/a.pdf", processed_at="2026-01-01T10:00:00"),
                make_sample("/data/a.pdf", processed_at="2026-01-01T10:00:01"),
            ]
        }
        new_results = [
            make_sample("/data/b.txt", processed_at="2026-01-02T10:00:00"),
        ]

        result = merge_results(reused, new_results, current_files)
        assert isinstance(result, list)
        assert all(isinstance(r, MultimodalSample) for r in result)
        assert len(result) == 3
