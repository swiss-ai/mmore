import logging
import os
from datetime import datetime, timedelta

import pytest

from mmore.process.incremental import (
    is_reusable_postprocess,
    is_reusable_process,
    load_previous_postprocess_results,
    load_previous_process_results,
    merge_results,
)
from mmore.type import MultimodalSample

# ---------------------------------------------------------------------------
# load_previous_process_results
# ---------------------------------------------------------------------------


class TestLoadPreviousProcessResults:
    def test_returns_single_sample_per_file_path(
        self, tmp_path, make_sample, write_jsonl
    ):
        jsonl = str(tmp_path / "results.jsonl")
        samples = [
            make_sample("/data/a.pdf", processed_at="2026-01-01T10:00:00"),
            make_sample("/data/b.txt", processed_at="2026-01-01T11:00:00"),
        ]
        write_jsonl(jsonl, samples)

        result = load_previous_process_results(jsonl)
        assert set(result.keys()) == {"/data/a.pdf", "/data/b.txt"}
        assert isinstance(result["/data/a.pdf"], MultimodalSample)
        assert isinstance(result["/data/b.txt"], MultimodalSample)

    def test_duplicates_collapse_to_latest_processed_at(
        self, tmp_path, make_sample, write_jsonl, caplog
    ):
        jsonl = str(tmp_path / "results.jsonl")
        samples = [
            make_sample("/data/a.pdf", text="old", processed_at="2026-01-01T10:00:00"),
            make_sample("/data/a.pdf", text="new", processed_at="2026-01-01T10:00:05"),
            make_sample("/data/a.pdf", text="mid", processed_at="2026-01-01T10:00:02"),
        ]
        write_jsonl(jsonl, samples)

        with caplog.at_level(logging.WARNING):
            result = load_previous_process_results(jsonl)

        assert set(result.keys()) == {"/data/a.pdf"}
        assert result["/data/a.pdf"].text == "new"
        assert any("/data/a.pdf" in rec.message for rec in caplog.records)

    def test_missing_processed_at_loses_tie_to_present(
        self, tmp_path, make_sample, write_jsonl
    ):
        jsonl = str(tmp_path / "results.jsonl")
        samples = [
            make_sample("/data/a.pdf", text="no_ts"),
            make_sample(
                "/data/a.pdf", text="has_ts", processed_at="2026-01-01T10:00:00"
            ),
        ]
        write_jsonl(jsonl, samples)

        result = load_previous_process_results(jsonl)
        assert result["/data/a.pdf"].text == "has_ts"

    def test_empty_file_returns_empty_dict(self, tmp_path):
        jsonl = str(tmp_path / "empty.jsonl")
        open(jsonl, "w").close()

        result = load_previous_process_results(jsonl)
        assert result == {}

    def test_raises_file_not_found_when_missing(self, tmp_path):
        missing = str(tmp_path / "nonexistent.jsonl")
        with pytest.raises(FileNotFoundError):
            load_previous_process_results(missing)


# ---------------------------------------------------------------------------
# load_previous_postprocess_results
# ---------------------------------------------------------------------------


class TestLoadPreviousPostprocessResults:
    def test_preserves_all_samples_per_file_path(
        self, tmp_path, make_sample, write_jsonl
    ):
        jsonl = str(tmp_path / "results.jsonl")
        samples = [
            make_sample("/data/a.pdf", processed_at="2026-01-01T10:00:00"),
            make_sample("/data/a.pdf", processed_at="2026-01-01T10:00:01"),
            make_sample("/data/b.txt", processed_at="2026-01-01T11:00:00"),
        ]
        write_jsonl(jsonl, samples)

        result = load_previous_postprocess_results(jsonl)
        assert set(result.keys()) == {"/data/a.pdf", "/data/b.txt"}
        assert len(result["/data/a.pdf"]) == 2
        assert len(result["/data/b.txt"]) == 1

    def test_samples_preserved(self, tmp_path, make_sample, write_jsonl):
        jsonl = str(tmp_path / "results.jsonl")
        sample = make_sample(
            "/data/a.pdf", text="custom content", processed_at="2026-01-01T10:00:00"
        )
        write_jsonl(jsonl, [sample])

        result = load_previous_postprocess_results(jsonl)
        assert result["/data/a.pdf"][0].text == "custom content"

    def test_empty_file_returns_empty_dict(self, tmp_path):
        jsonl = str(tmp_path / "empty.jsonl")
        open(jsonl, "w").close()

        assert load_previous_postprocess_results(jsonl) == {}

    def test_raises_file_not_found_when_missing(self, tmp_path):
        missing = str(tmp_path / "nonexistent.jsonl")
        with pytest.raises(FileNotFoundError):
            load_previous_postprocess_results(missing)


# ---------------------------------------------------------------------------
# is_reusable_process
# ---------------------------------------------------------------------------


class TestIsReusableProcess:
    def test_true_when_file_unchanged(self, tmp_path, make_sample):
        real_file = str(tmp_path / "doc.pdf")
        with open(real_file, "w") as f:
            f.write("content")

        future = (datetime.now() + timedelta(hours=1)).isoformat()
        previous = {real_file: make_sample(real_file, processed_at=future)}

        assert is_reusable_process(real_file, previous)

    def test_false_when_file_modified_after_processing(self, tmp_path, make_sample):
        real_file = str(tmp_path / "doc.pdf")
        with open(real_file, "w") as f:
            f.write("content")

        past = (datetime.now() - timedelta(hours=1)).isoformat()
        previous = {real_file: make_sample(real_file, processed_at=past)}

        assert not is_reusable_process(real_file, previous)

    def test_false_when_not_in_previous_results(self, tmp_path):
        real_file = str(tmp_path / "doc.pdf")
        with open(real_file, "w") as f:
            f.write("content")

        assert not is_reusable_process(real_file, {})

    def test_false_when_cached_missing_processed_at(self, tmp_path, make_sample):
        real_file = str(tmp_path / "doc.pdf")
        with open(real_file, "w") as f:
            f.write("content")

        previous = {real_file: make_sample(real_file)}
        assert not is_reusable_process(real_file, previous)

    def test_mtime_equal_to_processed_at_is_reusable(self, tmp_path, make_sample):
        real_file = str(tmp_path / "doc.pdf")
        with open(real_file, "w") as f:
            f.write("content")

        mtime = os.path.getmtime(real_file)
        processed_at = datetime.fromtimestamp(mtime).isoformat()
        previous = {real_file: make_sample(real_file, processed_at=processed_at)}

        assert is_reusable_process(real_file, previous)


# ---------------------------------------------------------------------------
# is_reusable_postprocess
# ---------------------------------------------------------------------------


class TestIsReusablePostprocess:
    def test_true_when_input_older_than_all_cached(self, make_sample):
        file_path = "/data/doc.pdf"
        input_time = "2026-03-01T10:00:00"
        previous = {
            file_path: [
                make_sample(file_path, processed_at="2026-03-01T11:00:00"),
                make_sample(file_path, processed_at="2026-03-01T12:00:00"),
            ]
        }

        assert is_reusable_postprocess(file_path, input_time, previous)

    def test_true_when_input_equal_to_min_cached(self, make_sample):
        file_path = "/data/doc.pdf"
        same_time = "2026-03-01T12:00:00"
        previous = {
            file_path: [
                make_sample(file_path, processed_at=same_time),
                make_sample(file_path, processed_at="2026-03-01T13:00:00"),
            ]
        }

        assert is_reusable_postprocess(file_path, same_time, previous)

    def test_false_when_input_newer_than_min_cached(self, make_sample):
        """Input between early and late cached times, min is early, not reusable."""
        file_path = "/data/doc.pdf"
        previous = {
            file_path: [
                make_sample(file_path, processed_at="2026-03-01T08:00:00"),
                make_sample(file_path, processed_at="2026-03-01T14:00:00"),
            ]
        }
        assert not is_reusable_postprocess(file_path, "2026-03-01T12:00:00", previous)

    def test_false_when_input_newer_than_all_cached(self, make_sample):
        file_path = "/data/doc.pdf"
        previous = {
            file_path: [
                make_sample(file_path, processed_at="2026-03-01T08:00:00"),
                make_sample(file_path, processed_at="2026-03-01T10:00:00"),
            ]
        }
        assert not is_reusable_postprocess(file_path, "2026-03-01T12:00:00", previous)

    def test_false_when_not_in_previous_results(self):
        assert not is_reusable_postprocess("/data/doc.pdf", "2026-03-01T12:00:00", {})

    def test_false_when_any_cached_sample_missing_processed_at(self, make_sample):
        file_path = "/data/doc.pdf"
        previous = {
            file_path: [
                make_sample(file_path, processed_at="2026-03-01T14:00:00"),
                make_sample(file_path),
            ]
        }
        assert not is_reusable_postprocess(file_path, "2026-03-01T10:00:00", previous)


# ---------------------------------------------------------------------------
# merge_results (unchanged behavior)
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
        file_paths = {str(r.metadata["file_path"]) for r in result}
        assert file_paths == {"/data/a.pdf", "/data/b.txt"}

    def test_drops_deleted_files(self, make_sample):
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
        assert merge_results({}, [], set()) == []

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
        current_files = {"/data/a.pdf"}
        new_results = [
            make_sample("/data/a.pdf", processed_at="2026-01-02T10:00:00"),
            make_sample("/data/deleted.pdf", processed_at="2026-01-02T11:00:00"),
        ]

        result = merge_results({}, new_results, current_files)
        assert len(result) == 1
        assert result[0].metadata["file_path"] == "/data/a.pdf"
