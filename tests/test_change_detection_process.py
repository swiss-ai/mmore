import json

from mmore.process.dispatcher import DispatcherConfig
from mmore.process.previous_results import (
    is_reusable_process,
    load_previous_results,
    merge_results,
)
from mmore.run_process import ProcessInference
from mmore.type import MultimodalSample


def _sample(file_path, **metadata) -> MultimodalSample:
    return MultimodalSample.from_dict(
        {
            "text": "x",
            "modalities": [],
            "metadata": {"file_path": file_path, **metadata},
        }
    )


class TestProcessStageReuse:
    """Test the full process-stage incremental workflow."""

    def _write_jsonl(self, path, samples):
        with open(path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

    def test_skips_unchanged_files(self, tmp_path):
        """Files with mtime <= processed_at are reused."""
        doc = tmp_path / "doc.pdf"
        doc.write_text("content")

        prev_path = tmp_path / "prev.jsonl"
        self._write_jsonl(
            str(prev_path),
            [
                {
                    "text": "old result",
                    "modalities": [],
                    "metadata": {
                        "file_path": str(doc),
                        "processed_at": "2099-01-01T00:00:00",
                        "processor_type": "PDFProcessor",
                    },
                },
            ],
        )

        previous = load_previous_results(str(prev_path))
        assert is_reusable_process(str(doc), previous) is True

    def test_reprocesses_modified_files(self, tmp_path):
        """Files with mtime > processed_at are not reused."""
        doc = tmp_path / "doc.pdf"
        doc.write_text("content")

        prev_path = tmp_path / "prev.jsonl"
        self._write_jsonl(
            str(prev_path),
            [
                {
                    "text": "old result",
                    "modalities": [],
                    "metadata": {
                        "file_path": str(doc),
                        "processed_at": "2000-01-01T00:00:00",
                        "processor_type": "PDFProcessor",
                    },
                },
            ],
        )

        previous = load_previous_results(str(prev_path))
        assert is_reusable_process(str(doc), previous) is False

    def test_processes_new_files(self, tmp_path):
        """Files not in previous results are not reusable."""
        new_doc = tmp_path / "new.pdf"
        new_doc.write_text("new content")
        assert is_reusable_process(str(new_doc), {}) is False

    def test_drops_deleted_from_merge(self):
        """Deleted files are excluded from merge output."""
        reused = {
            "/exists.pdf": [_sample("/exists.pdf")],
            "/deleted.pdf": [_sample("/deleted.pdf")],
        }
        new = [_sample("/new.txt")]
        current = {"/exists.pdf", "/new.txt"}

        merged = merge_results(reused, new, current)
        fps = {s.metadata["file_path"] for s in merged}
        assert fps == {"/exists.pdf", "/new.txt"}

    def test_metadata_fields_present(self, tmp_path):
        """Previous results have expected metadata fields."""
        prev_path = tmp_path / "prev.jsonl"
        self._write_jsonl(
            str(prev_path),
            [
                {
                    "text": "x",
                    "modalities": [],
                    "metadata": {
                        "file_path": "/x.pdf",
                        "processed_at": "2026-01-01T00:00:00",
                        "processor_type": "PDFProcessor",
                    },
                },
            ],
        )

        previous = load_previous_results(str(prev_path))
        sample = previous["/x.pdf"][0]
        assert "processed_at" in sample.metadata
        assert "processor_type" in sample.metadata


class TestProcessInferenceConfig:
    def test_previous_results_default_none(self):
        config = ProcessInference(
            data_path=".",
            google_drive_ids=[],
            dispatcher_config=DispatcherConfig(output_path="/tmp/test_mmore_proc"),
        )
        assert config.previous_results is None

    def test_previous_results_can_be_set(self):
        config = ProcessInference(
            data_path=".",
            google_drive_ids=[],
            dispatcher_config=DispatcherConfig(output_path="/tmp/test_mmore_proc2"),
            previous_results="/path/to/prev.jsonl",
        )
        assert config.previous_results == "/path/to/prev.jsonl"
