import json

from mmore.process.post_processor.pipeline import OutputConfig, PPPipelineConfig
from mmore.process.previous_results import (
    is_reusable_postprocess,
    load_previous_results,
    merge_results,
)


class TestPostprocessStageReuse:
    """Test the post-process incremental workflow."""

    def _write_jsonl(self, path, samples):
        with open(path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

    def test_reuses_unchanged_documents(self, tmp_path):
        """Document groups where input processed_at <= cached processed_at are reused."""
        prev_path = tmp_path / "prev_pp.jsonl"
        self._write_jsonl(
            str(prev_path),
            [
                {
                    "text": "chunked",
                    "modalities": [],
                    "metadata": {
                        "file_path": "/a.pdf",
                        "processed_at": "2026-06-01T00:00:00",
                    },
                },
            ],
        )

        previous = load_previous_results(str(prev_path))
        assert (
            is_reusable_postprocess("/a.pdf", "2026-01-01T00:00:00", previous) is True
        )

    def test_reprocesses_changed_documents(self, tmp_path):
        """Document groups where input processed_at > cached processed_at need reprocessing."""
        prev_path = tmp_path / "prev_pp.jsonl"
        self._write_jsonl(
            str(prev_path),
            [
                {
                    "text": "old chunked",
                    "modalities": [],
                    "metadata": {
                        "file_path": "/a.pdf",
                        "processed_at": "2026-01-01T00:00:00",
                    },
                },
            ],
        )

        previous = load_previous_results(str(prev_path))
        assert (
            is_reusable_postprocess("/a.pdf", "2026-06-01T00:00:00", previous) is False
        )

    def test_processes_new_documents(self):
        """New documents not in previous results are not reusable."""
        assert is_reusable_postprocess("/new.pdf", "2026-01-01T00:00:00", {}) is False

    def test_drops_deleted_documents(self):
        """Documents absent from input are dropped from merge."""
        reused = {
            "/a.pdf": [
                {"text": "a", "modalities": [], "metadata": {"file_path": "/a.pdf"}}
            ],
            "/gone.pdf": [
                {
                    "text": "gone",
                    "modalities": [],
                    "metadata": {"file_path": "/gone.pdf"},
                }
            ],
        }
        current_fps = {"/a.pdf", "/new.pdf"}
        new = [{"text": "new", "modalities": [], "metadata": {"file_path": "/new.pdf"}}]

        merged = merge_results(reused, new, current_fps)
        fps = {s["metadata"]["file_path"] for s in merged}
        assert fps == {"/a.pdf", "/new.pdf"}


class TestPPPipelineConfig:
    def test_previous_results_default_none(self):
        config = PPPipelineConfig(
            pp_modules=[],
            output=OutputConfig(output_path="/tmp/test_mmore_pp.jsonl"),
        )
        assert config.previous_results is None

    def test_previous_results_can_be_set(self):
        config = PPPipelineConfig(
            pp_modules=[],
            output=OutputConfig(output_path="/tmp/test_mmore_pp2.jsonl"),
            previous_results="/path/to/prev.jsonl",
        )
        assert config.previous_results == "/path/to/prev.jsonl"
