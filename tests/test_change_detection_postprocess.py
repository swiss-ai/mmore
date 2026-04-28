from mmore.process.incremental import (
    is_reusable_postprocess,
    load_previous_postprocess_results,
    merge_results,
)
from mmore.process.post_processor.pipeline import OutputConfig, PPPipelineConfig


class TestPostProcessPipelineReuse:
    """Test the post-process pipeline incremental workflow."""

    def test_reuses_unchanged_documents(self, tmp_path, make_sample, write_jsonl):
        """Document groups where input processed_at <= cached processed_at are reused."""
        prev_path = tmp_path / "prev_pp.jsonl"
        write_jsonl(
            str(prev_path),
            [
                make_sample(
                    "/a.pdf",
                    text="chunked",
                    processed_at="2026-06-01T00:00:00",
                ),
            ],
        )

        previous = load_previous_postprocess_results(str(prev_path))
        assert is_reusable_postprocess("/a.pdf", "2026-01-01T00:00:00", previous)

    def test_reprocesses_changed_documents(self, tmp_path, make_sample, write_jsonl):
        """Document groups where input processed_at > cached processed_at need reprocessing."""
        prev_path = tmp_path / "prev_pp.jsonl"
        write_jsonl(
            str(prev_path),
            [
                make_sample(
                    "/doc.pdf",
                    text="old chunked",
                    processed_at="2026-01-01T00:00:00",
                ),
            ],
        )

        previous = load_previous_postprocess_results(str(prev_path))
        assert not is_reusable_postprocess("/doc.pdf", "2026-06-01T00:00:00", previous)

    def test_processes_new_documents(self):
        """New documents not in previous results are not reusable."""
        assert not is_reusable_postprocess("/new.pdf", "2026-01-01T00:00:00", {})

    def test_drops_deleted_documents(self, tmp_path, make_sample):
        """Documents absent from input are dropped from merge."""
        exists = str(tmp_path / "exists.pdf")
        deleted = str(tmp_path / "deleted.pdf")
        new_file = str(tmp_path / "new.txt")

        reused = {
            exists: [make_sample(exists)],
            deleted: [make_sample(deleted)],
        }
        new = [make_sample(new_file)]
        current_fps = {exists, new_file}

        merged = merge_results(reused, new, current_fps)
        fps = {str(s.metadata["file_path"]) for s in merged}
        assert fps == {exists, new_file}


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
