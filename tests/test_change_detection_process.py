import pytest

from mmore.process.incremental import (
    is_reusable_process,
    load_previous_process_results,
    merge_results,
)


class TestProcessPipelineReuse:
    """Test the full process pipeline incremental workflow."""

    def test_skips_unchanged_files(self, tmp_path, make_sample, write_jsonl):
        """Files with mtime <= processed_at are reused."""
        doc = tmp_path / "doc.pdf"
        doc.write_text("content")

        prev_path = tmp_path / "prev.jsonl"
        write_jsonl(
            str(prev_path),
            [
                make_sample(
                    str(doc),
                    text="old result",
                    processed_at="2099-01-01T00:00:00",
                    processor_type="PDFProcessor",
                ),
            ],
        )

        previous = load_previous_process_results(str(prev_path))
        assert is_reusable_process(str(doc), previous)

    def test_reprocesses_modified_files(self, tmp_path, make_sample, write_jsonl):
        """Files with mtime > processed_at are not reused."""
        doc = tmp_path / "doc.pdf"
        doc.write_text("content")

        prev_path = tmp_path / "prev.jsonl"
        write_jsonl(
            str(prev_path),
            [
                make_sample(
                    str(doc),
                    text="old result",
                    processed_at="2000-01-01T00:00:00",
                    processor_type="PDFProcessor",
                ),
            ],
        )

        previous = load_previous_process_results(str(prev_path))
        assert not is_reusable_process(str(doc), previous)

    def test_processes_new_files(self, tmp_path):
        """Files not in previous results are not reusable."""
        new_doc = tmp_path / "new.pdf"
        new_doc.write_text("new content")
        assert not is_reusable_process(str(new_doc), {})

    def test_drops_deleted_from_merge(self, tmp_path, make_sample):
        """Deleted files are excluded from merge output."""
        exists = str(tmp_path / "exists.pdf")
        deleted = str(tmp_path / "deleted.pdf")
        new_file = str(tmp_path / "new.txt")

        reused = {
            exists: [make_sample(exists)],
            deleted: [make_sample(deleted)],
        }
        new = [make_sample(new_file)]
        current = {exists, new_file}

        merged = merge_results(reused, new, current)
        fps = {str(s.metadata["file_path"]) for s in merged}
        assert fps == {exists, new_file}

    @pytest.mark.parametrize(
        "include_url, expected_deleted",
        [(True, 0), (False, 1)],
        ids=["url_still_present", "url_removed"],
    )
    def test_deleted_count_with_url_in_previous(
        self, tmp_path, make_sample, write_jsonl, include_url, expected_deleted
    ):
        """URLs in previous results are only counted as deleted when not present in the crawl."""
        local_file = tmp_path / "doc.pdf"
        local_file.write_text("content")
        url = "https://example.com/page"

        prev_path = tmp_path / "prev.jsonl"
        write_jsonl(
            str(prev_path),
            [
                make_sample(str(local_file), processed_at="2099-01-01T00:00:00"),
                make_sample(url, processed_at="2099-01-01T00:00:00"),
            ],
        )

        previous = load_previous_process_results(str(prev_path))
        all_crawled_paths = {str(local_file)}
        if include_url:
            all_crawled_paths.add(url)

        n_deleted = len(set(previous.keys()) - all_crawled_paths)
        assert n_deleted == expected_deleted

    def test_metadata_fields_present(self, tmp_path, make_sample, write_jsonl):
        """Previous results have expected metadata fields."""
        prev_path = tmp_path / "prev.jsonl"
        write_jsonl(
            str(prev_path),
            [
                make_sample(
                    "/x.pdf",
                    processed_at="2026-01-01T00:00:00",
                    processor_type="PDFProcessor",
                ),
            ],
        )

        previous = load_previous_process_results(str(prev_path))
        sample = previous["/x.pdf"]
        assert "processed_at" in sample.metadata
        assert "processor_type" in sample.metadata
