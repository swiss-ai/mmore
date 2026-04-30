"""Unit tests for the two layers that produce `paragraph_positions`:

- `PDFProcessor._parse_pagination` turns marker-paginated text into
  `paragraph_starts: [(char_offset, page_id, para_idx), ...]` + clean text.
- `MultimodalChunker._assign_paragraph_positions` maps each chunk's byte
  range to the paragraphs it overlaps, producing `[[page, para], ...]`.

Both are pure Python with no Milvus / model dependencies.
"""

from chonkie import Chunk

from mmore.process.post_processor.chunker.multimodal import MultimodalChunker
from mmore.process.processors.pdf_processor import PDFProcessor
from mmore.type import MultimodalSample

# --------------------------------------------------------------------------
# _parse_pagination
# --------------------------------------------------------------------------


def test_parse_pagination_no_separators_returns_empty_and_unchanged_text():
    text = "Just plain text with no marker separators."
    paragraph_starts, clean_text = PDFProcessor._parse_pagination(text)

    assert paragraph_starts == []
    assert clean_text == text


def test_parse_pagination_multi_page_with_trailing_content():
    """Two paragraphs on page 0, one paragraph on page 1 (trailing)."""
    text = "Para A.\n\nPara B.\n\n{0}----\n\nPara C."

    paragraph_starts, clean_text = PDFProcessor._parse_pagination(text)

    assert paragraph_starts == [
        (0, 0, 0),
        (9, 0, 1),
        (16, 1, 0),
        (23, -1, -1),
    ]
    assert clean_text == "Para A.\n\nPara B.Para C."
    assert paragraph_starts[-1][0] == len(clean_text)


def test_parse_pagination_empty_trailing_does_not_create_extra_page():
    """Whitespace-only trailing content must not produce a phantom page."""
    text = "Hello.\n\n{0}----\n\n   "

    paragraph_starts, clean_text = PDFProcessor._parse_pagination(text)

    assert clean_text == "Hello."
    assert paragraph_starts == [(0, 0, 0), (6, -1, -1)]


# --------------------------------------------------------------------------
# _assign_paragraph_positions
# --------------------------------------------------------------------------


def _sample_with_starts(paragraph_starts, text="Para A.\n\nPara B.Para C."):
    return MultimodalSample(
        text=text,
        modalities=[],
        metadata={"file_path": "x", "paragraph_starts": paragraph_starts},
    )


def _chunk(start: int, end: int) -> Chunk:
    return Chunk(text="x", start_index=start, end_index=end, token_count=1)


def test_assign_paragraph_positions_empty_starts_returns_empty_dicts():
    """No paragraph_starts → each chunk gets an empty dict, not a key with []."""
    sample = _sample_with_starts([])
    chunks = [_chunk(0, 5), _chunk(5, 10)]

    result = MultimodalChunker._assign_paragraph_positions(None, sample, chunks)

    assert result == [{}, {}]


def test_assign_paragraph_positions_chunk_inside_single_paragraph():
    starts = [(0, 0, 0), (9, 0, 1), (16, 1, 0), (23, -1, -1)]
    sample = _sample_with_starts(starts)

    result = MultimodalChunker._assign_paragraph_positions(None, sample, [_chunk(0, 7)])

    assert result == [{"paragraph_positions": [[0, 0]]}]


def test_assign_paragraph_positions_chunk_spanning_two_paragraphs_same_page():
    starts = [(0, 0, 0), (9, 0, 1), (16, 1, 0), (23, -1, -1)]
    sample = _sample_with_starts(starts)

    result = MultimodalChunker._assign_paragraph_positions(
        None, sample, [_chunk(5, 13)]
    )

    assert result == [{"paragraph_positions": [[0, 0], [0, 1]]}]


def test_assign_paragraph_positions_chunk_spanning_page_boundary():
    starts = [(0, 0, 0), (9, 0, 1), (16, 1, 0), (23, -1, -1)]
    sample = _sample_with_starts(starts)

    result = MultimodalChunker._assign_paragraph_positions(
        None, sample, [_chunk(11, 19)]
    )

    assert result == [{"paragraph_positions": [[0, 1], [1, 0]]}]


def test_assign_paragraph_positions_end_to_end_from_parse_pagination():
    """Feed the real output of _parse_pagination into _assign_paragraph_positions
    to pin the full pipeline contract between the two layers."""
    text = "Para A.\n\nPara B.\n\n{0}----\n\nPara C."
    paragraph_starts, clean_text = PDFProcessor._parse_pagination(text)
    sample = MultimodalSample(
        text=clean_text,
        modalities=[],
        metadata={"file_path": "x", "paragraph_starts": paragraph_starts},
    )

    chunks = [
        _chunk(0, 7),  # "Para A." - page 0, para 0 only
        _chunk(16, 23),  # "Para C." - page 1, para 0 only
    ]

    result = MultimodalChunker._assign_paragraph_positions(None, sample, chunks)

    assert result == [
        {"paragraph_positions": [[0, 0]]},
        {"paragraph_positions": [[1, 0]]},
    ]
