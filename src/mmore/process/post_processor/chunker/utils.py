import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from chonkie import (
    BaseChunker,
    Chunk,
    SemanticChunker,
    SentenceChunker,
    TokenChunker,
    WordChunker,
)

logger = logging.getLogger(__name__)

# Regexes obtained from link below, then improved to detect empty cells
# and trailing/leading colons for alignment
# https://stackoverflow.com/questions/9837935/regex-for-markdown-table-syntax

# Matches a table row
_TABLE_ROW_RE = re.compile(r"^\|(?:[^|\r\n]*\|)+$")

# Matches the delimiter row
_TABLE_SEPARATOR_RE = re.compile(r"^(?:\| *:?-+:? *)+\|$")


def _strip_separator_cell(cell: str) -> str:
    """Normalize a separator cell to its minimal form, preserving alignment colons."""
    cell = cell.strip()
    match (cell.startswith(":"), cell.endswith(":")):
        case (True, True):
            return ":---:"
        case (True, False):
            return ":---"
        case (False, True):
            return "---:"
        case _:
            return "---"


def _strip_table_row(row: str) -> str:
    """Strip padding whitespace from each cell in a markdown table row."""
    if not row.strip().startswith("|"):
        return row

    parts = row.split("|")
    stripped = [p.strip() for p in parts[1:-1]]  # parts[0] and parts[-1] are empty

    if _TABLE_SEPARATOR_RE.match(row.strip()):
        stripped = [_strip_separator_cell(p) for p in stripped]

    return "| " + " | ".join(stripped) + " |"


def _strip_table_text(text: str) -> str:
    """Strip padding from all lines that are part of a markdown table row."""
    return "\n".join(_strip_table_row(line) for line in text.split("\n"))


@dataclass
class TableRegion:
    """A detected markdown table region within a text."""

    start_index: int  # char offset in original text
    end_index: int  # char offset in original text (exclusive)
    header: str  # header row + separator row
    body_rows: List[str] = field(default_factory=list)


def detect_markdown_tables(text: str) -> List[TableRegion]:
    """Detect markdown pipe-delimited tables in text.

    Scans line-by-line. A table is a sequence of consecutive lines starting/ending
    with '|' where the second line is a separator row (e.g. |---|---|).

    Args:
        text: The input text to scan for markdown tables.

    Returns:
        A list of TableRegion.
    """
    tables: List[TableRegion] = []
    lines = text.split("\n")

    i = 0  # tracks current line
    char_offset = 0  # tracks position in original text

    while i < len(lines):
        line = lines[i]

        # Check if this line could be the start of a table (header row)
        if _TABLE_ROW_RE.match(line.strip()) and i + 1 < len(lines):
            next_line = lines[i + 1].strip()

            # Second line must be a separator row
            if _TABLE_SEPARATOR_RE.match(next_line):
                table_start = char_offset
                header_line = line
                separator_line = lines[i + 1]
                header = header_line + "\n" + separator_line
                body_rows: List[str] = []

                # Advance past header + separator
                j = i + 2
                body_char_offset = char_offset + len(line) + 1 + len(lines[i + 1]) + 1

                # Collect body rows
                while j < len(lines) and _TABLE_ROW_RE.match(lines[j].strip()):
                    body_rows.append(lines[j])
                    j += 1

                # Calculate end index
                if body_rows:
                    table_end = body_char_offset + sum(len(r) + 1 for r in body_rows)
                    # Don't count the trailing newline after the last row if at end of text
                    if table_end > len(text):
                        table_end = len(text)
                else:
                    # Table with only header + separator, no body
                    table_end = char_offset + len(header_line) + 1 + len(separator_line)
                    if j < len(lines):
                        table_end += 1  # account for newline after separator

                tables.append(
                    TableRegion(
                        start_index=table_start,
                        end_index=table_end,
                        header=header,
                        body_rows=body_rows,
                    )
                )

                # Advance past the table
                char_offset = table_end
                i = j
                continue

        char_offset += len(line) + 1
        i += 1

    return tables


def chunk_table(
    table: TableRegion,
    max_tokens: int,
    count_tokens: Callable[[str], int],
) -> List[Chunk]:
    """Split a table into chunks, prepending the header to each chunk.

    If the table fits within max_tokens, returns it as a single chunk.
    Otherwise, greedily groups consecutive body rows, prepending the header
    to each group, until adding another row would exceed max_tokens.

    Args:
        table: The detected table region.
        max_tokens: Maximum tokens per chunk.
        count_tokens: Callable that returns token count for a string.

    Returns:
        List of Chunk objects representing the table pieces.
    """
    header = _strip_table_text(table.header)
    body_rows = [_strip_table_row(r) for r in table.body_rows]

    full_text = header
    if body_rows:
        full_text += "\n" + "\n".join(body_rows)

    full_token_count = count_tokens(full_text)

    # If the whole table fits, return as single chunk
    if full_token_count <= max_tokens:
        return [
            Chunk(
                text=full_text,
                start_index=table.start_index,
                end_index=table.end_index,
                token_count=full_token_count,
            )
        ]

    # Split by rows, prepending header to each chunk
    chunks: List[Chunk] = []
    current_rows: List[str] = []

    # Track char offset for body rows within the original text
    table_body_start_offset = table.start_index + len(table.header) + 1

    row_offsets: List[int] = []
    offset = table_body_start_offset
    for row in table.body_rows:
        row_offsets.append(offset)
        offset += len(row) + 1

    def flush_rows(rows: List[str], first_row_idx: int, end_index: int):
        """Helper to flush accumulated rows as a single chunk."""
        chunk_text = header + "\n" + "\n".join(rows)
        token_count = count_tokens(chunk_text)

        # For the first chunk, start_index is the table start (includes header)
        # For subsequent chunks, start_index is the first row's offset
        if not chunks:
            chunk_start = table.start_index
        else:
            chunk_start = row_offsets[first_row_idx]

        chunks.append(
            Chunk(
                text=chunk_text,
                start_index=chunk_start,
                end_index=min(end_index, table.end_index),
                token_count=token_count,
            )
        )

    for idx, row in enumerate(body_rows):
        current_rows.append(row)
        chunk_text = header + "\n" + "\n".join(current_rows)
        token_count = count_tokens(chunk_text)

        if len(current_rows) > 1 and token_count > max_tokens:
            current_rows.pop()
            flush_rows(
                current_rows,
                first_row_idx=idx - len(current_rows),
                end_index=row_offsets[idx],
            )
            current_rows = [row]

        # If a single row + header already exceeds max_tokens, flush it immediately
        if len(current_rows) == 1:
            token_count = count_tokens(header + "\n" + row)
            if token_count > max_tokens:
                logger.debug(
                    "Table row %d exceeds max_tokens (%d > %d) even alone with header.\n"
                    "Emitting oversized chunk.",
                    idx,
                    token_count,
                    max_tokens,
                )
                flush_rows(
                    current_rows,
                    first_row_idx=idx,
                    end_index=row_offsets[idx + 1]
                    if idx + 1 < len(row_offsets)
                    else table.end_index,
                )
                current_rows = []

    # Flush remaining rows
    if current_rows:
        flush_rows(
            current_rows,
            first_row_idx=len(body_rows) - len(current_rows),
            end_index=table.end_index,
        )

    return chunks


def chunk_table_single_row(
    table: TableRegion,
    count_tokens: Callable[[str], int],
) -> List[Chunk]:
    """Split a table into one chunk per body row, prepending the header to each.

    Args:
        table: The detected table region.
        count_tokens: Callable that returns token count for a string.

    Returns:
        List of Chunk objects, one per body row.
    """
    header = _strip_table_text(table.header)
    body_rows = [_strip_table_row(r) for r in table.body_rows]

    if not body_rows:
        token_count = count_tokens(header)
        return [
            Chunk(
                text=header,
                start_index=table.start_index,
                end_index=table.end_index,
                token_count=token_count,
            )
        ]

    chunks = []
    table_body_start_offset = table.start_index + len(table.header) + 1

    offset = table_body_start_offset
    for idx, (row, unstripped_row) in enumerate(zip(body_rows, table.body_rows)):
        chunk_text = header + "\n" + row
        token_count = count_tokens(chunk_text)

        # First chunk starts at table start, otherwise starts at given offset
        chunk_start = table.start_index if idx == 0 else offset

        row_end = offset + len(unstripped_row) + 1
        if row_end > table.end_index:
            row_end = table.end_index

        chunks.append(
            Chunk(
                text=chunk_text,
                start_index=chunk_start,
                end_index=row_end,
                token_count=token_count,
            )
        )
        offset += len(unstripped_row) + 1

    return chunks


def load_chonkie(chunking_strategy: str, chunking_args: Dict[str, Any]) -> BaseChunker:
    if chunking_strategy == "sentence":
        return SentenceChunker(**chunking_args)
    elif chunking_strategy == "semantic":
        return SemanticChunker(**chunking_args)
    elif chunking_strategy == "word":
        return WordChunker(**chunking_args)
    elif chunking_strategy == "token":
        return TokenChunker(**chunking_args)
    else:
        raise ValueError(f"Unsupported chunker: {chunking_strategy}")
