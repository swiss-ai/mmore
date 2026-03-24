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

# Regexes obtained from
# https://stackoverflow.com/questions/9837935/regex-for-markdown-table-syntax

# Matches a table row
_TABLE_ROW_RE = re.compile(r"^(?:\| *[^|\r\n]+ *)+\|$")

# Matches the delimiter row
_TABLE_SEPARATOR_RE = re.compile(r"^(?:\|[ :]?-+[ :]?)+\|$")


@dataclass
class TableRegion:
    """A detected markdown table region within a text."""

    start_index: int  # char offset in original text
    end_index: int  # char offset in original text (exclusive)
    header: str  # header row + separator row (prepended to sub-chunks)
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
    full_text = table.header
    if table.body_rows:
        full_text += "\n" + "\n".join(table.body_rows)

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
    header_tokens = count_tokens(table.header)
    chunks: List[Chunk] = []
    current_rows: List[str] = []
    current_token_count = header_tokens

    # Track char offset for body rows within the original text
    table_body_start_offset = table.start_index + len(table.header) + 1

    row_offsets: List[int] = []
    offset = table_body_start_offset
    for row in table.body_rows:
        row_offsets.append(offset)
        offset += len(row) + 1

    def flush_rows(
        rows: List[str], first_row_idx: int, end_index: int, token_count: int
    ):
        """Flush accumulated rows as a single chunk."""
        chunk_text = table.header + "\n" + "\n".join(rows)

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

    for idx, row in enumerate(table.body_rows):
        row_token_count = count_tokens(row)

        if current_rows and (current_token_count + row_token_count + 1) > max_tokens:
            flush_rows(
                current_rows,
                first_row_idx=idx - len(current_rows),
                end_index=row_offsets[idx],
                token_count=current_token_count,
            )
            current_rows = []
            current_token_count = header_tokens

        current_rows.append(row)
        current_token_count += row_token_count + 1

        # If a single row + header already exceeds max_tokens, flush it immediately
        if len(current_rows) == 1 and current_token_count > max_tokens:
            logger.warning(
                "Table row %d exceeds max_tokens (%d > %d) even alone with header.\n"
                "Emitting oversized chunk.",
                idx,
                current_token_count,
                max_tokens,
            )
            flush_rows(
                current_rows,
                first_row_idx=idx,
                end_index=row_offsets[idx],
                token_count=current_token_count,
            )
            current_rows = []
            current_token_count = header_tokens

    # Flush remaining rows
    if current_rows:
        flush_rows(
            current_rows,
            first_row_idx=len(table.body_rows) - len(current_rows),
            end_index=table.end_index,
            token_count=current_token_count,
        )

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
