import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, cast

from chonkie import BaseChunker, Chunk

from ....type import MultimodalSample
from .. import BasePostProcessor
from .utils import (
    TableRegion,
    _strip_table_row,
    _strip_table_text,
    chunk_table,
    chunk_table_single_row,
    detect_markdown_tables,
    load_chonkie,
)

logger = logging.getLogger(__name__)

# Good balance between retrieval precision and context preservation for RAG
_DEFAULT_CHUNK_SIZE = 512


class TableHandlingMode(str, Enum):
    """Valid modes when chunking tables."""

    SINGLE_ROW = "single_row"
    MULTI_ROWS = "multi_rows"
    KEEP_WHOLE = "keep_whole"
    NONE = "none"


@dataclass
class MultimodalChunkerConfig:
    chunking_strategy: str = "sentence"
    text_chunker_config: Dict[str, Any] = field(default_factory=dict)
    table_handling: str = "single_row"


class MultimodalChunker(BasePostProcessor):
    text_chunker: BaseChunker
    table_handling: TableHandlingMode

    def __init__(self, text_chunker: BaseChunker, table_handling: str = "single_row"):
        super().__init__("🦛 Chunker")
        self.text_chunker = text_chunker
        try:
            self.table_handling = TableHandlingMode(table_handling)
        except ValueError:
            raise ValueError(
                f"Invalid table_handling mode '{table_handling}'. "
                f"Must be one of: {[m.value for m in TableHandlingMode]}"
            )

    @classmethod
    def from_config(cls, config: MultimodalChunkerConfig):
        text_chunker = load_chonkie(
            config.chunking_strategy, config.text_chunker_config
        )
        return cls(
            text_chunker=text_chunker,
            table_handling=config.table_handling,
        )

    def process(self, sample: MultimodalSample, **kwargs) -> List[MultimodalSample]:
        return self.chunk(sample)

    @staticmethod
    def _chunk_modalities(sample: MultimodalSample, text_chunks: List[Chunk]):
        # Find all attachment
        attachment_indices = [
            m.start() for m in re.finditer(r"<attachment>", sample.text)
        ]
        # Create an empty list to hold modalities for each chunk
        chunked_modalities = [[] for _ in range(len(text_chunks))]

        m = 0  # To track which modality to assign
        for idx in attachment_indices:
            if m >= len(sample.modalities) - 1:
                break
            chunk_index = _text_index_to_chunk_index(idx, text_chunks)
            assert chunk_index is not None
            chunked_modalities[chunk_index].append(sample.modalities[m])
            m += 1

        return chunked_modalities

    def _count_tokens(self, text: str) -> int:
        """Count tokens using the text chunker's tokenizer."""
        if hasattr(self.text_chunker, "tokenizer") and hasattr(
            self.text_chunker.tokenizer, "count_tokens"
        ):
            return self.text_chunker.tokenizer.count_tokens(text)
        if hasattr(self.text_chunker, "tokenizer") and hasattr(
            self.text_chunker.tokenizer, "encode"
        ):
            return len(self.text_chunker.tokenizer.encode(text))
        raise RuntimeError(
            "Cannot count tokens: text chunker does not have a tokenizer with count_tokens() or encode()"
        )

    def _get_chunk_size(self) -> int:
        """Get the max chunk size from the text chunker."""
        if hasattr(self.text_chunker, "chunk_size"):
            return self.text_chunker.chunk_size
        return _DEFAULT_CHUNK_SIZE

    def _chunk_with_table_awareness(
        self, text: str, tables: Optional[List[TableRegion]] = None
    ) -> List[Chunk]:
        """Split text into chunks with special handling for markdown tables.

        Detects tables, splits text into table/non-table segments, applies
        the standard chunker to non-table segments, and applies table-specific
        chunking to table segments. Reassembles in document order.
        """
        if self.table_handling is TableHandlingMode.NONE:
            return self.text_chunker.chunk(text)

        if tables is None:
            tables = detect_markdown_tables(text)
        if not tables:
            return self.text_chunker.chunk(text)

        max_tokens = self._get_chunk_size()
        all_chunks: List[Chunk] = []

        # Build segments: alternating non-table and table regions
        prev_end = 0
        for table in tables:
            # Non-table segment before this table
            if table.start_index > prev_end:
                non_table_text = text[prev_end : table.start_index]
                if non_table_text.strip():
                    segment_chunks = self.text_chunker.chunk(non_table_text)
                    # Adjust start/end indices to original text positions
                    for chunk in segment_chunks:
                        chunk.start_index += prev_end
                        chunk.end_index += prev_end
                    all_chunks.extend(segment_chunks)

            # Table segment
            if self.table_handling is TableHandlingMode.KEEP_WHOLE:
                full_text = _strip_table_text(table.header)
                if table.body_rows:
                    full_text += "\n" + "\n".join(
                        _strip_table_row(r) for r in table.body_rows
                    )
                token_count = self._count_tokens(full_text)
                all_chunks.append(
                    Chunk(
                        text=full_text,
                        start_index=table.start_index,
                        end_index=table.end_index,
                        token_count=token_count,
                    )
                )
            elif self.table_handling is TableHandlingMode.MULTI_ROWS:
                table_chunks = chunk_table(table, max_tokens, self._count_tokens)
                all_chunks.extend(table_chunks)
            elif self.table_handling is TableHandlingMode.SINGLE_ROW:
                table_chunks = chunk_table_single_row(table, self._count_tokens)
                all_chunks.extend(table_chunks)

            prev_end = table.end_index

        # Remaining non-table text after the last table
        if prev_end < len(text):
            remaining_text = text[prev_end:]
            if remaining_text.strip():
                segment_chunks = self.text_chunker.chunk(remaining_text)
                for chunk in segment_chunks:
                    chunk.start_index += prev_end
                    chunk.end_index += prev_end
                all_chunks.extend(segment_chunks)

        return all_chunks

    def _is_table_chunk(
        self, chunk: Chunk, tables: List[TableRegion]
    ) -> Optional[TableRegion]:
        """Check if a chunk overlaps with a detected table region."""
        for table in tables:
            if (
                table.start_index <= chunk.start_index
                and chunk.end_index <= table.end_index
            ):
                return table
        return None

    def chunk(self, sample: MultimodalSample) -> List[MultimodalSample]:
        """Split sample into chunks according to the implementation strategy.

        Args:
            sample: Input sample to be chunked

        Returns:
            List of Chunk objects containing the chunked text and metadata
        """
        if not sample.text or not sample.text.strip():
            logger.warning(f"Empty text in sample {sample.id}. Skipping chunking.")
            return []

        # Detect tables before chunking
        tables = []
        if self.table_handling is not TableHandlingMode.NONE:
            tables = detect_markdown_tables(sample.text)

        try:
            text_chunks = self._chunk_with_table_awareness(sample.text, tables)
        except Exception as e:
            logger.error(
                f"Chunking error on sample with length {len(sample.text)}: {e}"
            )
            return []

        # Chunk modalities according to the text chunks
        modalities_chunks = MultimodalChunker._chunk_modalities(sample, text_chunks)
        para_info_chunks = self._assign_paragraph_positions(sample, text_chunks)

        chunks = []
        for i, (chunk, mods, para_info) in enumerate(
            zip(text_chunks, modalities_chunks, para_info_chunks)
        ):
            chunk_metadata = sample.metadata.copy()
            chunk_metadata.update(para_info)
            chunk_metadata.pop("paragraph_starts", None)

            # Add table metadata if this chunk comes from a table
            table = self._is_table_chunk(chunk, tables)
            if table is not None:
                chunk_metadata["is_table_chunk"] = True
                chunk_metadata["table_header"] = _strip_table_text(table.header)

            s = MultimodalSample(
                text=chunk.text,
                modalities=mods,
                metadata=chunk_metadata,
                id=f"{sample.id}+{i}",
            )
            chunks.append(s)

        return chunks

    def _assign_paragraph_positions(
        self, sample: MultimodalSample, text_chunks: List[Chunk]
    ) -> List[Dict[str, Any]]:
        """Assign paragraph numbers (per-page) using paragraph start positions."""
        para_info_chunks: List[Dict[str, Any]] = []
        paragraph_starts = cast(
            List[Tuple[int, int, int]],
            sample.metadata.get("paragraph_starts", []),
        )

        if len(paragraph_starts) == 0:
            for chunk in text_chunks:
                para_info_chunks.append({})
            return para_info_chunks

        for chunk in text_chunks:
            chunk_paragraphs = []

            for i in range(len(paragraph_starts) - 1):
                para_start, page_num, para_idx = paragraph_starts[i]
                next_start, _, _ = paragraph_starts[i + 1]
                if chunk.start_index < next_start and chunk.end_index > para_start:
                    chunk_paragraphs.append([page_num, para_idx])

            para_info_chunks.append({"paragraph_positions": chunk_paragraphs})

        return para_info_chunks


def _text_index_to_chunk_index(index: int, chunks: List[Chunk]) -> Optional[int]:
    for i, chunk in enumerate(chunks):
        if chunk.start_index <= index < chunk.end_index:
            return i
