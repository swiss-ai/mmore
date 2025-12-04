import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from chonkie import BaseChunker, Chunk

from ....type import MultimodalSample
from .. import BasePostProcessor
from .utils import load_chonkie

logger = logging.getLogger(__name__)


@dataclass
class MultimodalChunkerConfig:
    chunking_strategy: str = "sentence"
    text_chunker_config: Dict[str, Any] = field(default_factory=dict)


class MultimodalChunker(BasePostProcessor):
    text_chunker: BaseChunker

    def __init__(self, text_chunker: BaseChunker):
        super().__init__("🦛 Chunker")
        self.text_chunker = text_chunker

    @classmethod
    def from_config(cls, config: MultimodalChunkerConfig):
        text_chunker = load_chonkie(
            config.chunking_strategy, config.text_chunker_config
        )
        return cls(text_chunker=text_chunker)

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

        try:
            # Chunk using the text chunker
            text_chunks = self.text_chunker.chunk(sample.text)
        except Exception as e:
            logger.error(
                f"Chunking error on sample with length: {len(sample.text): {e}} "
            )
            return []

        # Chunk modalities according to the text chunks
        modalities_chunks = MultimodalChunker._chunk_modalities(sample, text_chunks)
        page_info_chunks = self._assign_page_numbers(sample, text_chunks)

        chunks = []
        for i, (chunk, mods, page_info) in enumerate(
            zip(text_chunks, modalities_chunks, page_info_chunks)
        ):
            chunk_metadata = sample.metadata.copy()
            chunk_metadata.update(page_info)
            chunk_metadata.pop("page_starts", None)

            s = MultimodalSample(
                text=chunk.text,
                modalities=mods,
                metadata=chunk_metadata,
                id=f"{sample.id}+{i}",
            )
            chunks.append(s)

        return chunks

    def _assign_page_numbers(
        self, sample: MultimodalSample, text_chunks: List[Chunk]
    ) -> List[Dict[str, Any]]:
        """Assign page numbers using page start positions."""
        page_info_chunks = []
        page_starts = sample.metadata.get("page_starts", [])

        if len(page_starts) == 0:
            for chunk in text_chunks:
                page_info_chunks.append({})
            return page_info_chunks

        for chunk in text_chunks:
            chunk_page_numbers = set()

            for i in range(len(page_starts) - 1):
                page_start, page_num = page_starts[i]
                next_start, _ = page_starts[i + 1]
                if chunk.start_index < next_start and chunk.end_index > page_start:
                    chunk_page_numbers.add(page_num)

            sorted_pages = sorted(chunk_page_numbers)
            page_info_chunks.append({"page_numbers": sorted_pages})

        return page_info_chunks


def _text_index_to_chunk_index(index: int, chunks: List[Chunk]) -> Optional[int]:
    for i, chunk in enumerate(chunks):
        if chunk.start_index <= index < chunk.end_index:
            return i
