import re
from typing import Dict, List, Any, Optional, Literal
from dataclasses import dataclass, field
from src.mmore.type import MultimodalSample
from multiprocessing import Pool, cpu_count
from chonkie import Chunk, BaseChunker, SentenceChunker, SemanticChunker

from mmore.index.postprocessor.base import BasePostProcessor

import logging
logger = logging.getLogger(__name__)

# ---------------------------------- Config ---------------------------------- #

@dataclass
class ChunkerConfig:
    chunking_strategy: str = "sentence"
    chunker_args: Dict[str, Any] = field(default_factory=lambda: {})


def load_chonkie(chunker_name: str, chunking_args: Dict[str, Any]) -> BaseChunker:
    if chunker_name == 'sentence':
        return SentenceChunker(**chunking_args)
    elif chunker_name == 'semantic':
        return SemanticChunker(**chunking_args)
    else:
        raise ValueError(f'Unsupported chunker: {chunker_name}')

# ---------------------------------------------------------------------------- #

class MultimodalChunker(BasePostProcessor):
    text_chunker: BaseChunker

    def __init__(self, text_chunker: BaseChunker):
        super().__init__(name=f"MultimodalChunker-{text_chunker.name}")
        self.text_chunker = text_chunker

    @classmethod
    def from_config(cls, config: ChunkerConfig):
        text_chunker = load_chonkie(config.chunking_strategy, config.chunker_args)
        return cls(text_chunker=text_chunker)
    
    def process(self, sample: MultimodalSample, **kwargs) -> MultimodalSample | List[MultimodalSample]:
        return self.chunk(sample)

    @staticmethod
    def _chunk_modalities(sample: MultimodalSample, text_chunks: List[Chunk]):
        # Find all attachment
        attachment_indices = [m.start() for m in re.finditer(r'<attachment>', sample.text)]
        # Create an empty list to hold modalities for each chunk
        chunked_modalities = [[] for _ in range(len(text_chunks))]

        m = 0  # To track which modality to assign
        for idx in attachment_indices:
            if m >= len(sample.modalities) - 1:
                break
            chunk_index = _text_index_to_chunk_index(idx, text_chunks)
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
        # Chunk using the text chunker
        text_chunks = self.text_chunker.chunk(sample.text)

        # Chunk modalities according to the text chunks
        modalities_chunks = MultimodalChunker._chunk_modalities(sample, text_chunks)

        return [MultimodalSample(text=chunk.text, modalities=mods, metadata=sample.metadata) for chunk, mods in
                zip(text_chunks, modalities_chunks)]

    # def chunk_batch(self, batch: List[MultimodalSample]) -> List[List[MultimodalSample]]:
    #     """Split a List of samples into their respective chunks
    #     By default, this method uses multiprocessing to parallelize the chunking process.

    #     Args:
    #         batch: List of input samples to be chunked
        
    #     Returns:
    #         List of lists of Chunk objects containing the chunked text, modalities and metadata
    #     """
    #     workers = self.text_chunker._determine_optimal_workers()
    #     # if workers > 1:
    #     if False:
    #         with Pool(workers) as pool:
    #             return pool.map(self.chunk, batch)
    #     else:
    #         return [self.chunk(t) for t in batch]


def _text_index_to_chunk_index(index: int, chunks: List[Chunk]):
    for i, chunk in enumerate(chunks):
        if chunk.start_index <= index < chunk.end_index:
            return i
