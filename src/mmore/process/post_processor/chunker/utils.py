from typing import Any, Dict

from chonkie import (
    BaseChunker,
    SemanticChunker,
    SentenceChunker,
    TokenChunker,
    WordChunker,
)


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
