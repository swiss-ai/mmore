from typing import Dict, List, Any, Optional, Literal

from chonkie import BaseChunker, SentenceChunker, SemanticChunker


def load_chonkie(chunker_name: str, chunking_args: Dict[str, Any]) -> BaseChunker:
    if chunker_name == 'sentence':
        return SentenceChunker(**chunking_args)
    elif chunker_name == 'semantic':
        return SemanticChunker(**chunking_args)
    else:
        raise ValueError(f'Unsupported chunker: {chunker_name}')
