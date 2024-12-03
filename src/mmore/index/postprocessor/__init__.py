from .base import BasePostProcessor, BasePostProcessorConfig

from .chunker import MultimodalChunker, ChunkerConfig
from .lang_detector import LangDetector
from .counter import ModalitiesCounter, WordsCounter

__all__ = ['BasePostProcessor', 'MultimodalChunker', 'LangDetector', 'ModalitiesCounter', 'WordsCounter']

def load_postprocessor(config: BasePostProcessorConfig) -> BasePostProcessor:
    if config.type == 'chunker':
        config = ChunkerConfig(**config.args)
        return MultimodalChunker.from_config(config)
    elif config.type == 'lang_detector':
        return LangDetector()
    elif config.type == 'modalities_counter':
        return ModalitiesCounter()
    elif config.type == 'words_counter':    
        return WordsCounter()
    else:
        raise ValueError(f"Unrecognized postprocessor type: {config.type}")
