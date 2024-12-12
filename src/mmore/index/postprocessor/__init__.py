from .base import BasePostProcessor, BasePostProcessorConfig

from .chunker import MultimodalChunker, ChunkerConfig
from .lang_detector import LangDetector
from .counter import ModalitiesCounter, WordsCounter
from .ner import NERecognizer, NERExtractorConfig

from mmore.utils import load_config

__all__ = ['BasePostProcessor', 'MultimodalChunker', 'LangDetector', 'ModalitiesCounter', 'WordsCounter', 'NERecognizer']

def load_postprocessor(config: BasePostProcessorConfig) -> BasePostProcessor:
    if config.type == 'chunker':
        config = load_config(config.args, ChunkerConfig)
        return MultimodalChunker.from_config(config)
    elif config.type == 'ner':
        config = load_config(config.args, NERExtractorConfig)
        return NERecognizer.from_config(config)
    elif config.type == 'lang_detector':
        return LangDetector()
    elif config.type == 'modalities_counter':
        return ModalitiesCounter()
    elif config.type == 'words_counter':    
        return WordsCounter()
    else:
        raise ValueError(f"Unrecognized postprocessor type: {config.type}")
