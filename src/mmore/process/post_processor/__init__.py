from typing import get_args

from .base import BasePostProcessor, BasePostProcessorConfig

from .chunker import MultimodalChunker, MultimodalChunkerConfig
from .ner import NERecognizer, NERExtractorConfig
from .tagger import TAGGER_TYPES, load_tagger, ModalitiesCounter, WordsCounter
from .filter import FILTER_TYPES, load_filter, BaseFilter

from mmore.utils import load_config

__all__ = ['BasePostProcessor', 'MultimodalChunker', 'LangDetector', 'ModalitiesCounter', 
           'WordsCounter', 'NERecognizer', 'BaseFilter']

def load_postprocessor(config: BasePostProcessorConfig) -> BasePostProcessor:
    if config.type in get_args(FILTER_TYPES):
        return load_filter(config)
    elif config.type in get_args(TAGGER_TYPES):
        return load_tagger(config)
    elif config.type == 'chunker':
        config = load_config(config.args, MultimodalChunkerConfig)
        return MultimodalChunker.from_config(config)
    elif config.type == 'ner':
        config = load_config(config.args, NERExtractorConfig)
        return NERecognizer.from_config(config)
    else:
        raise ValueError(f"Unrecognized postprocessor type: {config.type}")