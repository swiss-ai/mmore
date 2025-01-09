from typing import get_args

from .base import BasePostProcessor, BasePostProcessorConfig

from .chunker import MultimodalChunker, MultimodalChunkerConfig
from .ner import NERecognizer, NERExtractorConfig
from .tagger import TAGGER_TYPES, load_tagger
from .filter import FILTER_TYPES, load_filter

from mmore.utils import load_config

__all__ = ['BasePostProcessor', 'BasePostProcessorConfig', 'load_postprocessor']

def load_postprocessor(config: BasePostProcessorConfig) -> BasePostProcessor:
    if config.type in FILTER_TYPES:
        return load_filter(config)
    elif config.type in TAGGER_TYPES:
        return load_tagger(config)
    elif config.type == 'chunker':
        config = load_config(config.args, MultimodalChunkerConfig)
        return MultimodalChunker.from_config(config)
    elif config.type == 'ner':
        config = load_config(config.args, NERExtractorConfig)
        return NERecognizer.from_config(config)
    else:
        raise ValueError(f"Unrecognized postprocessor type: {config.type}")