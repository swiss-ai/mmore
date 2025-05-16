from typing import cast

from ...utils import load_config
from .base import BasePostProcessor, BasePostProcessorConfig
from .chunker import MultimodalChunker, MultimodalChunkerConfig
from .filter import FILTER_TYPES, load_filter
from .filter.base import BaseFilterConfig
from .ner import NERecognizer, NERExtractorConfig
from .tagger import TAGGER_TYPES, load_tagger
from .tagger.base import BaseTaggerConfig

__all__ = ["BasePostProcessor", "BasePostProcessorConfig", "load_postprocessor"]


def load_postprocessor(config: BasePostProcessorConfig) -> BasePostProcessor:
    if config.type in FILTER_TYPES:
        return load_filter(cast(BaseFilterConfig, config))
    elif config.type in TAGGER_TYPES:
        return load_tagger(cast(BaseTaggerConfig, config))
    elif config.type == "chunker":
        config_chunk = load_config(config.args, MultimodalChunkerConfig)
        return MultimodalChunker.from_config(config_chunk)
    elif config.type == "ner":
        config_ner = load_config(config.args, NERExtractorConfig)
        return NERecognizer.from_config(config_ner)
    else:
        raise ValueError(f"Unrecognized postprocessor type: {config.type}")
