from .base import BaseTaggerConfig, TAGGER_TYPES

from .modalities import ModalitiesCounter
from .words import WordsCounter
from .lang_detector import LangDetector

from mmore.utils import load_config

__all__ = ['ModalitiesCounter', 'WordsCounter', 'LangDetector']

def load_tagger(config: BaseTaggerConfig):
    if config.type == 'modalities_counter':
        return ModalitiesCounter()
    elif config.type == 'words_counter':
        return WordsCounter()
    elif config.type == 'lang_detector':
        return LangDetector()
    else:
        raise ValueError(f"Unrecognized tagger type: {config.type}")