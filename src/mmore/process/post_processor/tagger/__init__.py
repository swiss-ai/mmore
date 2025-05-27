from .base import BaseTaggerConfig
from .lang_detector import LangDetector
from .modalities import ModalitiesCounter
from .words import WordsCounter

__all__ = ["ModalitiesCounter", "WordsCounter", "LangDetector"]

TAGGERS_LOADERS_MAP = {
    "modalities_counter": ModalitiesCounter,
    "words_counter": WordsCounter,
    "lang_detector": LangDetector,
}
TAGGER_TYPES = list(TAGGERS_LOADERS_MAP.keys())


def load_tagger(config: BaseTaggerConfig):
    if config.type in TAGGER_TYPES:
        return TAGGERS_LOADERS_MAP[config.type].from_config(config)
    else:
        raise ValueError(f"Unrecognized tagger type: {config.type}")
