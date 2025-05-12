from .base import BaseFilter, BaseFilterConfig
from .datatrove_wrapper import DATATROVE_FILTERS, DatatroveFilter

__all__ = ["BaseFilter", "DatatroveFilter"]

DATATROVE_MAP = {c: DatatroveFilter for c in DATATROVE_FILTERS}

FILTERS_LOADERS_MAP = {**DATATROVE_MAP}
FILTER_TYPES = list(FILTERS_LOADERS_MAP.keys())


def load_filter(config: BaseFilterConfig) -> BaseFilter:
    if config.type in FILTERS_LOADERS_MAP:
        return FILTERS_LOADERS_MAP[config.type].from_config(config)
    else:
        raise ValueError(f"Unrecognized filter type: {config.type}")
