from .base import BaseFilter, BaseFilterConfig

from .datatrove_wrapper import DatatroveFilter, DatatroveFilterConfig

from mmore.utils import load_config

__all__ = ['BaseFilter', 'DatatroveFilter']

def load_filter(config: BaseFilterConfig) -> BaseFilter:
    if config.type == 'datatrove':
        config = load_config(config.args, DatatroveFilterConfig)
        return DatatroveFilter.from_config(config)
    else:
        raise ValueError(f"Unrecognized postprocessor type: {config.type}")