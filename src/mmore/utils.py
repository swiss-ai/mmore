from typing import Dict, Type, TypeVar, Union, cast

import yaml
from dacite import from_dict

T = TypeVar("T")


def load_config(yaml_dict_or_path: Union[str, Dict, T], config_class: Type[T]) -> T:
    if isinstance(yaml_dict_or_path, config_class):
        return yaml_dict_or_path

    if isinstance(yaml_dict_or_path, str):
        with open(yaml_dict_or_path, "r") as file:
            data = yaml.safe_load(file)
    else:
        data = yaml_dict_or_path

    return from_dict(config_class, cast(Dict, data))
