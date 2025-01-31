from typing import List, Dict
import yaml

from dacite import Config, from_dict

def load_config(yaml_dict_or_path: str | Dict, config_class: Config) -> Config:
    if isinstance(yaml_dict_or_path, str):
        with open(yaml_dict_or_path, 'r') as file:
            data = yaml.safe_load(file)
    else:
        data = yaml_dict_or_path
    return from_dict(config_class, data)
