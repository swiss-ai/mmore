from typing import List
from dacite import Config, from_dict
import yaml


def load_config(yaml_path: str, config_class: Config) -> Config:
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
        return from_dict(config_class, data)
