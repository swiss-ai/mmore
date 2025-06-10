from typing import Dict, Type, TypeVar, Union, cast, Any
from dataclasses import asdict, is_dataclass
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

# Custom Dumper to preserve \n and avoid wrapping
class LiteralStringDumper(yaml.SafeDumper):
    pass

def str_presenter(dumper, data):
    if "\n" in data or "\r" in data:
        # Force double-quoted string with literal escapes
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)

LiteralStringDumper.add_representer(str, str_presenter)

def save_config(config: Any, path: str) -> None:
    if not is_dataclass(config):
        raise ValueError("Provided config is not a dataclass instance.")

    with open(path, "w") as file:
        yaml.dump(
            asdict(config),
            file,
            Dumper=LiteralStringDumper,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
            width=float("inf"),  # prevents line wrapping
        )