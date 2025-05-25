import os
from pathlib import Path

import click
import yaml

default_config = {
    "processor": {
        "MediaProcessor": [
            {"normal_model": "openai/whisper-large-v3-turbo"},
            {"fast_model": "openai/whisper-tiny"},
        ],
        "PDFProcessor": [
            {"PDFTEXT_CPU_WORKERS": 0},  # We use cpu_count()
            {"DETECTOR_BATCH_SIZE": 120},
            {"DETECTOR_POSTPROCESSING_CPU_WORKERS": 0},  # We use cpu_count()
            {"RECOGNITION_BATCH_SIZE": 64},
            {"OCR_PARALLEL_WORKERS": 0},  # We use cpu_count()
            {"TEXIFY_BATCH_SIZE": 120},
            {"LAYOUT_BATCH_SIZE": 120},
            {"ORDER_BATCH_SIZE": 90},
            {"TABLE_REC_BATCH_SIZE": 120},
        ],
    },
    "dispatcher": {
        "node_batch_sizes": [
            {"URLProcessor": 40},
            {"DOCXProcessor": 100},
            {"PDFProcessor": 3000},
            {"MediaProcessor": 40},
            {"SpreadsheetProcessor": 100},
            {"TXTProcessor": 100},
            {"PPTXProcessor": 100},
            {"MarkdownProcessor": 100},
            {"EMLProcessor": 100},
            {"HTMLProcessor": 100},
        ]
    },
}


def get_config_path():
    """Get the path to the YAML config file."""
    config_path = os.getenv("MMORE_CONFIG")
    if config_path:
        return Path(config_path)

    app_dir = Path(click.get_app_dir("mmore"))
    try:
        app_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        click.echo(f"Error creating config directory: {e}", err=True)
        raise
    return app_dir / "config.yaml"


def load_config():
    """Load the data store from the YAML file."""
    file_path = get_config_path()
    if file_path.exists():
        try:
            with file_path.open("r") as file:
                return yaml.safe_load(file) or default_config
        except yaml.YAMLError as e:
            click.echo(f"Error loading config file: {e}", err=True)
            return default_config
    return default_config


def save_config(data):
    """Save the data store to the YAML file."""
    file_path = get_config_path()
    try:
        with file_path.open("w") as file:
            yaml.safe_dump(data, file)
    except Exception as e:
        click.echo(f"Error saving config file: {e}", err=True)


def set_nested_value(d, key_path, value):
    keys = key_path.split(".")
    current = d
    for key in keys[:-1]:
        if key not in current:
            click.echo(f"Warning: Key '{key}' does not exist in the config.", err=True)
            return False
        current = current[key]
    if keys[-1] not in current:
        click.echo(f"Warning: Key '{keys[-1]}' does not exist in the config.", err=True)
        return False
    current[keys[-1]] = value
    return True


def get_nested_value(d, key_path):
    if not key_path:
        return d
    keys = key_path.split(".")
    current = d
    for key in keys:
        if key not in current:
            click.echo(f"Error: Key '{key}' does not exist in the config.", err=True)
            return None
        current = current[key]
    return current


def get(key_path=None):
    """Get a value from the config based on the provided key path."""
    config = load_config()
    value = get_nested_value(config, key_path)
    return value


def set(key_path, value):
    """Set a value in the config based on the provided key path."""
    config = load_config()
    if set_nested_value(config, key_path, value):
        save_config(config)

    return get_config_path()
