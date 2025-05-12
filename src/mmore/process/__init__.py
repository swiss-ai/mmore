import click

from .config import get as get_config
from .config import set as set_config


@click.group()
def main():
    """mmore"""
    pass


# main.add_command(test)
@main.group()
def config():
    """mmore config"""
    pass


@config.command()
@click.argument("key_path", required=False)
def get(key_path):
    value = get_config(key_path)
    if value is not None:
        if key_path:
            click.echo(f"{key_path}: {value}")
        else:
            click.echo(f"{value}")


@config.command()
@click.argument("key_path")
@click.argument("value")
def set(key_path, value):
    """Set a value in the config based on the provided key path."""
    config_path = set_config(key_path, value)
    if config_path:
        click.echo(f"Saved config {key_path} with value {value} to: {config_path}")


if __name__ == "__main__":
    main()
