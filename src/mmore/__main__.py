# __main__.py

import click
import yaml

@click.group()
def main():
    """CLI for mmore commands."""
    pass


@main.command()
@click.option(
    "--config-file", type=str, required=True, help="Dispatcher configuration file path."
)
def process(config_file):
    """Process documents from a directory."""
    from .run_process import process as run_process

    run_process(config_file)


@main.command()
@click.option(
    "--config-file",
    type=str,
    required=True,
    help="Path to the postprocess configuration file.",
)
@click.option(
    "--input-data", type=str, required=True, help="Path to the jsonl of the documents."
)
def postprocess(config_file, input_data):
    """Run the post-processors pipeline."""
    from .run_postprocess import postprocess as run_postprocess

    run_postprocess(config_file, input_data)


@main.command()
@click.option(
    "--config-file",
    "-c",
    type=str,
    required=True,
    help="Path to the configuration file.",
)
@click.option(
    "--documents-path", "-f", type=str, required=False, help="Path to the JSONL data."
)
@click.option(
    "--collection-name",
    "-n",
    type=str,
    required=False,
    help="Name of the collection to index.",
)
def index(config_file, documents_path, collection_name):
    """Run the indexer."""
    from .run_index import index as run_index

    run_index(config_file, documents_path, collection_name)


@main.command()
@click.option(
    "--config-file",
    "-c",
    type=str,
    required=True,
    help="Dispatcher configuration file path.",
)
@click.option(
    "--input-file", "-f", type=str, required=True, help="Path to the input file."
)
@click.option(
    "--output-file", "-o", type=str, required=True, help="Path to the output file."
)
def retrieve(config_file, input_file, output_file):
    """Retrieve documents for specified queries."""
    from .run_retriever import retrieve as run_retrieve

    run_retrieve(config_file, input_file, output_file)


@main.command()
@click.option(
    "--config-file", type=str, required=True, help="Dispatcher configuration file path."
)
def rag(config_file):
    """Run the Retrieval-Augmented Generation (RAG) pipeline."""
    from .run_rag import rag as run_rag

    run_rag(config_file)


@main.command()
@click.option(
    "--config-file",
    type=str,
    required=True,
    help="Path to the Websearch configuration file (YAML).",
)
def websearch(config_file):
    """Run the Websearch (+ optional RAG) pipeline."""
    from .run_websearch import run_websearch

    # Load your YAML and pass it into the runner
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)

    run_websearch(config_dict)


if __name__ == "__main__":
    main()
