import click


@click.group()
def main():
    """CLI for mmore commands."""
    pass


@main.command()
@click.option(
    "--config-file", type=str, required=True, help="Dispatcher configuration file path."
)
def process(config_file: str):
    """Process documents from a directory.

    Args:
      config_file: Dispatcher configuration file path.

    Returns:

    """
    from .run_process import process as run_process

    run_process(config_file)


@main.command()
@click.option(
    "--config-file",
    type=str,
    required=True,
    help="Path to the config file for post-processing.",
)
@click.option(
    "--input-data",
    type=str,
    required=True,
    help="Path to the input JSONL file of documents.",
)
def postprocess(config_file: str, input_data: str):
    """Run the post-processors pipeline.

    Args:
      config_file: path to the config file for post-processing.
      input_data: path to the input JSONL file of documents.

    Returns:

    """
    from .run_postprocess import postprocess as run_postprocess

    run_postprocess(config_file, input_data)


@main.command()
@click.option(
    "--config-file",
    "-c",
    type=str,
    required=True,
    help="Path to the config file for indexing.",
)
@click.option(
    "--documents-path",
    "-f",
    type=str,
    required=False,
    help="Path to the JSONL file of the (post)processed documents.",
)
@click.option(
    "--collection-name",
    "-n",
    type=str,
    required=False,
    help="Name that will be used to refer to this collection of documents.",
)
def index(config_file: str, documents_path: str, collection_name: str):
    """Run the indexer.

    Args:
      config_file: path to the config file for indexing.
      documents_path: path to the JSONL file of the (post)processed documents.
      collection_name: name that will be used to refer to this collection of documents.

    Returns:

    """
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
    "--input-file",
    "-f",
    type=str,
    required=True,
    help="Path to the JSONL file of the input queries.",
)
@click.option(
    "--output-file",
    "-o",
    type=str,
    required=True,
    help="Path to which save the results of the retriever as a JSON.",
)
def retrieve(config_file: str, input_file: str, output_file: str):
    """Retrieve documents for specified queries.

    Args:
      config_file: path to the config file for the retriver.
      input_file: path to the JSONL file of the input queries.
      output_file: path to which save the results of the retriever as a JSON.

    Returns:

    """
    from .run_retriever import retrieve as run_retrieve

    run_retrieve(config_file, input_file, output_file)


@main.command()
@click.option(
    "--config-file", type=str, required=True, help="Dispatcher configuration file path."
)
def rag(config_file: str):
    """Run the Retrieval-Augmented Generation (RAG) pipeline.

    Args:
      config_file: Dispatcher configuration file path.

    Returns:

    """
    from .run_rag import rag as run_rag

    run_rag(config_file)


@main.command()
@click.option(
    "--host", type=str, default="0.0.0.0", help="Host on which the API should be run."
)
@click.option(
    "--port", type=int, default=8000, help="Port on which the API should be run."
)
def index_api(host, port):
    """Run the Index API.

    Args:
      host: Host on which the API should be run.
      port: Port on which the API should be run.

    Returns:

    """
    from .run_index_api import run_api

    run_api(host, port)


@main.command()
@click.option(
    "--host",
    type=str,
    default="0.0.0.0",
    help="Host on which the dashboard API should be run.",
)
@click.option(
    "--port",
    type=int,
    default=8000,
    help="Port on which the dashboard API should be run.",
)
def dashboard_backend(host, port):
    """Run the dashboard backend.

    Args:
      host:
      port:

    Returns:

    """
    from .run_dashboard_backend import run_api

    run_api(host, port)


if __name__ == "__main__":
    main()
