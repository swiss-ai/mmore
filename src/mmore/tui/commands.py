"""Registry of mmore commands callable from the TUI.

Each entry mirrors a Click command in `mmore.cli` so the TUI is a thin wrapper:
the `run` callable is the same `run_*` function the CLI uses.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class CommandSpec:
    name: str
    description: str
    example_config: Optional[str]
    run: Callable[..., None]
    needs_input_data: bool = False
    config_globs: list[str] = field(default_factory=list)
    # Lazy importer returning the dataclass to validate YAML against.
    # Returns None if no validation is wired up for this stage.
    config_dataclass: Optional[Callable[[], Any]] = None


def _process(config_file: str, **_):
    from mmore.run_process import process

    process(config_file)


def _postprocess(config_file: str, input_data: str, **_):
    from mmore.run_postprocess import postprocess

    postprocess(config_file, input_data)


def _index(
    config_file: str,
    documents_path: Optional[str] = None,
    collection_name: Optional[str] = None,
    **_,
):
    from mmore.run_index import index

    index(config_file, documents_path, collection_name)


def _retrieve(config_file: str, **_):
    from mmore.run_retriever import run_api

    run_api(config_file, "0.0.0.0", 8001)


def _rag(config_file: str, **_):
    from mmore.run_rag import rag

    rag(config_file)


def _ragcli(config_file: str, **_):
    from mmore.run_ragcli import RagCLI

    RagCLI(config_file).launch_cli()


def _websearch(config_file: str, **_):
    from mmore.run_websearch import run_websearch

    run_websearch(config_file)


# Lazy dataclass importers — keeps heavy deps out of TUI startup.
def _dc_process():
    from mmore.run_process import ProcessInference

    return ProcessInference


def _dc_postprocess():
    from mmore.process.post_processor.pipeline import PPPipelineConfig

    return PPPipelineConfig


def _dc_index():
    from mmore.run_index import IndexConfig

    return IndexConfig


def _dc_rag():
    from mmore.run_rag import RAGInferenceConfig

    return RAGInferenceConfig


REGISTRY: dict[str, CommandSpec] = {
    "process": CommandSpec(
        name="process",
        description="Crawl + extract documents into a JSONL",
        example_config="examples/process/config.yaml",
        run=_process,
        config_globs=[
            "examples/process/**/*.yaml",
            "examples/process/**/*.yml",
        ],
        config_dataclass=_dc_process,
    ),
    "postprocess": CommandSpec(
        name="postprocess",
        description="Chunk / clean processed documents",
        example_config="examples/postprocessor/config.yaml",
        run=_postprocess,
        needs_input_data=True,
        config_globs=[
            "examples/postprocessor/**/*.yaml",
            "examples/postprocessor/**/*.yml",
        ],
        config_dataclass=_dc_postprocess,
    ),
    "index": CommandSpec(
        name="index",
        description="Embed + store documents in Milvus",
        example_config="examples/index/config.yaml",
        run=_index,
        config_globs=[
            "examples/index/**/*.yaml",
            "examples/index/**/*.yml",
        ],
        config_dataclass=_dc_index,
    ),
    "retrieve": CommandSpec(
        name="retrieve",
        description="Run retriever API server",
        example_config="examples/rag/config.yaml",
        run=_retrieve,
        config_globs=[
            "examples/rag/**/*.yaml",
            "examples/rag/**/*.yml",
        ],
        config_dataclass=_dc_rag,
    ),
    "rag": CommandSpec(
        name="rag",
        description="Run a one-shot RAG pipeline",
        example_config="examples/rag/config.yaml",
        run=_rag,
        config_globs=[
            "examples/rag/**/*.yaml",
            "examples/rag/**/*.yml",
        ],
        config_dataclass=_dc_rag,
    ),
    "ragcli": CommandSpec(
        name="ragcli",
        description="Interactive RAG chat",
        example_config="examples/rag/config.yaml",
        run=_ragcli,
        config_globs=[
            "examples/rag/**/*.yaml",
            "examples/rag/**/*.yml",
        ],
        config_dataclass=_dc_rag,
    ),
    "websearch": CommandSpec(
        name="websearch",
        description="Web search (+ optional RAG)",
        example_config="examples/websearchRAG/config.yaml",
        run=_websearch,
        config_globs=[
            "examples/websearchRAG/**/*.yaml",
            "examples/websearchRAG/**/*.yml",
        ],
    ),
}
