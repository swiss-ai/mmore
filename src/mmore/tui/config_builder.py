"""Generate YAML config files via guided prompts.

Templates here mirror the example configs under `examples/`. The user is
asked only for the fields most likely to change between runs; everything else
falls back to the example defaults. The resulting dict is dumped to a YAML
file under `./tui-configs/`.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Optional

import questionary
import yaml
from questionary import Style
from rich.panel import Panel
from rich.text import Text

from mmore.tui.commands import CommandSpec

CONFIG_DIR = Path("./tui-configs")

QSTYLE = Style([
    ("qmark", "fg:#5fd7ff bold"),
    ("question", "bold"),
    ("answer", "fg:#ff5fd7 bold"),
    ("pointer", "fg:#5fd7ff bold"),
    ("highlighted", "fg:#5fd7ff bold"),
    ("selected", "fg:#ff5fd7"),
    ("instruction", "fg:#808080 italic"),
])
QMARK = "▸"


def _prompt(question: str, default: str = "") -> str:
    answer = questionary.text(question, default=default, style=QSTYLE, qmark=QMARK).ask()
    if answer is None:
        raise KeyboardInterrupt
    return answer


def _confirm(question: str, default: bool = False) -> bool:
    answer = questionary.confirm(question, default=default, style=QSTYLE, qmark=QMARK).ask()
    if answer is None:
        raise KeyboardInterrupt
    return answer


def _save(name: str, data: dict[str, Any]) -> str:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIG_DIR / f"{name}-{int(time.time())}.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return str(path)


def build_process_config() -> str:
    data_path = _prompt("Data path (folder with documents to process)", "examples/sample_data/")
    output_path = _prompt("Output path (where merged_results.jsonl will be written)",
                          "examples/process/outputs/")
    use_fast = _confirm("Use fast (lower-quality) processors?", default=False)
    distributed = _confirm("Use distributed processing (Dask)?", default=False)
    extract_images = _confirm("Extract images from documents?", default=True)

    cfg = {
        "data_path": data_path,
        "google_drive_ids": [],
        "previous_results": None,
        "dispatcher_config": {
            "output_path": output_path,
            "use_fast_processors": use_fast,
            "distributed": distributed,
            "extract_images": extract_images,
            "scheduler_file": None,
            "process_batch_sizes": [
                {"URLProcessor": 40},
                {"DOCXProcessor": 100},
                {"PDFProcessor": 4000},
                {"MediaProcessor": 40},
                {"SpreadsheetProcessor": 100},
                {"TXTProcessor": 100},
                {"PPTXProcessor": 100},
                {"MarkdownProcessor": 100},
                {"EMLProcessor": 100},
                {"HTMLProcessor": 100},
            ],
            "processor_config": {
                "MediaProcessor": [
                    {"normal_model": "openai/whisper-large-v3-turbo"},
                    {"fast_model": "openai/whisper-tiny"},
                    {"type": "automatic-speech-recognition"},
                    {"sample_rate": 10},
                    {"batch_size": 4},
                ],
                "PDFProcessor": [
                    {"PDFTEXT_CPU_WORKERS": 0},
                    {"DETECTOR_BATCH_SIZE": 1},
                    {"DETECTOR_POSTPROCESSING_CPU_WORKERS": 0},
                    {"RECOGNITION_BATCH_SIZE": 1},
                    {"OCR_PARALLEL_WORKERS": 0},
                    {"TEXIFY_BATCH_SIZE": 1},
                    {"LAYOUT_BATCH_SIZE": 1},
                    {"ORDER_BATCH_SIZE": 1},
                    {"TABLE_REC_BATCH_SIZE": 1},
                ],
            },
        },
    }
    return _save("process", cfg)


def build_postprocess_config() -> str:
    strategy = questionary.select(
        "Chunking strategy",
        choices=["sentence", "token", "word", "semantic"],
        default="sentence",
        style=QSTYLE, qmark=QMARK,
    ).ask()
    if strategy is None:
        raise KeyboardInterrupt
    table_handling = questionary.select(
        "Table handling",
        choices=["single_row", "multi_rows", "keep_whole", "none"],
        default="single_row",
        style=QSTYLE, qmark=QMARK,
    ).ask()
    if table_handling is None:
        raise KeyboardInterrupt
    output_path = _prompt("Output JSONL path",
                          "examples/postprocessor/outputs/merged/results.jsonl")

    cfg = {
        "previous_results": None,
        "pp_modules": [
            {"type": "chunker", "args": {
                "chunking_strategy": strategy,
                "table_handling": table_handling,
            }},
        ],
        "output": {"output_path": output_path, "save_each_step": True},
    }
    return _save("postprocess", cfg)


def build_index_config(documents_path: Optional[str] = None) -> str:
    dense = _prompt("Dense embedding model",
                    "sentence-transformers/all-MiniLM-L6-v2")
    sparse = _prompt("Sparse embedding model", "splade")
    db_uri = _prompt("DB URI (Milvus Lite file or server URL)", "./proc_demo.db")
    db_name = _prompt("DB name", "my_db")
    collection = _prompt("Collection name", "my_docs")
    docs = documents_path or _prompt(
        "Documents JSONL path",
        "examples/postprocessor/outputs/merged/results.jsonl",
    )
    cfg = {
        "indexer": {
            "dense_model": {"model_name": dense, "is_multimodal": False},
            "sparse_model": {"model_name": sparse, "is_multimodal": False},
            "db": {"uri": db_uri, "name": db_name},
        },
        "collection_name": collection,
        "documents_path": docs,
    }
    return _save("index", cfg)


BUILDERS = {
    "process": build_process_config,
    "postprocess": build_postprocess_config,
    "index": build_index_config,
}


def find_yaml_configs(spec: CommandSpec, root: str = ".") -> list[str]:
    """Find candidate YAML configs scoped to this stage.

    Includes:
    - files matching any of `spec.config_globs`
    - previously-generated `tui-configs/<stage>-*.yaml`
    """
    root_path = Path(root)
    matches: list[str] = []
    for pattern in spec.config_globs:
        for p in root_path.glob(pattern):
            matches.append(str(p))
    # Generated configs from previous TUI runs
    generated = root_path / "tui-configs"
    if generated.exists():
        for p in sorted(generated.glob(f"{spec.name}-*.yaml")):
            matches.append(str(p))

    seen: set[str] = set()
    out: list[str] = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _validate_yaml(path: str, spec: CommandSpec) -> Optional[str]:
    """Return None on success, an error message string on failure."""
    if spec.config_dataclass is None:
        return None
    try:
        from mmore.utils import load_config
        dataclass_cls = spec.config_dataclass()
        load_config(path, dataclass_cls)
        return None
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"


def _show_error_panel(path: str, err: str) -> None:
    from mmore.tui.theme import console
    console.print(Panel(
        Text.assemble(
            (f"{path}\n\n", "bold"),
            (err, "red"),
        ),
        title="[bold red]invalid config[/]",
        border_style="red",
        padding=(1, 2),
    ))


def _ranked_choices(spec: CommandSpec, candidates: list[str]) -> list[Any]:
    """Put `spec.example_config` first as ★ recommended; rest under a separator."""
    choices: list[Any] = []
    rec = spec.example_config
    rest = list(candidates)
    if rec and rec in rest:
        choices.append(questionary.Choice(f"★ {rec}  (recommended)", value=rec))
        rest.remove(rec)
    elif rec and Path(rec).exists():
        choices.append(questionary.Choice(f"★ {rec}  (recommended)", value=rec))
    if rest:
        if choices:
            choices.append(questionary.Separator("── other configs ──"))
        for c in rest:
            choices.append(questionary.Choice(c, value=c))
    return choices


def pick_or_build_config(spec: CommandSpec, documents_path: Optional[str] = None) -> str:
    """Ask the user to either pick an existing YAML or generate one.

    Validates the chosen YAML against the stage's dataclass and re-prompts
    on failure rather than letting the run blow up later.
    """
    while True:
        choice = questionary.select(
            f"Config for `{spec.name}`?",
            choices=[
                questionary.Choice("📂 Pick existing YAML", value="pick"),
                questionary.Choice("✨ Generate new YAML (guided)", value="build"),
                questionary.Choice("⌨  Type a path manually", value="manual"),
            ],
            style=QSTYLE, qmark=QMARK,
        ).ask()
        if choice is None:
            raise KeyboardInterrupt

        path: Optional[str] = None

        if choice == "pick":
            candidates = find_yaml_configs(spec)
            ranked = _ranked_choices(spec, candidates)
            if not ranked:
                questionary.print(
                    f"No YAML configs found for `{spec.name}`, falling back to manual entry.",
                    style="fg:yellow",
                )
                choice = "manual"
            else:
                picked = questionary.select(
                    f"Select a config for `{spec.name}`",
                    choices=ranked,
                    style=QSTYLE, qmark=QMARK,
                ).ask()
                if picked is None:
                    raise KeyboardInterrupt
                path = picked

        if choice == "manual":
            manual = _prompt("Path to YAML config")
            if not os.path.exists(manual):
                _show_error_panel(manual, "file not found")
                continue
            path = manual

        if choice == "build":
            builder = BUILDERS.get(spec.name)
            if builder is None:
                questionary.print(
                    f"No guided builder for `{spec.name}` — pick an existing YAML.",
                    style="fg:yellow",
                )
                continue
            if spec.name == "index":
                path = builder(documents_path=documents_path)  # type: ignore[call-arg]
            else:
                path = builder()

        assert path is not None
        err = _validate_yaml(path, spec)
        if err is None:
            return path
        _show_error_panel(path, err)
        if not _confirm("Try a different config?", default=True):
            raise KeyboardInterrupt
