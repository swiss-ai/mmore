"""mmore TUI entry point."""
from __future__ import annotations

import time

import questionary
from questionary import Style
from rich.spinner import Spinner
from rich.live import Live
from rich.text import Text

from mmore.tui.commands import REGISTRY
from mmore.tui.config_builder import pick_or_build_config
from mmore.tui.pipeline import run_full_pipeline
from mmore.tui.theme import ACCENT, ACCENT2, MUTED, OK, console, section, show_banner

QSTYLE = Style([
    ("qmark", "fg:#5fd7ff bold"),
    ("question", "bold"),
    ("answer", "fg:#ff5fd7 bold"),
    ("pointer", "fg:#5fd7ff bold"),
    ("highlighted", "fg:#5fd7ff bold"),
    ("selected", "fg:#ff5fd7"),
    ("instruction", "fg:#808080 italic"),
])


def _run_with_spinner(label: str, fn, **kwargs) -> None:
    start = time.time()
    spinner = Spinner("dots", text=Text(f"  {label}…", style=ACCENT))
    with Live(spinner, console=console, refresh_per_second=12, transient=True):
        fn(**kwargs)
    console.print(
        f"  [{OK}]✓[/] {label} [dim]({time.time() - start:.1f}s)[/dim]"
    )


def _run_single_command() -> None:
    choices = [
        questionary.Choice(f"{spec.name:<12} — {spec.description}", value=spec.name)
        for spec in REGISTRY.values()
    ]
    name = questionary.select(
        "Pick a command", choices=choices, style=QSTYLE, qmark="▸",
    ).ask()
    if name is None:
        return
    spec = REGISTRY[name]
    config_file = pick_or_build_config(spec)
    kwargs = {"config_file": config_file}
    if spec.needs_input_data:
        input_data = questionary.text(
            "Input JSONL path",
            default="examples/process/outputs/merged/merged_results.jsonl",
            style=QSTYLE, qmark="▸",
        ).ask()
        if input_data is None:
            return
        kwargs["input_data"] = input_data

    console.print()
    console.print(section(
        f"Running {name}",
        Text(f"config: {config_file}", style=MUTED),
        style=ACCENT2,
    ))
    interactive = name in {"ragcli", "retrieve", "rag"}
    if interactive:
        spec.run(**kwargs)
    else:
        _run_with_spinner(spec.description, spec.run, **kwargs)
    console.print(f"[{OK}]✓ {name} finished[/]")


def _chat_only() -> None:
    config_file = pick_or_build_config(REGISTRY["ragcli"])
    console.print()
    console.print(section("RAG chat", Text(f"config: {config_file}", style=MUTED)))
    REGISTRY["ragcli"].run(config_file=config_file)


def _main_menu() -> str | None:
    return questionary.select(
        "What do you want to do?",
        choices=[
            questionary.Choice("⚙  Run a single command", value="single"),
            questionary.Choice(
                "🚀 Run full pipeline  (process → postprocess → index)",
                value="pipeline",
            ),
            questionary.Choice("💬 Chat with indexed documents", value="chat"),
            questionary.Separator(),
            questionary.Choice("✕  Quit", value="quit"),
        ],
        style=QSTYLE,
        qmark="▸",
    ).ask()


def run() -> None:
    console.clear()
    show_banner("interactive launcher")
    while True:
        try:
            mode = _main_menu()
            if mode in (None, "quit"):
                console.print(f"[{ACCENT}]bye![/]")
                return
            if mode == "single":
                _run_single_command()
            elif mode == "pipeline":
                run_full_pipeline()
            elif mode == "chat":
                _chat_only()
        except KeyboardInterrupt:
            console.print(f"\n[{ACCENT2}]interrupted.[/]")
            return
        except Exception as e:  # noqa: BLE001
            console.print(f"[bold red]error:[/] {e}")
            if not questionary.confirm("Continue?", default=True, style=QSTYLE).ask():
                return
