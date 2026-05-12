"""mmore TUI entry point."""

from __future__ import annotations

import time

import questionary
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from rich.panel import Panel

from mmore.tui.commands import REGISTRY, check_stage_available
from mmore.tui.config_builder import (
    build_full_pipeline_wizard,
    pick_or_build_config,
)
from mmore.tui.exceptions import CancelledByUser
from mmore.tui.paths import cwd_default
from mmore.tui.pipeline import run_full_pipeline, run_pipeline_with_configs
from mmore.tui.theme import (
    ACCENT,
    ACCENT2,
    MUTED,
    OK,
    QMARK,
    QSTYLE,
    console,
    section,
    show_banner,
)


def _show_missing_extras(spec_name: str, hint: str) -> None:
    console.print(
        Panel(
            Text.assemble(
                (f"Stage `{spec_name}` can't run.\n\n", "bold"),
                (hint, "yellow"),
            ),
            title="[bold yellow]missing dependencies[/]",
            border_style="yellow",
            padding=(1, 2),
        )
    )


def _run_with_spinner(label: str, fn, **kwargs) -> None:
    start = time.time()
    spinner = Spinner("dots", text=Text(f"  {label}…", style=ACCENT))
    with Live(spinner, console=console, refresh_per_second=12, transient=True):
        fn(**kwargs)
    console.print(f"  [{OK}]✓[/] {label} [dim]({time.time() - start:.1f}s)[/dim]")


def _run_single_command() -> None:
    choices = []
    for spec in REGISTRY.values():
        hint = check_stage_available(spec)
        label = f"{spec.name:<12} — {spec.description}"
        if hint:
            label += "  [dim](extras missing)[/dim]"
            choices.append(
                questionary.Choice(label, value=spec.name, disabled=hint)
            )
        else:
            choices.append(questionary.Choice(label, value=spec.name))
    name = questionary.select(
        "Pick a command",
        choices=choices,
        style=QSTYLE,
        qmark=QMARK,
    ).ask()
    if name is None:
        return
    spec = REGISTRY[name]
    # Defensive re-check in case the user typed past the disabled state.
    hint = check_stage_available(spec)
    if hint:
        _show_missing_extras(spec.name, hint)
        return
    config_file = pick_or_build_config(spec)
    kwargs = {"config_file": config_file}
    if spec.needs_input_data:
        input_data = questionary.text(
            "Input JSONL path",
            default=cwd_default("outputs/process/merged/merged_results.jsonl"),
            style=QSTYLE,
            qmark=QMARK,
        ).ask()
        if input_data is None:
            return
        kwargs["input_data"] = input_data

    console.print()
    console.print(
        section(
            f"Running {name}",
            Text(f"config: {config_file}", style=MUTED),
            style=ACCENT2,
        )
    )
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


def _run_full_wizard() -> None:
    paths = build_full_pipeline_wizard()
    console.print()
    console.print(
        section(
            "Wizard complete",
            Text(
                "process:     " + paths["process"] + "\n"
                "postprocess: " + paths["postprocess"] + "\n"
                "index:       " + paths["index"],
                style=MUTED,
            ),
            style=ACCENT2,
        )
    )
    if questionary.confirm(
        "Run the pipeline now with these configs?",
        default=True,
        style=QSTYLE,
        qmark=QMARK,
    ).ask():
        run_pipeline_with_configs(paths["process"], paths["postprocess"], paths["index"])


def _pipeline_hint() -> str | None:
    """Return a combined hint if any of process/postprocess/index is missing."""
    hints = [
        check_stage_available(REGISTRY[s])
        for s in ("process", "postprocess", "index")
    ]
    hints = [h for h in hints if h]
    return " | ".join(hints) if hints else None


def _main_menu() -> str | None:
    pipeline_hint = _pipeline_hint()
    chat_hint = check_stage_available(REGISTRY["ragcli"])

    pipeline_choice = questionary.Choice(
        "🚀 Run full pipeline  (process → postprocess → index)"
        + ("  [dim](extras missing)[/dim]" if pipeline_hint else ""),
        value="pipeline",
        disabled=pipeline_hint,
    )
    wizard_choice = questionary.Choice(
        "🧙  Build a full pipeline config (guided wizard)",
        value="wizard",
    )  # wizard only writes YAML, no heavy imports needed
    chat_choice = questionary.Choice(
        "💬 Chat with indexed documents"
        + ("  [dim](extras missing)[/dim]" if chat_hint else ""),
        value="chat",
        disabled=chat_hint,
    )

    return questionary.select(
        "What do you want to do?",
        choices=[
            questionary.Choice("⚙  Run a single command", value="single"),
            pipeline_choice,
            wizard_choice,
            chat_choice,
            questionary.Separator(),
            questionary.Choice("✕  Quit", value="quit"),
        ],
        style=QSTYLE,
        qmark=QMARK,
    ).ask()


def run() -> None:
    console.clear()
    show_banner("interactive launcher")
    while True:
        # Ctrl-C at the main menu itself quits; inside any sub-flow it
        # cancels and returns here.
        try:
            mode = _main_menu()
        except KeyboardInterrupt:
            console.print(f"\n[{ACCENT}]bye![/]")
            return
        if mode in (None, "quit"):
            console.print(f"[{ACCENT}]bye![/]")
            return

        try:
            if mode == "single":
                _run_single_command()
            elif mode == "pipeline":
                run_full_pipeline()
            elif mode == "wizard":
                _run_full_wizard()
            elif mode == "chat":
                _chat_only()
        except (CancelledByUser, KeyboardInterrupt):
            console.print(f"[{ACCENT2}]cancelled — back to menu.[/]")
            continue
        except Exception as e:  # noqa: BLE001
            console.print(f"[bold red]error:[/] {e}")
            try:
                cont = questionary.confirm(
                    "Continue?", default=True, style=QSTYLE
                ).ask()
            except KeyboardInterrupt:
                return
            if not cont:
                return
