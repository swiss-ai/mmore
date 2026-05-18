"""mmore TUI entry point."""

from __future__ import annotations

import threading

import questionary
from rich.panel import Panel
from rich.text import Text

from mmore.tui.commands import REGISTRY, check_stage_available
from mmore.tui.config_builder import (
    build_full_pipeline_wizard,
    pick_or_build_config,
)
from mmore.tui.exceptions import UserCancelledError
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
    run_step,
    section,
    show_banner,
)

_PIPELINE_STAGES = ("process", "postprocess", "index")


def _warm_pipeline_dataclasses() -> None:
    """Pre-load process/postprocess/index dataclasses in a daemon thread.

    Called when entering the wizard or full-pipeline flows, where several YAML
    validations happen back-to-back. The import cost then overlaps with the
    wizard's own prompts. Daemon = no impact on exit. Stages whose canary
    imports are missing are skipped so partial installs don't crash the warm-up.
    """

    def _warm() -> None:
        for stage in _PIPELINE_STAGES:
            spec = REGISTRY[stage]
            if check_stage_available(spec) is not None or spec.config_dataclass is None:
                continue
            try:
                spec.config_dataclass()
            except Exception:  # noqa: BLE001
                pass

    threading.Thread(target=_warm, daemon=True).start()


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


def _missing_extras_notice() -> Panel | None:
    """One-line-per-install-command notice — kept compact so the banner stays visible."""
    install_to_stages: dict[str, list[str]] = {}
    for name, spec in REGISTRY.items():
        hint = check_stage_available(spec)
        if hint and "Install with: " in hint:
            cmd = hint.split("Install with: ", 1)[1].strip()
            install_to_stages.setdefault(cmd, []).append(name)

    if not install_to_stages:
        return None

    body = Text()
    for i, (cmd, stages) in enumerate(install_to_stages.items()):
        if i > 0:
            body.append("\n")
        body.append(", ".join(stages), style="bold white")
        body.append("  →  ", style="yellow")
        body.append(cmd, style="cyan")

    return Panel(
        body,
        title="[bold yellow]⚠  missing extras[/]",
        border_style="yellow",
        padding=(0, 1),
    )


def _disabled_label(label: str) -> str:
    """Prefix a menu label so its disabled state is immediately readable."""
    return f"⚠  {label}"


def _run_single_command() -> None:
    choices = []
    enabled_count = 0
    for spec in REGISTRY.values():
        hint = check_stage_available(spec)
        label = f"{spec.name:<12} — {spec.description}"
        if hint:
            choices.append(
                questionary.Choice(
                    _disabled_label(label), value=spec.name, disabled="missing extras"
                )
            )
        else:
            choices.append(questionary.Choice(label, value=spec.name))
            enabled_count += 1

    # questionary crashes ("InquirerControl has no attribute 'pointed_at'") when
    # every choice is disabled because it can't pick an initial pointer. Bail
    # out with a clear notice instead.
    if enabled_count == 0:
        notice = _missing_extras_notice()
        if notice is not None:
            console.print(notice)
        return

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
        run_step(spec.description, spec.run, **kwargs)
    console.print(f"[{OK}]✓ {name} finished[/]")


def _chat_only() -> None:
    config_file = pick_or_build_config(REGISTRY["ragcli"])
    console.print()
    console.print(section("RAG chat", Text(f"config: {config_file}", style=MUTED)))
    REGISTRY["ragcli"].run(config_file=config_file)


def _run_full_wizard() -> None:
    _warm_pipeline_dataclasses()
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
        run_pipeline_with_configs(
            paths["process"], paths["postprocess"], paths["index"]
        )


def _pipeline_hint() -> str | None:
    """Return a combined hint if any of process/postprocess/index is missing."""
    hints = [
        check_stage_available(REGISTRY[s]) for s in ("process", "postprocess", "index")
    ]
    hints = [h for h in hints if h]
    return " | ".join(hints) if hints else None


def _main_menu() -> str | None:
    notice = _missing_extras_notice()
    if notice is not None:
        console.print(notice)

    pipeline_hint = _pipeline_hint()
    chat_hint = check_stage_available(REGISTRY["ragcli"])
    # The wizard validates each generated YAML against the stage's dataclass,
    # which transitively imports torch / transformers / etc. — so it needs the
    # same extras as the full pipeline. Reuse `_pipeline_hint()` to stay aligned.
    wizard_hint = _pipeline_hint()

    pipeline_label = "🚀 Run full pipeline  (process → postprocess → index)"
    wizard_label = "🧙  Build a full pipeline config (guided wizard)"
    chat_label = "💬 Chat with indexed documents"

    pipeline_choice = questionary.Choice(
        _disabled_label(pipeline_label) if pipeline_hint else pipeline_label,
        value="pipeline",
        disabled="missing extras" if pipeline_hint else None,
    )
    wizard_choice = questionary.Choice(
        _disabled_label(wizard_label) if wizard_hint else wizard_label,
        value="wizard",
        disabled="missing extras" if wizard_hint else None,
    )
    chat_choice = questionary.Choice(
        _disabled_label(chat_label) if chat_hint else chat_label,
        value="chat",
        disabled="missing extras" if chat_hint else None,
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
                _warm_pipeline_dataclasses()
                run_full_pipeline()
            elif mode == "wizard":
                _run_full_wizard()
            elif mode == "chat":
                _chat_only()
        except (UserCancelledError, KeyboardInterrupt):
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
