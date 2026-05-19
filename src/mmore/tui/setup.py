"""Setup wizard: install extras + generate .env in one guided flow."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import questionary
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from mmore.tui.commands import REGISTRY, check_stage_available
from mmore.tui.config_builder import _ask, _confirm, _prompt
from mmore.tui.theme import ACCENT, ACCENT2, MUTED, OK, QMARK, QSTYLE, console

# ---------------------------------------------------------------------------
# Stage → extras mapping
# ---------------------------------------------------------------------------

_STAGE_EXTRAS: dict[str, list[str]] = {
    "process": ["process"],
    "postprocess": ["process"],
    "index": ["index"],
    "rag": ["rag"],
    "ragcli": ["rag"],
    "retrieve": ["rag", "api"],
    "websearch": ["websearch"],
}

_COMPUTE_EXTRAS = [
    ("cpu", "CPU-only (no CUDA)"),
    ("cu126", "CUDA 12.6 (GPU)"),
]

# ---------------------------------------------------------------------------
# Stage → env vars that may be needed
# ---------------------------------------------------------------------------

_STAGE_ENV_VARS: dict[str, list[tuple[str, str, str]]] = {
    # (var_name, description, default_or_empty)
    "process": [
        ("ROOT_OUT_DIR", "Root output directory for processed results", ""),
        ("ROOT_IN_DIR", "Root input directory for source documents", ""),
    ],
    "rag": [
        ("OPENAI_API_KEY", "OpenAI API key (for GPT models)", ""),
        ("ANTHROPIC_API_KEY", "Anthropic API key (for Claude models)", ""),
        ("MISTRAL_API_KEY", "Mistral API key", ""),
        ("COHERE_API_KEY", "Cohere API key", ""),
        ("HF_TOKEN", "HuggingFace token (for gated models)", ""),
    ],
    "websearch": [
        ("TAVILY_API_KEY", "Tavily API key (optional, DuckDuckGo used otherwise)", ""),
    ],
}

# Aliases: ragcli and retrieve share rag's env vars
_STAGE_ENV_VARS["ragcli"] = _STAGE_ENV_VARS["rag"]
_STAGE_ENV_VARS["retrieve"] = _STAGE_ENV_VARS["rag"]

# Profiling env vars (always available)
_PROFILING_VARS: list[tuple[str, str, str]] = [
    ("MMORE_PROFILING_ENABLED", "Enable profiling", "false"),
    ("MMORE_PROFILING_OUTPUT_DIR", "Profiling output directory", "./profiling_output"),
]


def _detect_installed_stages() -> dict[str, bool]:
    """Check which stages have their deps installed."""
    return {name: check_stage_available(spec) is None for name, spec in REGISTRY.items()}


def _pick_stages() -> list[str]:
    """Ask the user which pipeline stages they want to use."""
    installed = _detect_installed_stages()
    choices = []
    for name, spec in REGISTRY.items():
        label = f"{name:<12} — {spec.description}"
        if installed[name]:
            label += "  [dim](installed)[/dim]"
        choices.append(questionary.Choice(label, value=name, checked=not installed[name]))

    selected = _ask(
        questionary.checkbox(
            "Which stages do you want to set up?",
            choices=choices,
            style=QSTYLE,
            qmark=QMARK,
        )
    )
    return selected


def _pick_compute() -> str:
    """Ask the user which compute backend to use."""
    choices = [
        questionary.Choice(f"{name:<6} — {desc}", value=name)
        for name, desc in _COMPUTE_EXTRAS
    ]
    return _ask(
        questionary.select(
            "Compute backend",
            choices=choices,
            style=QSTYLE,
            qmark=QMARK,
        )
    )


def _build_uv_command(stages: list[str], compute: str) -> list[str]:
    """Build the uv sync command from selected stages + compute."""
    extras: set[str] = {"tui"}  # always include TUI
    for stage in stages:
        extras.update(_STAGE_EXTRAS.get(stage, []))
    extras.add(compute)

    cmd = [sys.executable, "-m", "uv", "sync"]
    for extra in sorted(extras):
        cmd.extend(["--extra", extra])
    return cmd


def _install_deps(stages: list[str], compute: str) -> bool:
    """Run uv sync with the right extras. Returns True on success."""
    cmd = _build_uv_command(stages, compute)
    display_cmd = " ".join(cmd[2:])  # skip python -m prefix for display
    console.print(f"\n  [bold]Running:[/] {display_cmd}\n")

    result = subprocess.run(cmd, cwd=os.getcwd())
    if result.returncode == 0:
        console.print(f"  [{OK}]✓[/] Dependencies installed successfully")
        return True
    console.print("  [bold red]✗[/] Installation failed — check output above")
    return False


def _collect_env_vars(stages: list[str]) -> dict[str, str]:
    """Prompt the user for env vars needed by their selected stages."""
    seen: set[str] = set()
    env_vars: dict[str, str] = {}

    # Gather all relevant vars (deduplicated)
    all_vars: list[tuple[str, str, str]] = []
    for stage in stages:
        for var in _STAGE_ENV_VARS.get(stage, []):
            if var[0] not in seen:
                seen.add(var[0])
                all_vars.append(var)

    if not all_vars:
        return env_vars

    console.print(
        Panel(
            "Set environment variables for your selected stages.\n"
            "Leave blank to skip — you can always edit the .env file later.",
            title="[bold]Environment variables[/bold]",
            border_style=ACCENT,
            padding=(1, 2),
        )
    )

    for var_name, description, default in all_vars:
        # Check if already set in environment
        current = os.environ.get(var_name, "")
        hint = f" [dim](current: {current[:20]}…)[/dim]" if current else ""
        value = _prompt(f"{var_name} — {description}{hint}", default=current or default)
        if value:
            env_vars[var_name] = value

    # Optionally add profiling vars
    if _confirm("Configure profiling settings?", default=False):
        for var_name, description, default in _PROFILING_VARS:
            value = _prompt(f"{var_name} — {description}", default=default)
            if value:
                env_vars[var_name] = value

    return env_vars


def _write_dotenv(env_vars: dict[str, str], path: str = ".env") -> str:
    """Write or merge env vars into a .env file.

    Existing variables in the file are preserved; new ones are appended.
    """
    existing: dict[str, str] = {}
    lines: list[str] = []
    env_path = Path(path)

    if env_path.exists():
        raw = env_path.read_text()
        for line in raw.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                key = stripped.split("=", 1)[0].strip()
                existing[key] = line
            lines.append(line)

    # Append new vars
    added = []
    for key, value in env_vars.items():
        if key in existing:
            continue  # don't overwrite existing
        # Quote values that contain spaces
        if " " in value:
            entry = f'{key}="{value}"'
        else:
            entry = f"{key}={value}"
        lines.append(entry)
        added.append(key)

    if lines and not lines[-1].endswith("\n"):
        content = "\n".join(lines) + "\n"
    else:
        content = "\n".join(lines)

    env_path.write_text(content)
    return str(env_path)


def _preview_dotenv(env_vars: dict[str, str]) -> None:
    """Show what will be written to .env."""
    if not env_vars:
        console.print("  [dim]No environment variables to write.[/dim]")
        return

    table = Table(
        title="[bold].env preview[/bold]",
        title_style=ACCENT2,
        border_style=ACCENT,
        show_lines=False,
    )
    table.add_column("Variable", style="bold")
    table.add_column("Value", style=MUTED)

    for key, value in env_vars.items():
        # Mask API keys
        if "KEY" in key or "TOKEN" in key:
            display = value[:4] + "…" + value[-4:] if len(value) > 8 else "****"
        else:
            display = value
        table.add_row(key, display)

    console.print(table)


def run_setup_wizard() -> None:
    """Full setup wizard: pick stages → install deps → generate .env."""
    console.print(
        Panel(
            Text(
                "This wizard will:\n"
                "  1. Install the right Python dependencies for your pipeline\n"
                "  2. Generate a .env file with the required environment variables",
            ),
            title="[bold]Setup wizard[/bold]",
            border_style=ACCENT2,
            padding=(1, 2),
        )
    )

    # Step 1: pick stages
    stages = _pick_stages()
    if not stages:
        console.print("  [dim]No stages selected — nothing to do.[/dim]")
        return

    # Step 2: pick compute backend
    compute = _pick_compute()

    # Step 3: show install command and confirm
    cmd = _build_uv_command(stages, compute)
    display_cmd = " ".join(cmd[2:])
    console.print(
        Panel(
            Text(display_cmd),
            title="[bold]Install command[/bold]",
            border_style=ACCENT,
            padding=(0, 2),
        )
    )
    if _confirm("Install dependencies now?", default=True):
        if not _install_deps(stages, compute):
            if not _confirm("Continue to .env setup despite install failure?", default=False):
                return

    # Step 4: collect env vars
    env_vars = _collect_env_vars(stages)

    # Step 5: preview and write .env
    if env_vars:
        _preview_dotenv(env_vars)
        env_path = _prompt(".env file path", default=".env")
        if _confirm(f"Write {len(env_vars)} variable(s) to {env_path}?", default=True):
            written = _write_dotenv(env_vars, env_path)
            console.print(f"  [{OK}]✓[/] Saved to {written}")
        else:
            console.print("  [dim]Skipped .env generation.[/dim]")
    else:
        console.print("  [dim]No environment variables needed for selected stages.[/dim]")

    console.print(f"\n  [{OK}]✓ Setup complete![/] Run [bold]mmore tui[/bold] to start.\n")
