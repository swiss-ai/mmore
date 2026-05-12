"""Shared visuals: banner, palette, panel helpers."""

from __future__ import annotations

from questionary import Style
from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text

console = Console()

QSTYLE = Style(
    [
        ("qmark", "fg:#5fd7ff bold"),
        ("question", "bold"),
        ("answer", "fg:#ff5fd7 bold"),
        ("pointer", "fg:#5fd7ff bold"),
        ("highlighted", "fg:#5fd7ff bold"),
        ("selected", "fg:#ff5fd7"),
        ("instruction", "fg:#808080 italic"),
        ("disabled", "fg:#ffaf00 italic"),
    ]
)
QMARK = "▸"

# Palette
ACCENT = "bright_cyan"
ACCENT2 = "magenta"
MUTED = "grey58"
OK = "bold green"
WARN = "yellow"
ERR = "bold red"

BANNER = r"""

 ███╗   ███╗███╗   ███╗ ██████╗ ██████╗ ███████╗
 ████╗ ████║████╗ ████║██╔═══██╗██╔══██╗██╔════╝
 ██╔████╔██║██╔████╔██║██║   ██║██████╔╝█████╗
 ██║╚██╔╝██║██║╚██╔╝██║██║   ██║██╔══██╗██╔══╝
 ██║ ╚═╝ ██║██║ ╚═╝ ██║╚██████╔╝██║  ██║███████╗
 ╚═╝     ╚═╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝
"""


def _gradient(text: str, start: str = "bright_cyan", end: str = "magenta") -> Text:
    """Cheap two-color gradient — top half ACCENT, bottom half ACCENT2."""
    lines = text.splitlines()
    half = max(1, len(lines) // 2)
    out = Text()
    for i, line in enumerate(lines):
        style = start if i < half else end
        out.append(line + "\n", style=style)
    return out


def show_banner(subtitle: str = "interactive launcher") -> None:
    body = Group(
        _gradient(BANNER),
        Align.center(Text(subtitle, style=f"italic {MUTED}")),
    )
    console.print(
        Panel(
            body,
            border_style=ACCENT,
            padding=(0, 2),
        )
    )


def section(title: str, body: str | Text, style: str = ACCENT) -> Panel:
    return Panel(
        body if isinstance(body, Text) else Text(body),
        title=f"[bold]{title}[/bold]",
        border_style=style,
        padding=(1, 2),
    )


def step_header(idx: int, total: int, name: str) -> None:
    bar = "─" * 4
    console.print()
    console.print(
        f"[{ACCENT}]{bar}[/] [bold]Step {idx}/{total}[/bold] "
        f"[{ACCENT2}]{name}[/] [{ACCENT}]{bar}[/]"
    )
