"""Shared visuals: banner, palette, panel helpers."""

from __future__ import annotations

import time
from typing import Any, Callable

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


def _mmore_logo(text: str) -> Text:
    """Color the banner like the mmore GitHub logo.

    Strategy, per character:
    - The second `M` (columns 12:23 of every row) is rendered fully in yellow.
    - Elsewhere: outline characters (`╔╗╚╝═║╔╝╗`, etc.) are white and the
      filled `█` blocks are black, giving the letters a hollow look.
    """
    OUTLINE = set("╔╗╚╝═║╠╣╦╩╬╔╝╗┌┐└┘─│")
    out = Text()
    for line in text.splitlines():
        if not line.strip():
            out.append(line + "\n")
            continue
        left = line[:12]
        mid = line[12:23]
        right = line[23:]

        def _emit(segment: str) -> None:
            for ch in segment:
                if ch == "█":
                    # explicit hex — terminal "black" often renders as dark grey
                    out.append(ch, style="#000000")
                elif ch in OUTLINE:
                    out.append(ch, style="bold #ffffff")
                else:
                    out.append(ch)

        _emit(left)
        out.append(mid, style="bold yellow")
        _emit(right)
        out.append("\n")
    return out


def show_banner(subtitle: str = "interactive launcher") -> None:
    body = Group(
        _mmore_logo(BANNER),
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


def run_step(label: str, fn: Callable[..., Any], **kwargs: Any) -> float:
    """Print a start line, call fn(**kwargs), print a timed done line.

    Heavy pipeline commands emit their own logs via logging/click which bypass
    rich.Console — a Live spinner would clash with them. Plain prints keep the
    output readable while still showing progress.
    """
    start = time.time()
    console.print(f"  [{ACCENT}]▸[/] {label}…")
    fn(**kwargs)
    elapsed = time.time() - start
    console.print(f"  [{OK}]✓[/] {label} [dim]({elapsed:.1f}s)[/dim]")
    return elapsed


def step_header(idx: int, total: int, name: str) -> None:
    bar = "─" * 4
    console.print()
    console.print(
        f"[{ACCENT}]{bar}[/] [bold]Step {idx}/{total}[/bold] "
        f"[{ACCENT2}]{name}[/] [{ACCENT}]{bar}[/]"
    )
