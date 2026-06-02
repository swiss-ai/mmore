"""Lightweight JSONL inspector for TUI result previews.

Streams the file line-by-line (no heavy imports like torch/transformers)
and prints a rich summary table + sample documents.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from mmore.tui.theme import ACCENT, ACCENT2, MUTED, console


def _iter_dicts(path: str):
    """Yield raw dicts from a JSONL file without importing MultimodalSample."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def inspect_jsonl(path: str, max_samples: int = 3) -> None:
    """Print a summary of a JSONL file: counts, breakdowns, sample docs."""
    if not os.path.exists(path):
        console.print(f"  [dim]no output file at {path}[/dim]")
        return

    total = 0
    processor_types: Counter[str] = Counter()
    file_extensions: Counter[str] = Counter()
    modality_types: Counter[str] = Counter()
    total_text_len = 0
    samples: list[dict[str, Any]] = []

    for doc in _iter_dicts(path):
        total += 1

        meta = doc.get("metadata", {})
        pt = meta.get("processor_type", "unknown")
        processor_types[pt] += 1

        fp = meta.get("file_path", "")
        ext = Path(fp).suffix.lower() if fp else "(none)"
        file_extensions[ext] += 1

        text = doc.get("text", "")
        if isinstance(text, str):
            total_text_len += len(text)

        for mod in doc.get("modalities", []):
            modality_types[mod.get("type", "unknown")] += 1

        if len(samples) < max_samples:
            samples.append(doc)

    if total == 0:
        console.print("  [dim]empty JSONL (0 documents)[/dim]")
        return

    # --- Stats table ---
    table = Table(
        title="[bold]Results summary[/bold]",
        title_style=ACCENT2,
        border_style=ACCENT,
        header_style=f"bold {ACCENT}",
        show_lines=False,
        padding=(0, 2),
    )
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    table.add_row("Total documents", str(total))
    table.add_row("Avg text length", f"{total_text_len // total:,} chars")

    if processor_types:
        breakdown = ", ".join(f"{k}: {v}" for k, v in processor_types.most_common())
        table.add_row("Processor types", breakdown)

    if file_extensions:
        breakdown = ", ".join(f"{k}: {v}" for k, v in file_extensions.most_common())
        table.add_row("File types", breakdown)

    if modality_types:
        breakdown = ", ".join(f"{k}: {v}" for k, v in modality_types.most_common())
        table.add_row("Modalities", breakdown)

    console.print()
    console.print(table)

    # --- Sample documents ---
    if samples:
        sample_text = Text()
        for i, doc in enumerate(samples, 1):
            meta = doc.get("metadata", {})
            fp = meta.get("file_path", "?")
            pt = meta.get("processor_type", "?")
            text = doc.get("text", "")
            if isinstance(text, str):
                preview = text[:200].replace("\n", " ")
                if len(text) > 200:
                    preview += "…"
            else:
                preview = "(structured content)"
            sample_text.append(f"#{i} ", style="bold")
            sample_text.append(f"{fp}  ")
            sample_text.append(f"({pt})", style="dim")
            sample_text.append("\n")
            sample_text.append(preview + "\n\n", style=MUTED)

        console.print(
            Panel(
                sample_text,
                title=f"[bold]Sample documents (first {len(samples)})[/bold]",
                border_style=ACCENT,
                padding=(1, 2),
            )
        )
