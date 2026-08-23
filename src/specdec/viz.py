"""Plots and a colorized SVG render of a speculative-decoding transcript."""

from __future__ import annotations

import html
from pathlib import Path


def plot_bars(labels: list[str], values: list[float], path, *, title: str, ylabel: str, fmt: str = "{:.1f}") -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    colors = ["#6c7a89", "#2e7d32"]
    bars = ax.bar(labels, values, color=colors[: len(labels)], width=0.5)
    for bar, val in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, val + max(values) * 0.02, fmt.format(val),
                 ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max(values) * 1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_acceptance_over_blocks(acceptance_rates: list[float], path, *, title: str, window: int = 15) -> None:
    """Each speculative block only accepts a handful of tokens, so the raw
    per-block rate is a noisy 0/0.5/0.67/... step sequence (see the faint
    line); the bold line is a trailing rolling average over `window` blocks,
    which is what actually shows the trend."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = range(1, len(acceptance_rates) + 1)
    ax.plot(x, acceptance_rates, color="#2e7d32", linewidth=1, alpha=0.25, label="per block")

    if len(acceptance_rates) >= window:
        kernel = np.ones(window) / window
        smoothed = np.convolve(acceptance_rates, kernel, mode="valid")
        smoothed_x = range(window, len(acceptance_rates) + 1)
        ax.plot(smoothed_x, smoothed, color="#1b5e20", linewidth=2.5, label=f"rolling mean (window={window})")

    ax.set_xlabel("speculative block")
    ax.set_ylabel("fraction of draft tokens accepted")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


_KIND_STYLE = {
    "accepted": ("#d7f2dc", "#1b5e20", "accepted from draft"),
    "corrected": ("#fde0dc", "#b71c1c", "draft rejected, corrected by main model"),
    "bonus": ("#dce8fd", "#0d47a1", "bonus token from main model"),
    "prompt": ("#eeeeee", "#555555", "prompt"),
}


def render_transcript_svg(
    pieces: list[tuple[str, str]],
    path: str | Path,
    *,
    title: str,
    chars_per_line: int = 84,
    font_size: int = 15,
) -> None:
    """``pieces``: list of ``(decoded_text, kind)``, ``kind`` one of the keys
    in ``_KIND_STYLE``. Renders a self-contained SVG (so it embeds directly
    in the README, unlike an HTML file) of one real generation run, with
    every token colored by how it was actually produced: fast-forwarded from
    the draft, corrected by the main model, or a free bonus token. This is a
    direct visualization of real output, not a mockup.
    """
    char_w = font_size * 0.6
    line_h = font_size * 1.7

    chars: list[tuple[str, str]] = []
    for text, kind in pieces:
        chars.extend((ch, kind) for ch in text)

    lines: list[list[tuple[str, str]]] = [[]]
    col = 0
    for ch, kind in chars:
        if ch == "\n":
            lines.append([])
            col = 0
            continue
        if col >= chars_per_line and ch == " ":
            lines.append([])
            col = 0
            continue
        lines[-1].append((ch, kind))
        col += 1

    width = int(chars_per_line * char_w) + 40
    top_margin = 70
    height = top_margin + len(lines) * line_h + 20

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height:.0f}" '
        f'viewBox="0 0 {width} {height:.0f}" font-family="Consolas, Menlo, monospace">',
        f'<rect width="{width}" height="{height:.0f}" fill="#12141a"/>',
        f'<text x="20" y="24" font-size="14" fill="#9aa0ab">{html.escape(title)}</text>',
    ]

    lx = 20
    ly = 46
    for kind in ("accepted", "corrected", "bonus"):
        bg, fg, label = _KIND_STYLE[kind]
        label_w = len(label) * 6.4 + 20
        svg.append(f'<rect x="{lx}" y="{ly - 12}" width="{label_w:.0f}" height="18" rx="4" fill="{bg}"/>')
        svg.append(f'<text x="{lx + 8}" y="{ly + 1}" font-size="12" fill="{fg}">{html.escape(label)}</text>')
        lx += label_w + 12

    y = top_margin + font_size
    for line in lines:
        x = 20.0
        i = 0
        while i < len(line):
            kind = line[i][1]
            j = i
            run = ""
            while j < len(line) and line[j][1] == kind:
                run += line[j][0]
                j += 1
            bg, fg, _ = _KIND_STYLE[kind]
            run_w = len(run) * char_w
            if kind != "prompt":
                svg.append(
                    f'<rect x="{x:.1f}" y="{y - font_size:.1f}" width="{run_w:.1f}" height="{line_h:.1f}" fill="{bg}"/>'
                )
            escaped = html.escape(run).replace(" ", "&#160;")
            svg.append(
                f'<text x="{x:.1f}" y="{y:.1f}" font-size="{font_size}" fill="{fg}" '
                f'xml:space="preserve">{escaped}</text>'
            )
            x += run_w
            i = j
        y += line_h

    svg.append("</svg>")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(svg), encoding="utf-8")
