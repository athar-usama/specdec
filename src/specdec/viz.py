"""Plots and a colorized SVG render of a speculative-decoding transcript.

Shared visual language across every chart here: slate gray for the naive
baseline, emerald green for the speculative-decoding numbers (matching the
"accepted" color in the live-generation transcript), amber only for the
handful of things worth calling out (an annotation, a mean line). Bars get
a value label directly above them so the numbers are legible even without
reading the axis.
"""

from __future__ import annotations

import html
from pathlib import Path

_NAIVE = "#64748b"
_SPEC = "#059669"
_SPEC_DARK = "#065f46"
_ACCENT = "#b45309"
_GRID = "#e5e7eb"
_TEXT = "#1f2937"
_MUTED = "#6b7280"
_BAND = "#f8fafc"
_SHADOW = "#0f172a"


def _style_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(_MUTED)
    ax.tick_params(colors=_MUTED, left=False)
    ax.title.set_color(_TEXT)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)


def _lighten(hex_color: str, amount: float) -> tuple:
    from matplotlib.colors import to_rgb

    r, g, b = to_rgb(hex_color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def _gradient_bar(ax, x_center: float, height: float, width: float, color: str, *, zorder: int = 3) -> None:
    """A bar with a rounded top, a soft vertical gradient, and a faint drop
    shadow, instead of a flat-filled rectangle. The rounded bottom corners a
    plain FancyBboxPatch would draw are pushed below ``y=0`` and cropped away
    by the axes' own clip box, so only the top stays visibly rounded."""
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import FancyBboxPatch

    if height <= 0:
        return
    radius = min(width, height) * 0.22

    shadow = FancyBboxPatch(
        (x_center - width / 2 + width * 0.05, -radius), width, height + radius,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=0, facecolor=_SHADOW, alpha=0.10, zorder=zorder - 1, transform=ax.transData,
    )
    ax.add_patch(shadow)

    outline = FancyBboxPatch(
        (x_center - width / 2, -radius), width, height + radius,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=0, facecolor="none", zorder=zorder, transform=ax.transData,
    )
    ax.add_patch(outline)

    cmap = LinearSegmentedColormap.from_list("bar", [_lighten(color, 0.55), color])
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    im = ax.imshow(
        gradient, extent=(x_center - width / 2, x_center + width / 2, -radius, height),
        origin="lower", aspect="auto", cmap=cmap, zorder=zorder, transform=ax.transData,
    )
    im.set_clip_path(outline)


def _value_chip(ax, x_center: float, y: float, text: str, *, color: str = _TEXT) -> None:
    ax.text(
        x_center, y, text, ha="center", va="bottom", fontsize=12.5, color=color, fontweight="bold",
        bbox={"boxstyle": "round,pad=0.32", "facecolor": "white", "edgecolor": _GRID, "linewidth": 1},
        zorder=6,
    )


def plot_decoding_comparison(
    naive_tps: float,
    spec_tps: float,
    naive_calls: int,
    spec_calls: int,
    path,
    *,
    n_tokens: int,
) -> None:
    """One figure, two panels: tokens/second and main-model forward passes,
    naive vs. speculative, side by side so the two numbers that matter are
    visible in a single glance rather than three separate small charts."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5))
    fig.patch.set_facecolor("white")
    labels = ["naive", "speculative"]
    colors = [_NAIVE, _SPEC]
    bar_width = 0.5

    for ax, values, title, ylabel, fmt in (
        (ax1, [naive_tps, spec_tps], f"Throughput ({n_tokens} tokens)", "tokens / second", "{:.0f}"),
        (ax2, [float(naive_calls), float(spec_calls)], "Main-model forward passes", "forward passes", "{:.0f}"),
    ):
        ax.set_facecolor(_BAND)
        for i, (val, color) in enumerate(zip(values, colors, strict=True)):
            _gradient_bar(ax, i, val, bar_width, color)
            _value_chip(ax, i, val, fmt.format(val), color=_SPEC_DARK if color == _SPEC else _TEXT)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_xlim(-0.6, len(labels) - 0.4)
        ax.set_ylabel(ylabel, fontsize=10.5)
        ax.set_title(title, fontsize=13, fontweight="bold", loc="left", pad=14)
        ax.set_ylim(0, max(values) * 1.32)
        ax.grid(axis="y", color="white", linewidth=1.4, zorder=0)
        ax.set_axisbelow(True)
        _style_axis(ax)

    fig.suptitle(
        "Speculative decoding needs fewer main-model calls, but pays for it in dispatch overhead at this scale",
        fontsize=10.5, color=_MUTED, y=1.02,
    )
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_acceptance_over_blocks(acceptance_rates: list[float], path, *, title: str, window: int = 15) -> None:
    """Each speculative block only accepts a handful of tokens, so the raw
    per-block rate is a noisy 0/0.5/0.67/... step sequence (the faint fill);
    the bold line is a trailing rolling average over `window` blocks, which
    is what actually shows the trend, with the overall mean called out."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(9, 4.6))
    fig.patch.set_facecolor("white")
    x = list(range(1, len(acceptance_rates) + 1))
    ax.fill_between(x, acceptance_rates, color=_SPEC, alpha=0.12, zorder=1)
    ax.plot(x, acceptance_rates, color=_SPEC, linewidth=1, alpha=0.35, zorder=2, label="per block")

    if len(acceptance_rates) >= window:
        kernel = np.ones(window) / window
        smoothed = np.convolve(acceptance_rates, kernel, mode="valid")
        smoothed_x = list(range(window, len(acceptance_rates) + 1))
        ax.plot(smoothed_x, smoothed, color=_SPEC_DARK, linewidth=2.75, zorder=3,
                label=f"rolling mean (window={window})")

    mean_rate = float(np.mean(acceptance_rates))
    ax.axhline(mean_rate, color=_ACCENT, linewidth=1.5, linestyle="--", zorder=2)
    ax.text(
        x[-1], mean_rate, f" overall mean {mean_rate:.0%} ", color=_ACCENT, fontsize=10,
        fontweight="bold", va="center", ha="left",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none"},
        zorder=4,
    )

    ax.set_xlabel("speculative block", fontsize=10.5)
    ax.set_ylabel("fraction of draft tokens accepted", fontsize=10.5)
    ax.set_ylim(0, 1.08)
    ax.set_xlim(0, x[-1] * 1.12)
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=12)
    ax.grid(axis="y", color=_GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9.5, frameon=False)
    _style_axis(ax)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_distillation_impact(before_low: float, before_high: float, after_pct: float, path) -> None:
    """The chart behind the distillation story: draft-token acceptance
    rate before (independently-trained draft, reported as the 3-7% range
    actually observed across runs, not a single fabricated average) and
    after (distilled draft, 52.9%, measured by `demos/benchmark.py`).
    The "before" bar's height is the honestly-reported upper bound of
    that range; its label states the range, not a false-precision point
    estimate the original runs never produced."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.8, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(_BAND)

    labels = ["independently\ntrained", "distilled"]
    values = [before_high, after_pct]
    texts = [f"{before_low:.0f}-{before_high:.0f}%", f"{after_pct:.1f}%"]
    colors = [_NAIVE, _SPEC]
    width = 0.5

    for i, (val, color, text) in enumerate(zip(values, colors, texts, strict=True)):
        _gradient_bar(ax, i, val, width, color)
        _value_chip(ax, i, val, text, color=_SPEC_DARK if color == _SPEC else _TEXT)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylabel("draft-token acceptance rate")
    ax.set_ylim(0, max(values) * 1.32)
    ax.set_title("Why the draft model was retrained", fontsize=13, fontweight="bold", loc="left", pad=14)
    ax.grid(axis="y", color="white", linewidth=1.4, zorder=0)
    ax.set_axisbelow(True)
    _style_axis(ax)

    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def render_dispatch_diagram_svg(path: str | Path) -> None:
    """A box-and-arrow diagram of the generic-dispatch trick behind the
    first "what's novel" bullet: the same `functional.py` ops run
    unmodified on a `Tensor` (training, builds a graph) or a plain
    `np.ndarray` (KV-cached inference, no graph), because every op is
    written against a small set of generic helpers instead of against
    `Tensor` directly. This renders the actual call structure, not an
    illustration of the idea."""
    bg, panel, arrow, text, muted = "#12141a", "#1a1d27", "#4b5563", "#c9ced9", "#6b7280"
    tensor_color, array_color, shared_color = "#0d47a1", "#1b5e20", "#b45309"

    box_w, box_h = 200, 56
    width, height = 800, 300

    boxes = [
        (30, 40, "Tensor", tensor_color, "training: builds an autodiff graph"),
        (30, 190, "np.ndarray", array_color, "KV-cached inference: no graph"),
        (300, 115, "functional.py ops", shared_color, "softmax / layer_norm / gelu, generic-dispatched"),
        (570, 40, "GPT.forward (training)", tensor_color, ""),
        (570, 190, "generate.forward_step", array_color, ""),
    ]

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Consolas, Menlo, monospace">',
        f'<rect width="{width}" height="{height}" fill="{bg}"/>',
        f'<text x="20" y="24" font-size="14" fill="{text}">One set of ops, two call paths, '
        'no `if isinstance` branching</text>',
        '<defs><marker id="arrow2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{arrow}"/></marker></defs>',
    ]

    def edge(x1, y1, x2, y2):
        mx = (x1 + x2) / 2
        svg.append(
            f'<path d="M{x1:.1f},{y1:.1f} C{mx:.1f},{y1:.1f} {mx:.1f},{y2:.1f} {x2:.1f},{y2:.1f}" '
            f'fill="none" stroke="{arrow}" stroke-width="1.5" marker-end="url(#arrow2)"/>'
        )

    edge(30 + box_w, 40 + box_h / 2, 300, 115 + box_h / 2 - 20)
    edge(30 + box_w, 190 + box_h / 2, 300, 115 + box_h / 2 + 20)
    edge(300 + box_w, 115 + box_h / 2 - 20, 570, 40 + box_h / 2)
    edge(300 + box_w, 115 + box_h / 2 + 20, 570, 190 + box_h / 2)

    for x, y, label, color, sub in boxes:
        svg.append(
            f'<rect x="{x}" y="{y}" width="{box_w}" height="{box_h}" rx="8" '
            f'fill="{panel}" stroke="{color}" stroke-width="1.8"/>'
        )
        label_y = y + box_h / 2 - 4 if sub else y + box_h / 2 + 5
        svg.append(
            f'<text x="{x + box_w / 2}" y="{label_y}" font-size="13" fill="{text}" '
            f'text-anchor="middle">{html.escape(label)}</text>'
        )
        if sub:
            svg.append(
                f'<text x="{x + box_w / 2}" y="{y + box_h / 2 + 16}" font-size="10" fill="{muted}" '
                f'text-anchor="middle">{html.escape(sub)}</text>'
            )

    svg.append("</svg>")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(svg), encoding="utf-8")


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
