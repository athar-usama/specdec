"""Runs speculative decoding and prints it live, color-coded by how each
token was produced, then saves the same transcript as an SVG for the README.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..checkpoint import load_weights
from ..generate import generate_speculative
from ..model import GPTConfig
from ..tokenizer import BPETokenizer
from ..viz import render_transcript_svg

ROOT = Path(__file__).resolve().parent.parent.parent.parent
CHECKPOINT_DIR = ROOT / "checkpoints"
ASSETS_DIR = ROOT / "assets"

_ANSI = {
    "accepted": "\033[42m\033[30m",  # green background
    "corrected": "\033[41m\033[97m",  # red background
    "bonus": "\033[44m\033[97m",  # blue background
    "prompt": "\033[2m",  # dim
}
_RESET = "\033[0m"


def _weights_with_config(path: Path) -> tuple[dict, GPTConfig]:
    weights = load_weights(path)
    return weights, weights["config"]


def main(
    prompt: str = "The lighthouse at Cold Point",
    n_tokens: int = 130,
    lookahead: int = 3,
    temperature: float = 0.5,
    seed: int = 1,
) -> None:
    tok = BPETokenizer.load(CHECKPOINT_DIR / "tokenizer.pkl")
    draft_weights, draft_cfg = _weights_with_config(CHECKPOINT_DIR / "draft.pkl")
    main_weights, main_cfg = _weights_with_config(CHECKPOINT_DIR / "main.pkl")

    prompt_ids = tok.encode(prompt)
    print(_ANSI["prompt"] + prompt + _RESET, end="", flush=True)

    pieces: list[tuple[str, str]] = [(prompt, "prompt")]

    def on_token(token_id: int, kind: str) -> None:
        text = tok.decode([token_id])
        print(_ANSI[kind] + text + _RESET, end="", flush=True)
        pieces.append((text, kind))

    rng = np.random.default_rng(seed)
    _, stats = generate_speculative(
        draft_weights, draft_cfg, main_weights, main_cfg,
        prompt_ids, n_tokens, rng, lookahead=lookahead, temperature=temperature, on_token=on_token,
    )
    print()
    print()
    acceptance_rate = stats.accepted / max(stats.proposed, 1)
    print(f"proposed {stats.proposed} draft tokens, accepted {stats.accepted} ({acceptance_rate:.1%})")
    print(f"draft forward passes: {stats.draft_forward_calls}   main forward passes: {stats.main_forward_calls}")

    ASSETS_DIR.mkdir(exist_ok=True)
    out_path = ASSETS_DIR / "live_generation.svg"
    render_transcript_svg(
        pieces, out_path,
        title=f"Speculative decoding, live: {acceptance_rate:.0%} of draft tokens accepted",
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
