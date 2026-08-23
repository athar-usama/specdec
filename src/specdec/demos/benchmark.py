"""Measures naive (main-model-only, one token at a time) vs. speculative
(draft+main) decoding throughput on the same trained main model, and plots
the result. These are the numbers the README quotes. This script is what
produced them, not an illustration.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..checkpoint import load_weights
from ..generate import generate_naive, generate_speculative
from ..model import GPTConfig
from ..tokenizer import BPETokenizer
from ..viz import plot_acceptance_over_blocks, plot_decoding_comparison

ROOT = Path(__file__).resolve().parent.parent.parent.parent
CHECKPOINT_DIR = ROOT / "checkpoints"
ASSETS_DIR = ROOT / "assets"

PROMPT = "The lighthouse at Cold Point"
N_TOKENS = 140  # prompt (6 tokens) + this must stay under max_seq_len (160)
LOOKAHEAD = 2
N_TRIALS = 3


def _weights_with_config(path: Path) -> tuple[dict, GPTConfig]:
    weights = load_weights(path)
    return weights, weights["config"]


def _time_it(fn, n_trials: int) -> float:
    best = float("inf")
    for _ in range(n_trials):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    tok = BPETokenizer.load(CHECKPOINT_DIR / "tokenizer.pkl")
    draft_weights, draft_cfg = _weights_with_config(CHECKPOINT_DIR / "draft.pkl")
    main_weights, main_cfg = _weights_with_config(CHECKPOINT_DIR / "main.pkl")
    prompt_ids = tok.encode(PROMPT)

    naive_time = _time_it(
        lambda: generate_naive(main_weights, main_cfg, prompt_ids, N_TOKENS, np.random.default_rng(0)),
        N_TRIALS,
    )
    naive_tps = N_TOKENS / naive_time

    # Per-block acceptance rate = accepted draft proposals / total draft
    # proposals in that block. "corrected" is a rejected proposal, so it
    # counts toward the block's size; "bonus" is not a proposal at all (it's
    # an extra token sampled straight from the main model after a fully
    # accepted block), so it must NOT inflate the denominator, or the
    # per-block rate understates the true accepted/proposed ratio reported
    # in stats (see the "same model" test in test_generate_integration.py
    # for why every accepted token here really was a draft proposal).
    block_sizes: list[int] = []
    block_accepted: list[int] = []

    def on_token(_token_id: int, kind: str) -> None:
        if not block_sizes:
            block_sizes.append(0)
            block_accepted.append(0)
        if kind == "accepted":
            block_sizes[-1] += 1
            block_accepted[-1] += 1
            return
        if kind == "corrected":
            block_sizes[-1] += 1
        block_sizes.append(0)
        block_accepted.append(0)

    stats_holder = {}

    def run_speculative():
        rng = np.random.default_rng(0)
        _, stats = generate_speculative(
            draft_weights, draft_cfg, main_weights, main_cfg,
            prompt_ids, N_TOKENS, rng, lookahead=LOOKAHEAD, on_token=on_token,
        )
        stats_holder["stats"] = stats

    block_sizes.clear()
    block_accepted.clear()
    spec_time = _time_it(run_speculative, N_TRIALS)
    spec_tps = N_TOKENS / spec_time
    stats = stats_holder["stats"]

    acceptance_rates = [a / s for s, a in zip(block_sizes, block_accepted, strict=True) if s > 0]

    print(f"naive decoding      : {naive_tps:6.1f} tok/s  ({naive_time:.2f}s for {N_TOKENS} tokens)")
    print(f"speculative decoding: {spec_tps:6.1f} tok/s  ({spec_time:.2f}s for {N_TOKENS} tokens)")
    print(f"wall-clock speedup: {spec_tps / naive_tps:.2f}x")
    print(f"draft acceptance rate: {stats.accepted}/{stats.proposed} ({stats.accepted / stats.proposed:.1%})")
    print(f"main-model forward passes: naive {N_TOKENS}, speculative {stats.main_forward_calls}")
    print(
        "note: at this model/corpus scale, plain Python and NumPy dispatch overhead per\n"
        "forward_step call dominates over raw FLOPs, so fewer (expensive) main-model calls\n"
        "doesn't translate into a wall-clock win the way it does on GPU-served, FLOP-bound\n"
        "production models. See the README for why."
    )

    ASSETS_DIR.mkdir(exist_ok=True)
    plot_decoding_comparison(
        naive_tps, spec_tps, N_TOKENS, stats.main_forward_calls,
        ASSETS_DIR / "decoding_comparison.png",
        n_tokens=N_TOKENS,
    )
    plot_acceptance_over_blocks(
        acceptance_rates,
        ASSETS_DIR / "acceptance_rate.png",
        title="Draft-token acceptance rate per speculative block",
    )
    print(f"wrote {ASSETS_DIR / 'decoding_comparison.png'}")
    print(f"wrote {ASSETS_DIR / 'acceptance_rate.png'}")


if __name__ == "__main__":
    main()
