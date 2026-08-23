"""End-to-end checks using real (tiny) trained-enough-to-run models, on top
of the isolated correctness proofs in ``test_speculative.py`` and
``test_model_consistency.py``.
"""

import numpy as np

from specdec.generate import generate_naive, generate_speculative
from specdec.model import GPT, GPTConfig


def _tiny_model(seed):
    cfg = GPTConfig(vocab_size=16, d_model=8, n_heads=2, n_layers=2, max_seq_len=64)
    return GPT(cfg, seed=seed), cfg


def test_generate_naive_produces_requested_token_count():
    model, cfg = _tiny_model(0)
    weights = model.export_weights()
    rng = np.random.default_rng(0)
    ids = generate_naive(weights, cfg, prompt_ids=[1, 2, 3], n_tokens=15, rng=rng)
    assert len(ids) == 3 + 15


def test_speculative_produces_requested_token_count():
    draft, cfg_d = _tiny_model(1)
    main, cfg_m = _tiny_model(2)
    rng = np.random.default_rng(0)
    ids, stats = generate_speculative(
        draft.export_weights(), cfg_d, main.export_weights(), cfg_m,
        prompt_ids=[1, 2, 3], n_tokens=20, rng=rng, lookahead=4,
    )
    assert len(ids) == 3 + 20
    assert stats.proposed >= stats.accepted >= 0


def test_using_the_same_model_as_draft_and_main_accepts_every_token():
    # If draft and main are literally the same weights, verification should
    # reproduce the proposal exactly (same forward_step, same math), so the
    # acceptance rate should be effectively 100%.
    model, cfg = _tiny_model(0)
    weights = model.export_weights()
    rng = np.random.default_rng(0)
    ids, stats = generate_speculative(
        weights, cfg, weights, cfg,
        prompt_ids=[1, 2, 3], n_tokens=40, rng=rng, lookahead=5,
    )
    assert len(ids) == 3 + 40
    assert stats.accepted == stats.proposed
