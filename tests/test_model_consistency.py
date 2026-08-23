"""The test that matters most for the inference engine: the KV-cached,
plain-NumPy incremental forward pass in ``generate.py`` must produce exactly
the logits a from-scratch full-sequence forward pass over the same tokens
would, since that's the whole premise a KV cache rests on. Checked two ways:
one token appended at a time, and a batch of several new tokens verified at
once (the shape speculative decoding's "verify" step actually uses).
"""

import numpy as np

from specdec.generate import KVCache, forward_step
from specdec.model import GPT, GPTConfig


def _make_model():
    cfg = GPTConfig(vocab_size=24, d_model=16, n_heads=2, n_layers=2, max_seq_len=32)
    model = GPT(cfg, seed=3)
    return model, cfg, model.export_weights()


def test_incremental_one_token_at_a_time_matches_full_sequence_forward():
    model, cfg, weights = _make_model()
    ids = [1, 5, 3, 7, 2, 9]

    full_logits = model.forward(np.array([ids])).data[0]  # (T, vocab)

    cache = KVCache(cfg.n_layers)
    incremental_logits = []
    for tok in ids:
        step_logits = forward_step(weights, cfg, np.array([tok]), cache)
        incremental_logits.append(step_logits[0])
    incremental_logits = np.stack(incremental_logits)

    assert np.allclose(full_logits, incremental_logits, atol=1e-8)


def test_batched_verify_step_matches_full_sequence_forward():
    model, cfg, weights = _make_model()
    prompt = [4, 4, 8]
    draft_block = [1, 2, 3, 4]  # verified together, as speculative decoding does

    full_logits = model.forward(np.array([prompt + draft_block])).data[0]

    cache = KVCache(cfg.n_layers)
    forward_step(weights, cfg, np.array(prompt), cache)
    verify_logits = forward_step(weights, cfg, np.array(draft_block), cache)

    # verify_logits[i] corresponds to sequence position len(prompt)+i
    expected = full_logits[len(prompt) : len(prompt) + len(draft_block)]
    assert np.allclose(verify_logits, expected, atol=1e-8)


def test_cache_length_tracks_total_tokens_processed():
    _, cfg, weights = _make_model()
    cache = KVCache(cfg.n_layers)
    forward_step(weights, cfg, np.array([1, 2, 3]), cache)
    assert cache.length == 3
    forward_step(weights, cfg, np.array([4, 5]), cache)
    assert cache.length == 5
