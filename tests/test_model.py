import numpy as np

from specdec.model import GPT, GPTConfig
from specdec.optim import Adam


def test_forward_shapes():
    cfg = GPTConfig(vocab_size=50, d_model=16, n_heads=2, n_layers=2, max_seq_len=32)
    model = GPT(cfg, seed=0)
    idx = np.array([[1, 2, 3, 4, 5]])
    logits = model.forward(idx)
    assert logits.shape == (1, 5, 50)


def test_model_overfits_a_tiny_batch():
    rng = np.random.default_rng(0)
    cfg = GPTConfig(vocab_size=20, d_model=16, n_heads=2, n_layers=2, max_seq_len=16)
    model = GPT(cfg, seed=0)
    opt = Adam(model.parameters(), lr=5e-3)

    idx = rng.integers(0, cfg.vocab_size, size=(4, 8))
    targets = np.roll(idx, -1, axis=1)  # next-token prediction on a fixed random sequence

    losses = []
    for _ in range(200):
        opt.zero_grad()
        loss = model.loss(idx, targets)
        loss.backward(np.array(1.0))
        opt.step()
        losses.append(float(loss.data))

    assert losses[-1] < losses[0] * 0.2, f"loss barely moved: {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_causal_masking_means_future_tokens_cannot_change_earlier_logits():
    cfg = GPTConfig(vocab_size=30, d_model=16, n_heads=2, n_layers=2, max_seq_len=16)
    model = GPT(cfg, seed=0)
    idx = np.array([[3, 7, 1, 9]])
    logits_a = model.forward(idx).data

    idx2 = idx.copy()
    idx2[0, 3] = 15  # change only the last token
    logits_b = model.forward(idx2).data

    # logits at positions 0..2 must be identical: they can't see position 3
    assert np.allclose(logits_a[0, :3], logits_b[0, :3])
    # logits at position 3 (which depends on the changed token as input) may differ
    assert not np.allclose(logits_a[0, 3], logits_b[0, 3])


def test_from_export_reconstructs_an_identical_model():
    cfg = GPTConfig(vocab_size=40, d_model=16, n_heads=2, n_layers=2, max_seq_len=32)
    model = GPT(cfg, seed=5)
    idx = np.array([[1, 2, 3, 4, 5, 6]])
    original_logits = model.forward(idx).data

    reloaded = GPT.from_export(model.export_weights())
    reloaded_logits = reloaded.forward(idx).data

    assert np.allclose(original_logits, reloaded_logits, atol=1e-12)
