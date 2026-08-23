"""A small GPT-style decoder-only transformer, and its training-time forward
pass (full sequence, builds a ``Tensor`` autodiff graph).

The incremental, KV-cached inference forward pass lives in ``generate.py``
as a separate, plain-NumPy implementation (no autodiff graph needed once
the weights are trained). See that module's docstring for why, and
``tests/test_model_consistency.py`` for the check that both paths agree
bit-for-bit given the same weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .functional import causal_mask, cross_entropy, gelu, layer_norm, softmax
from .tensor import Tensor


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    max_seq_len: int = 128
    ffn_mult: int = 4

    @property
    def d_head(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads


def _init(rng: np.random.Generator, *shape: int, scale: float) -> Tensor:
    return Tensor(rng.normal(0.0, scale, size=shape), requires_grad=True)


class Block:
    """One transformer block: causal self-attention + feed-forward, each
    with a pre-norm and a residual connection (standard GPT-2 layout)."""

    def __init__(self, cfg: GPTConfig, rng: np.random.Generator):
        d, h = cfg.d_model, cfg.ffn_mult * cfg.d_model
        s1, s2 = 1.0 / np.sqrt(d), 1.0 / np.sqrt(h)
        self.ln1_g, self.ln1_b = Tensor(np.ones(d), True), Tensor(np.zeros(d), True)
        self.ln2_g, self.ln2_b = Tensor(np.ones(d), True), Tensor(np.zeros(d), True)
        self.wq, self.bq = _init(rng, d, d, scale=s1), Tensor(np.zeros(d), True)
        self.wk, self.bk = _init(rng, d, d, scale=s1), Tensor(np.zeros(d), True)
        self.wv, self.bv = _init(rng, d, d, scale=s1), Tensor(np.zeros(d), True)
        self.wo, self.bo = _init(rng, d, d, scale=s1), Tensor(np.zeros(d), True)
        self.w1, self.b1 = _init(rng, d, h, scale=s1), Tensor(np.zeros(h), True)
        self.w2, self.b2 = _init(rng, h, d, scale=s2), Tensor(np.zeros(d), True)
        self.cfg = cfg

    def parameters(self) -> list[Tensor]:
        return [
            self.ln1_g, self.ln1_b, self.ln2_g, self.ln2_b,
            self.wq, self.bq, self.wk, self.bk, self.wv, self.bv, self.wo, self.bo,
            self.w1, self.b1, self.w2, self.b2,
        ]

    def __call__(self, x: Tensor) -> Tensor:
        cfg = self.cfg
        b, t, d = x.shape
        h, dh = cfg.n_heads, cfg.d_head

        normed = layer_norm(x, self.ln1_g, self.ln1_b)
        q = (normed @ self.wq + self.bq).reshape(b, t, h, dh).transpose(0, 2, 1, 3)
        k = (normed @ self.wk + self.bk).reshape(b, t, h, dh).transpose(0, 2, 1, 3)
        v = (normed @ self.wv + self.bv).reshape(b, t, h, dh).transpose(0, 2, 1, 3)

        scores = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / np.sqrt(dh))
        mask = causal_mask(t)
        scores = scores.masked_fill(mask, -1e9)
        attn = softmax(scores, axis=-1) @ v  # (b, h, t, dh)
        attn = attn.transpose(0, 2, 1, 3).reshape(b, t, d)
        x = x + (attn @ self.wo + self.bo)

        normed2 = layer_norm(x, self.ln2_g, self.ln2_b)
        ffn = gelu(normed2 @ self.w1 + self.b1) @ self.w2 + self.b2
        return x + ffn


class GPT:
    def __init__(self, cfg: GPTConfig, seed: int = 0):
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        self.token_emb = _init(rng, cfg.vocab_size, cfg.d_model, scale=0.02)
        self.pos_emb = _init(rng, cfg.max_seq_len, cfg.d_model, scale=0.02)
        self.blocks = [Block(cfg, rng) for _ in range(cfg.n_layers)]
        self.ln_f_g, self.ln_f_b = Tensor(np.ones(cfg.d_model), True), Tensor(np.zeros(cfg.d_model), True)

    def parameters(self) -> list[Tensor]:
        params = [self.token_emb, self.pos_emb, self.ln_f_g, self.ln_f_b]
        for block in self.blocks:
            params += block.parameters()
        return params

    def forward(self, idx: np.ndarray) -> Tensor:
        """``idx``: int array (batch, seq_len). Returns logits (batch, seq_len, vocab)."""
        b, t = idx.shape
        tok = self.token_emb[idx]  # (b, t, d)
        pos = self.pos_emb[np.arange(t)]  # (t, d), broadcasts over batch
        x = tok + pos
        for block in self.blocks:
            x = block(x)
        x = layer_norm(x, self.ln_f_g, self.ln_f_b)
        logits = x @ self.token_emb.transpose()  # weight-tied output projection
        return logits

    def loss(self, idx: np.ndarray, targets: np.ndarray) -> Tensor:
        logits = self.forward(idx)
        b, t, v = logits.shape
        return cross_entropy(logits.reshape(b * t, v), targets.reshape(b * t))

    def export_weights(self) -> dict:
        """Detach every parameter to a plain-NumPy dict for ``generate.py``."""
        blocks = []
        for blk in self.blocks:
            blocks.append(
                {
                    "ln1_g": blk.ln1_g.data, "ln1_b": blk.ln1_b.data,
                    "ln2_g": blk.ln2_g.data, "ln2_b": blk.ln2_b.data,
                    "wq": blk.wq.data, "bq": blk.bq.data,
                    "wk": blk.wk.data, "bk": blk.bk.data,
                    "wv": blk.wv.data, "bv": blk.bv.data,
                    "wo": blk.wo.data, "bo": blk.bo.data,
                    "w1": blk.w1.data, "b1": blk.b1.data,
                    "w2": blk.w2.data, "b2": blk.b2.data,
                }
            )
        return {
            "token_emb": self.token_emb.data,
            "pos_emb": self.pos_emb.data,
            "ln_f_g": self.ln_f_g.data,
            "ln_f_b": self.ln_f_b.data,
            "blocks": blocks,
            "config": self.cfg,
        }

    @classmethod
    def from_export(cls, weights: dict) -> GPT:
        """Rebuild a ``GPT`` (Tensor-parameterized, so it can run through
        ``forward``/``loss`` again) from an ``export_weights()`` dict.
        Parameters are loaded with ``requires_grad=False``. This is meant
        for using a previously-trained model as a frozen teacher (see
        ``demos/train.py``'s distillation step), not for resuming training."""
        model = cls(weights["config"], seed=0)
        model.token_emb = Tensor(weights["token_emb"])
        model.pos_emb = Tensor(weights["pos_emb"])
        model.ln_f_g = Tensor(weights["ln_f_g"])
        model.ln_f_b = Tensor(weights["ln_f_b"])
        for blk, saved in zip(model.blocks, weights["blocks"], strict=True):
            blk.ln1_g, blk.ln1_b = Tensor(saved["ln1_g"]), Tensor(saved["ln1_b"])
            blk.ln2_g, blk.ln2_b = Tensor(saved["ln2_g"]), Tensor(saved["ln2_b"])
            blk.wq, blk.bq = Tensor(saved["wq"]), Tensor(saved["bq"])
            blk.wk, blk.bk = Tensor(saved["wk"]), Tensor(saved["bk"])
            blk.wv, blk.bv = Tensor(saved["wv"]), Tensor(saved["bv"])
            blk.wo, blk.bo = Tensor(saved["wo"]), Tensor(saved["bo"])
            blk.w1, blk.b1 = Tensor(saved["w1"]), Tensor(saved["b1"])
            blk.w2, blk.b2 = Tensor(saved["w2"]), Tensor(saved["b2"])
        return model
