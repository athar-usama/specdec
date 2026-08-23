"""Composite ops built out of ``Tensor`` primitives.

None of these implement their own backward pass. They're built purely out
of the primitives in ``tensor.py``, so autodiff composes them correctly
automatically. The only thing worth checking against finite differences is
the primitives themselves (see ``tests/test_tensor.py``); these inherit
correctness for free.

Every function here also runs unmodified when given plain ``np.ndarray``
operands instead of ``Tensor``. NumPy arrays already have ``.max``/``.mean``
with the same ``(axis, keepdims)`` signature ``Tensor`` mirrors, so those
compose for free; ``exp``/``log``/``tanh``/``masked_fill`` aren't ndarray
*methods* the way they're ``Tensor`` methods, so the four dispatch helpers
below pick ``np.exp``/``np.log``/``np.tanh``/``np.where`` when the operand
is a plain array (the same generic-dispatch trick project 1 used to let one
autodiff core wrap arbitrary types). That's what lets ``generate.py``'s
KV-cache inference path reuse this exact code with plain-NumPy operands and
no autodiff graph at all. See ``tests/test_model_consistency.py``.
"""

from __future__ import annotations

import math

import numpy as np

from .tensor import Tensor


def _exp(x):
    return x.exp() if hasattr(x, "exp") else np.exp(x)


def _log(x):
    return x.log() if hasattr(x, "log") else np.log(x)


def _tanh(x):
    return x.tanh() if hasattr(x, "tanh") else np.tanh(x)


def masked_fill(x, mask: np.ndarray, value: float):
    return x.masked_fill(mask, value) if hasattr(x, "masked_fill") else np.where(mask, value, x)


def softmax(x, axis: int = -1):
    m = x.max(axis=axis, keepdims=True)
    e = _exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax(x, axis: int = -1):
    m = x.max(axis=axis, keepdims=True)
    shifted = x - m
    return shifted - _log(_exp(shifted).sum(axis=axis, keepdims=True))


def layer_norm(x, gamma, beta, eps: float = 1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    centered = x - mean
    var = (centered * centered).mean(axis=-1, keepdims=True)
    std = (var + eps) ** 0.5
    return (centered / std) * gamma + beta


_GELU_C = math.sqrt(2.0 / math.pi)


def gelu(x):
    inner = (x + (x**3) * 0.044715) * _GELU_C
    return x * ((_tanh(inner) + 1.0) * 0.5)


def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Mean negative log-likelihood of ``targets`` under ``logits``.

    ``logits`` is a ``Tensor`` of shape ``(N, vocab)``; ``targets`` is a plain
    integer array of shape ``(N,)``.
    """
    logp = log_softmax(logits, axis=-1)
    n = targets.shape[0]
    picked = logp[np.arange(n), targets]
    return -picked.mean()


def soft_cross_entropy(logits: Tensor, target_probs: np.ndarray) -> Tensor:
    """Cross-entropy against a full target *distribution* rather than a
    single hard label: the standard knowledge-distillation loss. Train a
    student's ``logits`` to reproduce a (frozen) teacher's output
    distribution ``target_probs``, both shape ``(N, vocab)``.
    """
    logp = log_softmax(logits, axis=-1)
    return -(logp * target_probs).sum(axis=-1).mean()


def causal_mask(t: int) -> np.ndarray:
    """Boolean ``(t, t)`` array, True at positions that must be masked
    (queries may not attend to keys at a later position)."""
    return np.triu(np.ones((t, t), dtype=bool), k=1)
