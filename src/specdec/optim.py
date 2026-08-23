"""Adam over a list of ``Tensor`` parameters (NumPy arrays under the hood,
so this is one array update per parameter rather than one per scalar)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .tensor import Tensor


class Adam:
    def __init__(self, params: Sequence[Tensor], lr: float = 3e-3, betas=(0.9, 0.999), eps: float = 1e-8):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        self.t += 1
        bias1 = 1 - self.b1**self.t
        bias2 = 1 - self.b2**self.t
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            m_hat = self.m[i] / bias1
            v_hat = self.v[i] / bias2
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
