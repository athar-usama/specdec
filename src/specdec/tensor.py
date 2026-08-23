"""A small NumPy-array reverse-mode autodiff engine.

This is the tensor-shaped counterpart to a scalar autograd (the same idea as
project 1's ``hypergrad``, at a different granularity): a computation graph
of ``Tensor`` nodes, each carrying a ``_backward`` closure, with ``.backward()``
doing a topological sort and running those closures in reverse. Every op
composes correctly for free once its own local gradient rule is right, so
only the primitives below need to be checked against finite differences
(``tests/test_tensor.py`` does exactly that). Everything built out of them
(layer norm, softmax, attention, cross-entropy) inherits correctness rather
than needing its own hand-derived backward pass.

``Tensor`` deliberately implements the same operator surface NumPy arrays
already have (``+ - * @``, ``.reshape``, ``.transpose``, indexing). That means
the model in ``nn.py``/``model.py`` is written once, generically, and runs
two ways: with ``Tensor`` operands during training (builds a graph, supports
``.backward()``), and with plain ``np.ndarray`` operands during incremental
inference (no graph, no overhead, just NumPy), the same pattern project 1
used to make one autodiff core serve two different composition needs.
"""

from __future__ import annotations

import numpy as np


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum ``grad`` down to ``shape`` along whatever axes NumPy broadcast."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    __slots__ = ("data", "grad", "requires_grad", "_backward", "_prev", "_op")

    def __init__(self, data, requires_grad: bool = False, _children: tuple = (), _op: str = ""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad: np.ndarray | None = None
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    # -- basics -----------------------------------------------------------
    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def zero_grad(self) -> None:
        self.grad = np.zeros_like(self.data)

    def backward(self, grad: np.ndarray | None = None) -> None:
        topo: list[Tensor] = []
        visited = set()

        def build(v: Tensor) -> None:
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        grad = np.ones_like(self.data) if grad is None else grad
        self.grad = grad if self.grad is None else self.grad + grad
        for v in reversed(topo):
            v._backward()

    @staticmethod
    def _ensure(x) -> Tensor:
        return x if isinstance(x, Tensor) else Tensor(x)

    def _needs_grad(self, *others: Tensor) -> bool:
        return self.requires_grad or any(o.requires_grad for o in others)

    # -- elementwise arithmetic --------------------------------------------
    def __add__(self, other):
        other = Tensor._ensure(other)
        out = Tensor(self.data + other.data, self._needs_grad(other), (self, other), "+")

        def _backward():
            if self.requires_grad:
                g = _unbroadcast(out.grad, self.data.shape)
                self.grad = g if self.grad is None else self.grad + g
            if other.requires_grad:
                g = _unbroadcast(out.grad, other.data.shape)
                other.grad = g if other.grad is None else other.grad + g

        out._backward = _backward
        return out

    __radd__ = __add__

    def __neg__(self):
        out = Tensor(-self.data, self.requires_grad, (self,), "neg")

        def _backward():
            if self.requires_grad:
                g = -out.grad
                self.grad = g if self.grad is None else self.grad + g

        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (-Tensor._ensure(other))

    def __rsub__(self, other):
        return Tensor._ensure(other) + (-self)

    def __mul__(self, other):
        other = Tensor._ensure(other)
        out = Tensor(self.data * other.data, self._needs_grad(other), (self, other), "*")

        def _backward():
            if self.requires_grad:
                g = _unbroadcast(out.grad * other.data, self.data.shape)
                self.grad = g if self.grad is None else self.grad + g
            if other.requires_grad:
                g = _unbroadcast(out.grad * self.data, other.data.shape)
                other.grad = g if other.grad is None else other.grad + g

        out._backward = _backward
        return out

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self * Tensor._ensure(other) ** -1.0

    def __rtruediv__(self, other):
        return Tensor._ensure(other) * self**-1.0

    def __pow__(self, power: float):
        out = Tensor(self.data**power, self.requires_grad, (self,), f"**{power}")

        def _backward():
            if self.requires_grad:
                g = out.grad * power * self.data ** (power - 1)
                self.grad = g if self.grad is None else self.grad + g

        out._backward = _backward
        return out

    # -- matrix ops ---------------------------------------------------------
    def __matmul__(self, other):
        other = Tensor._ensure(other)
        out = Tensor(self.data @ other.data, self._needs_grad(other), (self, other), "@")

        def _backward():
            # NumPy's `@` broadcasts leading (batch) dimensions the same way
            # `+`/`*` do, so gradients need the same _unbroadcast reduction
            # before being assigned back (e.g. a (4,8,16) @ (16,20): batched
            # activations against an un-batched weight matrix).
            if self.requires_grad:
                g = out.grad @ np.swapaxes(other.data, -1, -2)
                g = _unbroadcast(g, self.data.shape)
                self.grad = g if self.grad is None else self.grad + g
            if other.requires_grad:
                g = np.swapaxes(self.data, -1, -2) @ out.grad
                g = _unbroadcast(g, other.data.shape)
                other.grad = g if other.grad is None else other.grad + g

        out._backward = _backward
        return out

    def transpose(self, *axes: int):
        axes = axes or tuple(reversed(range(self.data.ndim)))
        out = Tensor(self.data.transpose(*axes), self.requires_grad, (self,), "transpose")
        inv = np.argsort(axes)

        def _backward():
            if self.requires_grad:
                g = out.grad.transpose(*inv)
                self.grad = g if self.grad is None else self.grad + g

        out._backward = _backward
        return out

    def reshape(self, *shape: int):
        out = Tensor(self.data.reshape(*shape), self.requires_grad, (self,), "reshape")
        orig_shape = self.data.shape

        def _backward():
            if self.requires_grad:
                g = out.grad.reshape(orig_shape)
                self.grad = g if self.grad is None else self.grad + g

        out._backward = _backward
        return out

    # -- reductions ---------------------------------------------------------
    def sum(self, axis=None, keepdims: bool = False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), self.requires_grad, (self,), "sum")
        in_shape = self.data.shape

        def _backward():
            if self.requires_grad:
                g = out.grad
                if not keepdims and axis is not None:
                    g = np.expand_dims(g, axis)
                g = np.broadcast_to(g, in_shape)
                self.grad = g if self.grad is None else self.grad + g

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims: bool = False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def max(self, axis=None, keepdims: bool = False):
        out_data = self.data.max(axis=axis, keepdims=True)
        out = Tensor(
            out_data if keepdims else out_data.squeeze(axis=axis) if axis is not None else out_data.reshape(()),
            self.requires_grad,
            (self,),
            "max",
        )
        mask = (self.data == out_data).astype(np.float64)
        mask = mask / mask.sum(axis=axis, keepdims=True)  # split gradient evenly across ties

        def _backward():
            if self.requires_grad:
                g = out.grad
                if not keepdims and axis is not None:
                    g = np.expand_dims(g, axis)
                g = g * mask
                self.grad = g if self.grad is None else self.grad + g

        out._backward = _backward
        return out

    # -- elementwise functions ----------------------------------------------
    def exp(self):
        e = np.exp(self.data)
        out = Tensor(e, self.requires_grad, (self,), "exp")

        def _backward():
            if self.requires_grad:
                g = out.grad * e
                self.grad = g if self.grad is None else self.grad + g

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), self.requires_grad, (self,), "log")

        def _backward():
            if self.requires_grad:
                g = out.grad / self.data
                self.grad = g if self.grad is None else self.grad + g

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, self.requires_grad, (self,), "tanh")

        def _backward():
            if self.requires_grad:
                g = out.grad * (1 - t * t)
                self.grad = g if self.grad is None else self.grad + g

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(self.data, 0.0), self.requires_grad, (self,), "relu")

        def _backward():
            if self.requires_grad:
                g = out.grad * (self.data > 0)
                self.grad = g if self.grad is None else self.grad + g

        out._backward = _backward
        return out

    # -- indexing / embedding lookup -----------------------------------------
    def __getitem__(self, idx):
        out = Tensor(self.data[idx], self.requires_grad, (self,), "getitem")

        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                np.add.at(self.grad, idx, out.grad)

        out._backward = _backward
        return out

    def masked_fill(self, mask: np.ndarray, value: float):
        """Set positions where ``mask`` is True to ``value`` (no gradient
        flows through the filled positions, matching PyTorch's semantics)."""
        out_data = np.where(mask, value, self.data)
        out = Tensor(out_data, self.requires_grad, (self,), "masked_fill")

        def _backward():
            if self.requires_grad:
                g = np.where(mask, 0.0, out.grad)
                self.grad = g if self.grad is None else self.grad + g

        out._backward = _backward
        return out

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad})"
