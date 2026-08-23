import numpy as np
import pytest

from specdec.tensor import Tensor


def fd_grad(f, x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        xp, xm = x.copy(), x.copy()
        xp[idx] += h
        xm[idx] -= h
        grad[idx] = (f(xp) - f(xm)) / (2 * h)
    return grad


def analytic_grad(expr, x: np.ndarray) -> np.ndarray:
    t = Tensor(x, requires_grad=True)
    out = expr(t)
    out.backward(np.ones_like(out.data))
    return t.grad


rng = np.random.default_rng(0)
A = rng.normal(size=(3, 4))
B = rng.normal(size=(4, 3))
POS = np.abs(rng.normal(size=(3, 4))) + 0.5  # for log/pow with fractional exponents


@pytest.mark.parametrize(
    "expr,x",
    [
        (lambda t: t + 2.0, A),
        (lambda t: t * 3.0, A),
        (lambda t: t - 1.5, A),
        (lambda t: -t, A),
        (lambda t: t**3, A),
        (lambda t: t.exp(), A),
        (lambda t: t.log(), POS),
        (lambda t: t.tanh(), A),
        (lambda t: t.relu(), A),
        (lambda t: t.sum(axis=1, keepdims=True), A),
        (lambda t: t.mean(axis=0), A),
        (lambda t: t.max(axis=1, keepdims=True), A),
        (lambda t: t.reshape(2, 6), A),
        (lambda t: t.transpose(), A),
    ],
)
def test_primitive_gradient_matches_finite_difference(expr, x):
    def f(arr):
        return float(expr(Tensor(arr)).data.sum())

    analytic = analytic_grad(expr, x)
    numeric = fd_grad(f, x)
    assert np.allclose(analytic, numeric, atol=1e-4, rtol=1e-3)


def test_matmul_gradient_matches_finite_difference():
    def f_a(a):
        return float((Tensor(a) @ Tensor(B)).data.sum())

    def f_b(b):
        return float((Tensor(A) @ Tensor(b)).data.sum())

    ta, tb = Tensor(A, requires_grad=True), Tensor(B, requires_grad=True)
    out = ta @ tb
    out.backward(np.ones_like(out.data))

    assert np.allclose(ta.grad, fd_grad(f_a, A), atol=1e-4, rtol=1e-3)
    assert np.allclose(tb.grad, fd_grad(f_b, B), atol=1e-4, rtol=1e-3)


def test_matmul_with_broadcast_batch_dimension_reduces_gradient_correctly():
    # (batch, seq, d) @ (d, vocab): `other` has no batch dim of its own, so
    # its gradient must be summed over the batch axis NumPy silently added.
    batch, seq, d, vocab = 3, 4, 5, 6
    x_data = rng.normal(size=(batch, seq, d))
    w_data = rng.normal(size=(d, vocab))

    def f_w(w):
        return float((Tensor(x_data) @ Tensor(w)).data.sum())

    x = Tensor(x_data, requires_grad=True)
    w = Tensor(w_data, requires_grad=True)
    out = x @ w
    assert out.shape == (batch, seq, vocab)
    out.backward(np.ones_like(out.data))

    assert w.grad.shape == w_data.shape
    assert np.allclose(w.grad, fd_grad(f_w, w_data), atol=1e-4, rtol=1e-3)


def test_broadcasting_add_reduces_gradient_correctly():
    x = Tensor(rng.normal(size=(5, 3)), requires_grad=True)
    bias = Tensor(rng.normal(size=(3,)), requires_grad=True)
    out = x + bias
    out.backward(np.ones_like(out.data))
    assert x.grad.shape == (5, 3)
    assert bias.grad.shape == (3,)
    assert np.allclose(bias.grad, np.full(3, 5.0))


def test_getitem_gather_scatters_gradient_with_repeated_indices():
    table = Tensor(rng.normal(size=(5, 2)), requires_grad=True)
    idx = np.array([0, 0, 2])  # index 0 used twice: gradient must accumulate
    out = table[idx]
    out.backward(np.ones_like(out.data))
    assert np.allclose(table.grad[0], [2.0, 2.0])
    assert np.allclose(table.grad[1], [0.0, 0.0])
    assert np.allclose(table.grad[2], [1.0, 1.0])


def test_masked_fill_blocks_gradient_at_masked_positions():
    x = Tensor(rng.normal(size=(2, 2)), requires_grad=True)
    mask = np.array([[False, True], [True, False]])
    out = x.masked_fill(mask, -1.0)
    assert np.array_equal(out.data, np.array([[x.data[0, 0], -1.0], [-1.0, x.data[1, 1]]]))
    out.backward(np.ones_like(out.data))
    assert x.grad[0, 1] == 0.0
    assert x.grad[1, 0] == 0.0
    assert x.grad[0, 0] == 1.0
    assert x.grad[1, 1] == 1.0


def test_backward_accumulates_over_shared_subexpressions():
    x = Tensor(np.array([2.0, 3.0]), requires_grad=True)
    y = x * x + x  # d/dx = 2x + 1
    y.backward(np.ones_like(y.data))
    assert np.allclose(x.grad, 2 * x.data + 1)
