import numpy as np

from specdec.functional import causal_mask, cross_entropy, gelu, layer_norm, log_softmax, soft_cross_entropy, softmax
from specdec.tensor import Tensor

rng = np.random.default_rng(1)


def fd_grad(f, x, h=1e-6):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        xp, xm = x.copy(), x.copy()
        xp[idx] += h
        xm[idx] -= h
        grad[idx] = (f(xp) - f(xm)) / (2 * h)
    return grad


def test_softmax_sums_to_one_and_matches_finite_difference_gradient():
    x = rng.normal(size=(4, 5))

    def f(arr):
        return float(softmax(Tensor(arr)).data.sum() ** 2)  # nonlinear so grad isn't trivially zero

    probs = softmax(Tensor(x)).data
    assert np.allclose(probs.sum(axis=-1), 1.0)

    t = Tensor(x, requires_grad=True)
    out = (softmax(t).sum() ** 2)
    out.backward(np.array(1.0))
    assert np.allclose(t.grad, fd_grad(f, x), atol=1e-4, rtol=1e-3)


def test_log_softmax_equals_log_of_softmax():
    x = rng.normal(size=(3, 7))
    a = log_softmax(Tensor(x)).data
    b = np.log(softmax(Tensor(x)).data)
    assert np.allclose(a, b, atol=1e-8)


def test_layer_norm_zero_mean_unit_variance_before_affine():
    x = rng.normal(size=(6, 8)) * 5 + 3
    gamma = Tensor(np.ones(8))
    beta = Tensor(np.zeros(8))
    out = layer_norm(Tensor(x), gamma, beta).data
    assert np.allclose(out.mean(axis=-1), 0.0, atol=1e-6)
    assert np.allclose(out.std(axis=-1), 1.0, atol=1e-3)


def test_layer_norm_gradient_matches_finite_difference():
    x = rng.normal(size=(3, 6))
    gamma_data = rng.normal(size=6)
    beta_data = rng.normal(size=6)

    def f(arr):
        return float(layer_norm(Tensor(arr), Tensor(gamma_data), Tensor(beta_data)).data.sum())

    t = Tensor(x, requires_grad=True)
    out = layer_norm(t, Tensor(gamma_data), Tensor(beta_data))
    out.backward(np.ones_like(out.data))
    assert np.allclose(t.grad, fd_grad(f, x), atol=1e-4, rtol=1e-3)


def test_gelu_gradient_matches_finite_difference():
    x = rng.normal(size=(5,))

    def f(arr):
        return float(gelu(Tensor(arr)).data.sum())

    t = Tensor(x, requires_grad=True)
    out = gelu(t)
    out.backward(np.ones_like(out.data))
    assert np.allclose(t.grad, fd_grad(f, x), atol=1e-4, rtol=1e-3)


def test_cross_entropy_matches_manual_nll_and_gradient():
    logits_data = rng.normal(size=(4, 6))
    targets = np.array([0, 5, 2, 2])

    manual_probs = np.exp(logits_data) / np.exp(logits_data).sum(axis=-1, keepdims=True)
    manual_nll = -np.log(manual_probs[np.arange(4), targets]).mean()

    loss = cross_entropy(Tensor(logits_data), targets)
    assert np.isclose(loss.data, manual_nll, atol=1e-8)

    def f(arr):
        return float(cross_entropy(Tensor(arr), targets).data)

    t = Tensor(logits_data, requires_grad=True)
    out = cross_entropy(t, targets)
    out.backward(np.array(1.0))
    assert np.allclose(t.grad, fd_grad(f, logits_data), atol=1e-4, rtol=1e-3)


def test_causal_mask_shape_and_content():
    m = causal_mask(4)
    assert m.shape == (4, 4)
    assert not m[0, 0]
    assert m[0, 1]  # position 0 may not see position 1
    assert not m[3, 0]  # position 3 may see everything before it


def test_soft_cross_entropy_matches_hard_cross_entropy_at_one_hot_targets():
    logits_data = rng.normal(size=(5, 6))
    targets = np.array([1, 0, 5, 2, 2])
    one_hot = np.eye(6)[targets]

    hard = cross_entropy(Tensor(logits_data), targets)
    soft = soft_cross_entropy(Tensor(logits_data), one_hot)
    assert np.isclose(hard.data, soft.data, atol=1e-8)


def test_soft_cross_entropy_gradient_matches_finite_difference():
    logits_data = rng.normal(size=(4, 5))
    target_probs = softmax(Tensor(rng.normal(size=(4, 5)))).data  # some other distribution

    def f(arr):
        return float(soft_cross_entropy(Tensor(arr), target_probs).data)

    t = Tensor(logits_data, requires_grad=True)
    out = soft_cross_entropy(t, target_probs)
    out.backward(np.array(1.0))
    assert np.allclose(t.grad, fd_grad(f, logits_data), atol=1e-4, rtol=1e-3)


def test_soft_cross_entropy_is_minimized_when_logits_match_target_distribution():
    target_probs = np.array([[0.7, 0.2, 0.1]])
    matching_logits = np.log(target_probs)  # softmax(log(p)) == p
    mismatched_logits = np.array([[0.1, 0.7, 0.2]])

    loss_matching = soft_cross_entropy(Tensor(matching_logits), target_probs).data
    loss_mismatched = soft_cross_entropy(Tensor(mismatched_logits), target_probs).data
    assert loss_matching < loss_mismatched
