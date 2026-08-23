"""The correctness property speculative decoding depends on: sampling a
token via propose-from-draft + accept-or-resample-against-main must be
*exactly* distributed as the main model alone, for any draft distribution
(Leviathan et al., 2023, "Fast Inference from Transformers via Speculative
Decoding"; the equivalent scheme also appears in Chen et al., 2023). This is
tested directly against ``speculative_accept_or_resample`` with hand-picked
probability vectors, no model involved, so it isolates the algorithm from
any question about whether the trained model itself is any good.
"""

import numpy as np

from specdec.generate import speculative_accept_or_resample


def _empirical_distribution(p_draft, p_main, n_trials, rng):
    vocab = len(p_main)
    counts = np.zeros(vocab)
    for _ in range(n_trials):
        draft_token = rng.choice(vocab, p=p_draft)
        output_token, _ = speculative_accept_or_resample(draft_token, p_draft, p_main, rng)
        counts[output_token] += 1
    return counts / n_trials


def _assert_matches_within_sampling_error(empirical, target, n_trials, n_sigma=5.0, floor=0.01):
    se = np.sqrt(target * (1 - target) / n_trials)
    tol = np.maximum(n_sigma * se, floor)
    assert np.all(np.abs(empirical - target) <= tol), (
        f"empirical={empirical}\ntarget={target}\ntolerance={tol}"
    )


def test_output_distribution_matches_main_model_when_draft_differs_a_lot():
    rng = np.random.default_rng(0)
    p_draft = np.array([0.7, 0.1, 0.1, 0.05, 0.05])
    p_main = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
    n = 40_000
    empirical = _empirical_distribution(p_draft, p_main, n, rng)
    _assert_matches_within_sampling_error(empirical, p_main, n)


def test_output_distribution_matches_main_model_when_draft_is_peaked_elsewhere():
    rng = np.random.default_rng(1)
    p_draft = np.array([0.02, 0.02, 0.9, 0.03, 0.03])
    p_main = np.array([0.25, 0.25, 0.05, 0.25, 0.2])
    n = 40_000
    empirical = _empirical_distribution(p_draft, p_main, n, rng)
    _assert_matches_within_sampling_error(empirical, p_main, n)


def test_identical_draft_and_main_always_accepts():
    rng = np.random.default_rng(2)
    p = np.array([0.2, 0.3, 0.5])
    for _ in range(2000):
        token = rng.choice(3, p=p)
        output, accepted = speculative_accept_or_resample(token, p, p, rng)
        assert accepted
        assert output == token


def test_never_accepts_where_main_assigns_zero_probability():
    rng = np.random.default_rng(3)
    p_draft = np.array([0.5, 0.5, 0.0])
    p_main = np.array([0.5, 0.0, 0.5])
    outputs = []
    for _ in range(5000):
        token = rng.choice(3, p=p_draft)
        output, _ = speculative_accept_or_resample(token, p_draft, p_main, rng)
        outputs.append(output)
    assert 1 not in outputs  # main model assigns it zero probability
    empirical = np.array([outputs.count(i) / len(outputs) for i in range(3)])
    _assert_matches_within_sampling_error(empirical, p_main, len(outputs))
