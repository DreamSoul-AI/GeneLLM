"""Nonlinear backprop, group removal, and optimization invariants."""

import numpy as np
import pytest

from medfs.selectors.group_lasso_mlp import (
    MLPParameters, fit_group_mlp, initialize_mlp, proximal_groups, smooth_loss_grad,
)


def test_all_layer_backprop_matches_independent_finite_differences():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(13, 4))
    y = rng.integers(0, 2, 13)
    params = initialize_mlp(4, 3, 42)
    shapes = [a.shape for a in params.arrays()]
    sizes = np.cumsum([a.size for a in params.arrays()])
    flat = np.concatenate([a.ravel() for a in params.arrays()])

    def loss(v):
        pieces = np.split(v, sizes[:-1])
        p = MLPParameters(*(a.reshape(s) for a, s in zip(pieces, shapes)))
        # Independent forward loss: avoids reusing smooth_loss_grad's value.
        logits = np.tanh(X @ p.W1 + p.b1) @ p.W2 + p.b2[0]
        return np.mean(np.logaddexp(0, logits) - y * logits) + .07 / 2 * (np.sum(p.W1**2) + np.sum(p.W2**2))

    _, grad = smooth_loss_grad(X, y, params, .07)
    eps = 1e-5
    numerical = np.array([(loss(flat + eps * e) - loss(flat - eps * e)) / (2 * eps)
                          for e in np.eye(len(flat))])
    np.testing.assert_allclose(np.concatenate([a.ravel() for a in grad.arrays()]), numerical, atol=1e-9)


def test_price_threshold_applies_to_the_entire_input_block():
    W = np.array([[3., 0.], [0., 4.], [3., 0.], [0., 4.], [2., 1.]])
    result = proximal_groups(W, [[0, 1], [2, 3]], [1., 6.])
    np.testing.assert_allclose(result[:2], W[:2] * .8)
    np.testing.assert_array_equal(result[2:4], 0.)
    np.testing.assert_array_equal(result[4], W[4])


def test_proximal_objective_descends_and_unbought_values_cannot_affect_prediction():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(100, 5))
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    fit = fit_group_mlp(X, y, [[0, 1], [2, 3]], .005,
                        hidden=8, max_iter=120, allowed_features=[0, 1, 4])
    assert (np.diff(fit.history) <= 1e-9).all()
    assert np.linalg.norm(fit.params.W2 - initialize_mlp(5, 8, 0).W2) > .01
    np.testing.assert_array_equal(fit.params.W1[[2, 3]], 0.)
    changed = X.copy()
    changed[:, [2, 3]] = rng.normal(1e6, 1e4, (100, 2))
    np.testing.assert_array_equal(fit.params.predict_proba(X), fit.params.predict_proba(changed))
    repeat = fit_group_mlp(changed, y, [[0, 1], [2, 3]], .005,
                           hidden=8, max_iter=120, allowed_features=[0, 1, 4])
    for a, b in zip(fit.params.arrays(), repeat.params.arrays()):
        np.testing.assert_array_equal(a, b)


def test_high_penalty_removes_paid_inputs_preserves_free_learning():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(100, 4))
    y = (X[:, 3] > 0).astype(int)
    fit = fit_group_mlp(X, y, [[0, 1], [2]], 10, hidden=8, max_iter=120)
    np.testing.assert_array_equal(fit.params.W1[:3], 0.)
    assert np.linalg.norm(fit.params.W1[3]) > .1
    assert np.mean((fit.params.predict_proba(X) > .5) == y) > .9


def test_iteration_limit_is_not_global_or_stationary_convergence():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(40, 3))
    y = np.tile([0, 1], 20)
    fit = fit_group_mlp(X, y, [[0, 1]], .01, max_iter=1, tol=1e-12)
    assert not fit.converged and fit.n_iter == 1 and fit.stationarity > 1e-12


def test_mlp_learns_xor_without_engineered_interaction_inputs():
    X = np.tile([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]], (20, 1))
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    fit = fit_group_mlp(X, y, [[0], [1]], .001, hidden=8, max_iter=500, seed=42)
    assert np.mean((fit.params.predict_proba(X) > .5) == y) == 1.
    assert all(np.linalg.norm(fit.params.W1[g]) > 0 for g in [0, 1])


@pytest.mark.parametrize("groups", [[[0, 1], [1, 2]], [[0, 0]], [[4]], [[]]])
def test_invalid_groups(groups):
    with pytest.raises(ValueError):
        fit_group_mlp(np.ones((10, 4)), np.tile([0, 1], 5), groups, .1)


def test_no_nan_weights_or_unregularized_layer_rescaling():
    X, y = np.ones((10, 4)), np.tile([0, 1], 5)
    with pytest.raises(ValueError):
        fit_group_mlp(X, y, [[0, 1]], .1, weights=[np.nan])
    with pytest.raises(ValueError):
        fit_group_mlp(X, y, [[0, 1]], .1, ridge=0)
