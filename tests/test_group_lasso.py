"""Independent optimality and input-safety checks for the pilot solver."""

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.special import expit

from medfs.selectors.group_lasso import fit_group_logistic


def example():
    rng = np.random.default_rng(917)
    X = rng.normal(size=(500, 7))
    y = rng.binomial(1, expit(-0.5 + 1.5 * X[:, 0] - X[:, 1] + 0.4 * X[:, 6]))
    return X, y, [[0, 1], [2, 3], [4, 5]]


def test_zero_group_penalty_matches_independent_scipy_solver():
    X, y, groups = example()
    fit = fit_group_logistic(X, y, groups, 0, ridge=0.01, tol=1e-9)

    def objective(v):
        logits = v[0] + X @ v[1:]
        return np.mean(np.logaddexp(0, logits) - y * logits) + 0.005 * (v[1:] @ v[1:])

    reference = minimize(objective, np.zeros(8), method="BFGS", options={"gtol": 1e-7})
    assert fit.converged
    assert abs(fit.objective - reference.fun) < 1e-8
    np.testing.assert_allclose(np.r_[fit.intercept, fit.coef], reference.x, atol=2e-5)


def test_nonzero_penalty_satisfies_numerically_differenced_kkt():
    X, y, groups = example()
    lam, ridge = 0.04, 0.01
    fit = fit_group_logistic(X, y, groups, lam, ridge=ridge, tol=1e-9)
    assert fit.converged
    params = np.r_[fit.intercept, fit.coef]

    def smooth(v):
        logits = v[0] + X @ v[1:]
        return np.mean(np.logaddexp(0, logits) - y * logits) + ridge * (v[1:] @ v[1:]) / 2

    eps = 1e-5
    grad = np.array([(smooth(params + eps * e) - smooth(params - eps * e)) / (2 * eps)
                     for e in np.eye(len(params))])
    assert np.linalg.norm(grad[[0, 7]]) < 1e-7
    active = []
    for group in groups:
        idx = np.array(group) + 1
        norm = np.linalg.norm(params[idx])
        if norm > 0:
            active.append(group)
            assert np.linalg.norm(grad[idx] + lam * np.sqrt(len(group)) * params[idx] / norm) < 1e-7
        else:
            assert np.linalg.norm(grad[idx]) <= lam * np.sqrt(len(group)) + 1e-7
    assert groups[0] in active
    assert len(active) < len(groups)


def test_large_penalty_removes_paid_groups_but_keeps_free_signal():
    X, y, groups = example()
    fit = fit_group_logistic(X, y, groups, 10)
    assert fit.converged
    np.testing.assert_array_equal(fit.coef[:6], 0)
    assert abs(fit.coef[6]) > 0.1


def test_iteration_limit_is_not_reported_as_convergence():
    X, y, groups = example()
    fit = fit_group_logistic(X, y, groups, 0.01, max_iter=1, tol=1e-12)
    assert not fit.converged


@pytest.mark.parametrize("groups", [[[0, 1], [1, 2]], [[0, 0]], [[7]], [[]]])
def test_invalid_or_overlapping_groups_are_rejected(groups):
    X, y, _ = example()
    with pytest.raises(ValueError):
        fit_group_logistic(X, y, groups, 0.1)


def test_nonfinite_inputs_and_single_class_are_rejected():
    X, y, groups = example()
    with pytest.raises(ValueError):
        fit_group_logistic(X, np.zeros_like(y), groups, 0.1)
    X[0, 0] = np.nan
    with pytest.raises(ValueError):
        fit_group_logistic(X, y, groups, 0.1)


def test_overlap_accounting_deduplicates_features_not_invoices():
    from run_experiment.run_group_lasso_pilot import plan_details

    groups = [[0, 1], [1, 2], [0, 1, 2]]
    features, cost = plan_details([0, 1], groups, [40, 50, 65], [])
    other_features, other_cost = plan_details([2], groups, [40, 50, 65], [])
    assert features == other_features == [0, 1, 2]
    assert cost == 90 and other_cost == 65


def test_selection_uses_validation_and_cost_never_test_scores():
    from run_experiment.run_group_lasso_pilot import select_candidate

    candidates = [{"mask_id": 0, "val_bce": .6, "cost": 0, "test_bce": .01},
                  {"mask_id": 1, "val_bce": .4, "cost": 100, "test_bce": .99}]
    assert select_candidate(candidates, 0)["mask_id"] == 1
    assert select_candidate(candidates, .005)["mask_id"] == 0
    candidates[0]["test_bce"], candidates[1]["test_bce"] = 100, 0
    assert select_candidate(candidates, 0)["mask_id"] == 1
