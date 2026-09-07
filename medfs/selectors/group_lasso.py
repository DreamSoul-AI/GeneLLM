"""Disjoint-group logistic Group Lasso with an unpenalized intercept.

Minimize mean binary cross entropy + ridge/2 * ||coef||^2
         + group_reg * sum(weight[g] * ||coef[group[g]]||_2).

Columns outside the supplied groups are free of the group penalty. The ridge
term still applies to them. Group norms are a selection surrogate, not prices.
This module deliberately does not implement overlapping or stochastic gates.
"""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.special import expit


@dataclass
class GroupLassoFit:
    coef: np.ndarray
    intercept: float
    objective: float
    kkt_residual: float
    n_iter: int
    converged: bool

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return expit(np.asarray(X) @ self.coef + self.intercept)


def fit_group_logistic(
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[Sequence[int]],
    group_reg: float,
    *,
    weights: Sequence[float] | None = None,
    ridge: float = 1e-4,
    tol: float = 1e-7,
    max_iter: int = 10000,
    initial: GroupLassoFit | None = None,
) -> GroupLassoFit:
    """Solve the convex surrogate by proximal gradient, stopping on KKT error.

    The step size uses the global logistic Hessian bound X_aug.T X_aug/(4n)
    plus ridge. Each update applies the group L2 proximal operator. Convergence
    is reported explicitly; callers must inspect it before using a solution.
    """
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    if X.ndim != 2 or y.shape != (X.shape[0],) or len(y) == 0:
        raise ValueError("X must be a nonempty sample matrix and y a matching vector")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Inputs must be finite; impute using training data first")
    if not np.array_equal(np.unique(y), [0, 1]):
        raise ValueError("Both binary classes 0 and 1 are required")
    if not np.isfinite([group_reg, ridge, tol]).all():
        raise ValueError("Regularization and tolerance must be finite")
    if group_reg < 0 or ridge < 0 or tol <= 0 or max_iter < 1:
        raise ValueError("Invalid regularization, tolerance or iteration count")
    group_arrays = []
    used: set[int] = set()
    for group in groups:
        raw = np.asarray(group)
        if raw.ndim != 1 or len(raw) == 0 or raw.dtype.kind not in "iu":
            raise ValueError("Each group must contain integer column indices")
        idx = raw.astype(int)
        if (idx < 0).any() or (idx >= X.shape[1]).any():
            raise ValueError("Group column index out of range")
        if len(set(idx.tolist())) != len(idx) or used.intersection(idx.tolist()):
            raise ValueError("Overlapping groups are not supported by this solver")
        used.update(idx.tolist())
        group_arrays.append(idx)
    if weights is None:
        weights = [np.sqrt(len(g)) for g in group_arrays]
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(group_arrays),) or not np.isfinite(weights).all():
        raise ValueError("One finite weight is required per group")
    if (weights <= 0).any():
        raise ValueError("Group weights must be positive")

    n, p = X.shape
    design = np.column_stack((np.ones(n), X))
    shifted = [g + 1 for g in group_arrays]
    free = np.array([j for j in range(p + 1) if j == 0 or j - 1 not in used])
    step = 1.0 / (np.linalg.eigvalsh(design.T @ design / (4 * n))[-1] + ridge)
    params = np.zeros(p + 1)
    params[0] = np.log(y.mean() / (1 - y.mean()))
    if initial is not None:
        params = np.r_[initial.intercept, initial.coef].astype(float)
        if params.shape != (p + 1,) or not np.isfinite(params).all():
            raise ValueError("Invalid warm start")

    def smooth_grad(value):
        logits = design @ value
        grad = design.T @ (expit(logits) - y) / n
        grad[1:] += ridge * value[1:]
        return grad

    def kkt(value, grad):
        residuals = [float(np.linalg.norm(grad[free]))]
        for idx, weight in zip(shifted, weights):
            norm = np.linalg.norm(value[idx])
            penalty = group_reg * weight
            if norm == 0:
                residuals.append(max(0.0, float(np.linalg.norm(grad[idx]) - penalty)))
            else:
                residuals.append(float(np.linalg.norm(grad[idx] + penalty * value[idx] / norm)))
        return max(residuals)

    grad = smooth_grad(params)
    residual = kkt(params, grad)
    iteration = 0
    for iteration in range(1, max_iter + 1):
        if residual <= tol:
            iteration -= 1
            break
        proposed = params - step * grad
        for idx, weight in zip(shifted, weights):
            norm = np.linalg.norm(proposed[idx])
            threshold = step * group_reg * weight
            proposed[idx] *= max(0.0, 1 - threshold / norm) if norm else 0.0
        params = proposed
        grad = smooth_grad(params)
        residual = kkt(params, grad)

    logits = design @ params
    objective = np.mean(np.logaddexp(0, logits) - y * logits)
    objective += ridge * np.dot(params[1:], params[1:]) / 2
    objective += group_reg * sum(w * np.linalg.norm(params[g]) for g, w in zip(shifted, weights))
    return GroupLassoFit(
        coef=params[1:].copy(), intercept=float(params[0]),
        objective=float(objective), kkt_residual=residual,
        n_iter=iteration, converged=residual <= tol,
    )
