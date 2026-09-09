"""One-hidden-layer tanh MLP with disjoint input-group proximal penalties.

Mean BCE + ridge/2 * (||W1||_F^2 + ||W2||_2^2)
         + group_reg * sum(weight[g] * ||W1[group[g], :]||_F).

All layers are trained. There are no gates or input bypasses. Prices can enter
through group weights; the norm penalty is not an acquisition invoice. NumPy
backpropagation keeps this small mechanism experiment dependency-light.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import expit


@dataclass
class MLPParameters:
    # W1 is stored as (input features, hidden units).
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray

    def arrays(self):
        return (self.W1, self.b1, self.W2, self.b2)

    def copy(self):
        return MLPParameters(*(a.copy() for a in self.arrays()))

    def decision_function(self, X):
        return np.tanh(np.asarray(X) @ self.W1 + self.b1) @ self.W2 + self.b2[0]

    def predict_proba(self, X):
        return expit(self.decision_function(X))


@dataclass
class GroupMLPFit:
    params: MLPParameters
    objective: float
    stationarity: float
    n_iter: int
    converged: bool
    step_size: float
    history: np.ndarray


def initialize_mlp(n_features, hidden, seed):
    rng = np.random.default_rng(seed)
    return MLPParameters(
        rng.normal(0, np.sqrt(2 / (n_features + hidden)), (n_features, hidden)),
        rng.normal(0, .1, hidden),
        rng.normal(0, np.sqrt(2 / (hidden + 1)), hidden), np.zeros(1))


def smooth_loss_grad(X, y, params, ridge):
    """Stable BCE and analytical backprop, including ridge on both matrices."""
    h = np.tanh(X @ params.W1 + params.b1)
    logits = h @ params.W2 + params.b2[0]
    error = (expit(logits) - y) / len(y)
    hidden_error = error[:, None] * params.W2 * (1 - h * h)
    loss = np.mean(np.logaddexp(0, logits) - y * logits)
    loss += ridge / 2 * (np.sum(params.W1 ** 2) + np.sum(params.W2 ** 2))
    gradient = MLPParameters(X.T @ hidden_error + ridge * params.W1,
                             hidden_error.sum(axis=0),
                             h.T @ error + ridge * params.W2,
                             np.array([error.sum()]))
    return float(loss), gradient


def proximal_groups(W1, groups, penalties):
    """Apply Frobenius soft-thresholding to complete input-to-hidden blocks."""
    result = W1.copy()
    for idx, threshold in zip(groups, penalties):
        norm = np.linalg.norm(result[idx, :])
        result[idx, :] *= max(0., 1 - threshold / norm) if norm else 0.
    return result


def group_norms(params, groups):
    return np.array([np.linalg.norm(params.W1[g, :]) for g in groups])


def fit_group_mlp(X, y, groups, group_reg, *, weights=None, hidden=24,
                  ridge=1e-3, seed=0, max_iter=1500, tol=1e-4,
                  step_size=1., max_step=5., initial=None, allowed_features=None):
    """Full-batch proximal gradient with a smooth-loss majorization line search.

    The objective is nonconvex. ``converged`` only means the proximal-gradient
    mapping norm met ``tol``; it is never a global-optimality certificate.
    Fixed-iteration fits are explicitly reported as unconverged. The optional
    allowed_features projection prevents deleted inputs re-entering a refit.
    """
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    if X.ndim != 2 or min(X.shape) < 1 or y.shape != (len(X),):
        raise ValueError("X must be a nonempty matrix and y a matching vector")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Inputs must be finite; preprocessing belongs to training data")
    if not np.array_equal(np.unique(y), [0, 1]):
        raise ValueError("Both binary classes are required")
    if not np.isfinite([group_reg, ridge, tol, step_size, max_step]).all():
        raise ValueError("Hyperparameters must be finite")
    if group_reg < 0 or ridge <= 0 or tol <= 0 or not 0 < step_size <= max_step:
        raise ValueError("Require nonnegative group_reg and positive ridge/tolerance/steps")
    if not isinstance(hidden, (int, np.integer)) or hidden < 1 or max_iter < 1:
        raise ValueError("Need positive hidden width and iteration limit")
    checked, used = [], set()
    for group in groups:
        idx = np.asarray(group)
        if idx.ndim != 1 or not len(idx) or idx.dtype.kind not in "iu":
            raise ValueError("Each group must have integer column indices")
        if (idx < 0).any() or (idx >= X.shape[1]).any():
            raise ValueError("Group column outside X")
        if len(set(idx.tolist())) != len(idx) or used.intersection(idx.tolist()):
            raise ValueError("Overlapping/duplicate group columns are unsupported")
        used.update(idx.tolist())
        checked.append(idx.astype(int))
    groups = checked
    weights = np.sqrt([len(g) for g in groups]) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != (len(groups),) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Need one finite nonnegative weight per group")
    allowed = np.ones(X.shape[1], dtype=bool)
    if allowed_features is not None:
        idx = np.asarray(allowed_features)
        if idx.ndim != 1 or (idx.size and idx.dtype.kind not in "iu"):
            raise ValueError("Allowed features must be integer indices")
        idx = idx.astype(int)
        if (idx < 0).any() or (idx >= X.shape[1]).any() or len(set(idx.tolist())) != len(idx):
            raise ValueError("Invalid allowed features")
        allowed[:] = False
        allowed[idx] = True
    # Eliminate hidden-value influence even on gradients/preprocessing-free refits.
    X = X.copy()
    X[:, ~allowed] = 0.
    params = initialize_mlp(X.shape[1], hidden, seed) if initial is None else initial.copy()
    expected = ((X.shape[1], hidden), (hidden,), (hidden,), (1,))
    if any(a.shape != shape or not np.isfinite(a).all() for a, shape in zip(params.arrays(), expected)):
        raise ValueError("Initial parameter shapes/values do not match the network")
    params.W1[~allowed] = 0.
    penalty = group_reg * weights
    smooth, grad = smooth_loss_grad(X, y, params, ridge)
    objective = smooth + penalty @ group_norms(params, groups)
    history = [float(objective)]
    step = step_size
    mapping = float("inf")
    converged = False
    for iteration in range(1, max_iter + 1):
        trial_step = min(max_step, step * 1.1)
        for _ in range(60):
            proposed = MLPParameters(*(a - trial_step * b for a, b in zip(params.arrays(), grad.arrays())))
            proposed.W1 = proximal_groups(proposed.W1, groups, trial_step * penalty)
            proposed.W1[~allowed] = 0.
            delta = [a - b for a, b in zip(proposed.arrays(), params.arrays())]
            squared = sum(float(np.sum(d * d)) for d in delta)
            linear = sum(float(np.sum(g * d)) for g, d in zip(grad.arrays(), delta))
            new_smooth, new_grad = smooth_loss_grad(X, y, proposed, ridge)
            if new_smooth <= smooth + linear + squared / (2 * trial_step) + 1e-12:
                break
            trial_step /= 2
        else:
            raise RuntimeError("MLP line search failed to find a finite descent step")
        new_objective = new_smooth + penalty @ group_norms(proposed, groups)
        if not np.isfinite(new_objective) or new_objective > objective + 1e-9:
            raise RuntimeError("Nonfinite or increasing proximal objective")
        params, smooth, grad, objective = proposed, new_smooth, new_grad, new_objective
        step = trial_step
        history.append(float(objective))
        # Evaluate mapping at the returned iterate, not the previous one.
        mapped = MLPParameters(*(a - step * b for a, b in zip(params.arrays(), grad.arrays())))
        mapped.W1 = proximal_groups(mapped.W1, groups, step * penalty)
        mapped.W1[~allowed] = 0.
        mapping = np.sqrt(sum(np.sum((a - b) ** 2) for a, b in zip(params.arrays(), mapped.arrays()))) / step
        if mapping <= tol:
            converged = True
            break
    return GroupMLPFit(params, float(objective), float(mapping), iteration,
                       converged, step, np.asarray(history))
