"""Independently replay saved MLP-pilot preprocessing, billing and evaluation."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits


def verify(output):
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for filename, expected in manifest["sha256"].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected, filename
    assert manifest["test_selection"] is False and manifest["cost_in_mlp_training"] is True
    groups, free, costs = manifest["groups"], manifest["free_features"], manifest["costs"]
    candidates = list(csv.DictReader((output / "candidates.csv").open(encoding="utf-8", newline="")))
    selected = list(csv.DictReader((output / "selected.csv").open(encoding="utf-8", newline="")))
    max_val_error = max_test_error = max_objective_error = 0.
    checked = 0
    with threadpool_limits(limits=1):
        for seed in manifest["seeds"]:
            with np.load(output / f"synthetic_{seed}.npz") as data:
                X, labels = data["X"], data["labels"]
            for t, task in enumerate(("linear_control", "nonlinear_interaction")):
                y = labels[:, t]
                with np.load(output / f"models_{task}_{seed}.npz") as store:
                    arrays = {k: store[k] for k in store.files}
                tr, va, te = (arrays[k + "_idx"] for k in ("train", "val", "test"))
                np.testing.assert_array_equal(np.sort(np.r_[tr, va, te]), np.arange(len(y)))
                np.testing.assert_allclose(arrays["scaler_mean"], X[tr].mean(axis=0), atol=1e-14)
                np.testing.assert_allclose(arrays["scaler_scale"], X[tr].std(axis=0), atol=1e-14)
                scaled = (X - arrays["scaler_mean"]) / arrays["scaler_scale"]
                traces = json.loads((output / f"traces_{task}_{seed}.json").read_text())
                rows = [r for r in candidates if r["task"] == task and int(r["seed"]) == seed]

                def forward(name, values):
                    if name + "_linear" in arrays:
                        params = arrays[name + "_linear"]
                        return params[0] + values @ params[1:]
                    return (np.tanh(values @ arrays[name + "_W1"] + arrays[name + "_b1"])
                            @ arrays[name + "_W2"] + arrays[name + "_b2"][0])

                def scores(idx, logits):
                    target = y[idx]
                    return {"bce": float(np.mean(np.logaddexp(0, logits) - target * logits)),
                            "auroc": float(roc_auc_score(target, logits)),
                            "ap": float(average_precision_score(target, logits))}

                for row in rows:
                    assert not any(k.startswith("test_") for k in row)
                    name = row["candidate_id"]
                    chosen = json.loads(row["selected_groups"])
                    features = sorted(set(free).union(*(set(groups[g]) for g in chosen)))
                    assert features == json.loads(row["selected_features"])
                    assert float(row["cost"]) == sum(costs[g] for g in chosen)
                    assert int(row["n_groups"]) == len(chosen) and int(row["n_features"]) == len(features)
                    for k, value in scores(va, forward(name, scaled[va])).items():
                        error = abs(value - float(row["val_" + k]))
                        max_val_error = max(max_val_error, error)
                        assert error < 1e-12
                    unavailable = sorted(set(range(X.shape[1])) - set(features))
                    changed = scaled[va].copy()
                    changed[:, unavailable] = 1e6
                    np.testing.assert_allclose(forward(name, changed), forward(name, scaled[va]), atol=0, rtol=0)
                    if name + "_W1" in arrays:
                        W1, W2 = arrays[name + "_W1"], arrays[name + "_W2"]
                        np.testing.assert_array_equal(W1[unavailable], 0.)
                        norms = np.array([np.linalg.norm(W1[g]) for g in groups])
                        if row["family"] not in ("mlp_all", "mlp_free"):
                            assert np.flatnonzero(norms > 0).tolist() == chosen
                        weights = np.sqrt([len(g) for g in groups])
                        if row["family"] == "mlp_cost_weighted":
                            weights *= np.asarray(costs) / np.median([c for c in costs if c > 0])
                        objective = scores(tr, forward(name, scaled[tr]))["bce"]
                        objective += manifest["ridge_all_weight_matrices"] / 2 * (np.sum(W1**2) + np.sum(W2**2))
                        objective += float(row["lambda"]) * (weights @ norms)
                        error = abs(objective - float(row["train_objective"]))
                        max_objective_error = max(max_objective_error, error)
                        assert error < 1e-12
                        assert np.max(np.diff(traces[name])) <= 1e-9
                        assert abs(traces[name][-1] - objective) < 1e-12
                        assert len(traces[name]) == int(row["iterations"]) + 1
                        assert (row["converged"] == "True") == (float(row["stationarity"]) <= 1e-4)
                    checked += 1

                selections = [r for r in selected if r["task"] == task and int(r["seed"]) == seed]
                for row in selections:
                    method = row["method"]
                    if row["alpha"]:
                        alpha = float(row["alpha"])
                        bounds = {"mlp_all", "mlp_free"} if method.startswith("mlp") else {"logistic_all", "logistic_free"}
                        pool = [r for r in rows if r["family"] == method or r["family"] in bounds]
                        expected = min(pool, key=lambda r: (float(r["val_bce"]) + alpha * float(r["cost"]),
                                                            float(r["cost"]), r["candidate_id"]))
                        assert row["candidate_id"] == expected["candidate_id"]
                        assert abs(float(row["validation_objective"]) - float(row["val_bce"]) - alpha * float(row["cost"])) < 1e-12
                    for k, value in scores(te, forward(row["candidate_id"], scaled[te])).items():
                        error = abs(value - float(row["test_" + k]))
                        max_test_error = max(max_test_error, error)
                        assert error < 1e-12
                for method in ("mlp_uniform", "mlp_cost_weighted", "logistic_cost_weighted"):
                    trajectory = sorted([r for r in selections if r["method"] == method], key=lambda r: float(r["alpha"]))
                    assert (np.diff([float(r["cost"]) for r in trajectory]) <= 0).all()

    result = {"candidate_models_verified": checked, "selected_rows_verified": len(selected),
              "max_validation_metric_error": max_val_error, "max_test_metric_error": max_test_error,
              "max_training_objective_error": max_objective_error,
              "split_scaling_billing_selection_and_input_invariance": "passed",
              "code_hashes": "matched"}
    (output / "validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(json.dumps(verify(parser.parse_args().output), indent=2))
