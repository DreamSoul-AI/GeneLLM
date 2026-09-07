"""Reproducible synthetic, disjoint-group pilot; no patient data or real prices.

Run from the project root:
  python -m run_experiment.run_group_lasso_pilot --output results/group_lasso_pilot_20260907

Group Lasso creates candidate supports on training data. Every support is
refitted by the same ridge-logistic procedure. Validation BCE + alpha * actual
cost chooses a support. Test data is evaluated only after all choices are made.
An enumerated-subset comparator uses exactly the same fitted candidate models.
"""

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
import sklearn
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from medfs.selectors.group_lasso import fit_group_logistic


TASKS = ("synthetic_cheap_signal", "synthetic_expensive_signal", "synthetic_free_signal")
COSTS = np.array([10., 20., 35., 60., 90., 140.])
GROUPS = [list(range(2 + 3 * g, 5 + 3 * g)) for g in range(6)]
FREE = [0, 1]
TRUTH = {TASKS[0]: [0, 1], TASKS[1]: [2, 4], TASKS[2]: []}
ALPHAS = [0., .00025, .0005, .001, .002, .005, .01, .02]
RIDGE = 1e-4


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def plan_details(selected, groups=GROUPS, costs=COSTS, free=FREE):
    """Deduplicate acquired features; charge each actually selected check once."""
    selected = tuple(sorted(set(selected)))
    features = sorted(set(free).union(*(set(groups[g]) for g in selected)))
    return features, float(sum(costs[g] for g in selected))


def select_candidate(candidates, alpha):
    # Test fields intentionally play no part in selection.
    return min(candidates, key=lambda row: (
        row["val_bce"] + alpha * row["cost"], row["cost"], row["mask_id"]
    ))


def make_data(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 20))
    for group in GROUPS:
        common = rng.normal(size=(n, 1))
        X[:, group] = np.sqrt(.75) * X[:, group] + .5 * common
    coefficients = np.zeros((3, 20))
    coefficients[0, 0] = .6
    coefficients[0, GROUPS[0]] = [1.6, -1.2, .8]
    coefficients[0, GROUPS[1]] = [1., .6, -.4]
    coefficients[1, 0] = .5
    coefficients[1, GROUPS[2]] = [.9, -.7, .5]
    coefficients[1, GROUPS[4]] = [1.5, .8, -.6]
    coefficients[2, FREE] = [1.4, -1.2]
    labels = [rng.binomial(1, expit(-.4 + X @ coef)) for coef in coefficients]
    return X, np.asarray(labels).T, coefficients


def fit_ridge(X, y):
    """Independent smooth fit; mean BCE + fixed ridge, unpenalized intercept."""
    def objective(v):
        logits = v[0] + X @ v[1:]
        err = expit(logits) - y
        loss = np.mean(np.logaddexp(0, logits) - y * logits)
        loss += RIDGE * (v[1:] @ v[1:]) / 2
        grad = np.r_[err.mean(), X.T @ err / len(y) + RIDGE * v[1:]]
        return loss, grad

    initial = np.zeros(X.shape[1] + 1)
    initial[0] = np.log(y.mean() / (1 - y.mean()))
    result = minimize(objective, initial, jac=True, method="L-BFGS-B",
                      options={"gtol": 1e-9, "ftol": 1e-14, "maxiter": 2000})
    gradient = float(np.linalg.norm(objective(result.x)[1], ord=np.inf))
    if not result.success or gradient > 2e-6:
        raise RuntimeError(f"Ridge refit failed: {result.message}; gradient={gradient}")
    return result.x, gradient


def metrics(y, logits):
    return {"bce": float(np.mean(np.logaddexp(0, logits) - y * logits)),
            "auroc": float(roc_auc_score(y, logits)),
            "ap": float(average_precision_score(y, logits))}


def run_one(X, y, task, seed, output, n_lambdas):
    all_idx = np.arange(len(y))
    train_idx, remaining = train_test_split(all_idx, test_size=.4, random_state=seed, stratify=y)
    val_idx, test_idx = train_test_split(remaining, test_size=.5, random_state=seed + 101,
                                        stratify=y[remaining])
    scaler = StandardScaler().fit(X[train_idx])
    train, val = scaler.transform(X[train_idx]), scaler.transform(X[val_idx])
    y_train, y_val = y[train_idx], y[val_idx]
    free_model, _ = fit_ridge(train[:, FREE], y_train)
    free_logits = free_model[0] + train[:, FREE] @ free_model[1:]
    null_grad = train.T @ (expit(free_logits) - y_train) / len(y_train)
    weights = np.sqrt([len(g) for g in GROUPS])
    lambda_max = max(np.linalg.norm(null_grad[g]) / w for g, w in zip(GROUPS, weights))
    lambdas = np.r_[lambda_max * np.geomspace(1.05, .005, n_lambdas), 0.]
    path_rows, path_coefs, path_intercepts = [], [], []
    # Boundary plans are explicitly supplied, not attributed to the path.
    path_masks = {0, 2 ** len(GROUPS) - 1}
    fitted = None
    for lam in lambdas:
        fitted = fit_group_logistic(train, y_train, GROUPS, float(lam), ridge=RIDGE,
                                    weights=weights, initial=fitted, tol=1e-7)
        if not fitted.converged:
            raise RuntimeError(f"Unconverged path fit: {task}, seed={seed}, lambda={lam}")
        selected = [g for g, idx in enumerate(GROUPS) if np.linalg.norm(fitted.coef[idx]) > 1e-6]
        mask_id = sum(1 << g for g in selected)
        path_masks.add(mask_id)
        _, cost = plan_details(selected)
        path_rows.append({"task": task, "seed": seed, "lambda": float(lam),
                          "lambda_max_fraction": float(lam / lambda_max),
                          "mask_id": mask_id, "selected_groups": json.dumps(selected),
                          "cost": cost, "objective": fitted.objective,
                          "kkt_residual": fitted.kkt_residual, "iterations": fitted.n_iter,
                          "converged": fitted.converged})
        path_coefs.append(fitted.coef)
        path_intercepts.append(fitted.intercept)

    candidates, model_params = [], []
    for mask_id in range(2 ** len(GROUPS)):
        selected = [g for g in range(len(GROUPS)) if mask_id & (1 << g)]
        features, cost = plan_details(selected)
        params, grad = fit_ridge(train[:, features], y_train)
        full_params = np.zeros(X.shape[1] + 1)
        full_params[0] = params[0]
        full_params[1 + np.asarray(features)] = params[1:]
        model_params.append(full_params)
        scores = metrics(y_val, params[0] + val[:, features] @ params[1:])
        candidates.append({"task": task, "seed": seed, "mask_id": mask_id,
                           "selected_groups": json.dumps(selected),
                           "selected_features": json.dumps(features),
                           "n_groups": len(selected), "n_features": len(features),
                           "cost": cost, "in_path_with_boundaries": mask_id in path_masks,
                           "refit_gradient_inf": grad,
                           **{"val_" + key: value for key, value in scores.items()}})

    path_candidates = [c for c in candidates if c["mask_id"] in path_masks]
    # Freeze every decision before transforming or scoring the test data.
    choices = [(method, alpha, select_candidate(pool, alpha))
               for alpha in ALPHAS
               for method, pool in [("group_lasso_path_refit", path_candidates),
                                    ("enumerated_refit", candidates)]]
    choices += [("free_only", None, candidates[0]), ("all_checks", None, candidates[-1])]
    test = scaler.transform(X[test_idx])
    selected_rows = []
    for method, alpha, chosen in choices:
        params = model_params[chosen["mask_id"]]
        scores = metrics(y[test_idx], params[0] + test @ params[1:])
        selected_rows.append({**chosen, "method": method, "alpha": alpha,
                              "validation_objective": None if alpha is None else chosen["val_bce"] + alpha * chosen["cost"],
                              "truth_groups": json.dumps(TRUTH[task]),
                              **{"test_" + key: value for key, value in scores.items()}})

    np.savez_compressed(output / f"models_{task}_{seed}.npz", refit_params=np.asarray(model_params),
                        path_coefs=path_coefs, path_intercepts=path_intercepts, lambdas=lambdas,
                        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
                        scaler_mean=scaler.mean_, scaler_scale=scaler.scale_)
    return path_rows, candidates, selected_rows


def aggregate(rows):
    aggregated = []
    keys = sorted({(r["task"], r["method"], -1 if r["alpha"] is None else r["alpha"]) for r in rows})
    for task, method, alpha_key in keys:
        alpha = None if alpha_key == -1 else alpha_key
        group = [r for r in rows if (r["task"], r["method"], r["alpha"]) == (task, method, alpha)]
        result = {"task": task, "method": method, "alpha": alpha, "repeats": len(group)}
        for field in ["cost", "n_groups", "n_features", "test_bce", "test_auroc", "test_ap"]:
            result[field + "_mean"] = float(np.mean([r[field] for r in group]))
            result[field + "_sd"] = float(np.std([r[field] for r in group], ddof=1)) if len(group) > 1 else 0.
        aggregated.append(result)
    return aggregated


def make_plot(output, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey="row")
    fig.subplots_adjust(left=.075, right=.985, bottom=.15, top=.86, hspace=.32, wspace=.18)
    titles = ["A: signal in cheap checks", "B: signal in expensive checks", "C: free information sufficient"]
    for col, (task, title) in enumerate(zip(TASKS, titles)):
        for row, metric in enumerate(["test_auroc", "test_bce"]):
            ax = axes[row, col]
            for method, color, label in [("group_lasso_path_refit", "#1565C0", "Group Lasso path + refit"),
                                          ("enumerated_refit", "#EF6C00", "All 64 subsets + refit")]:
                points = sorted([r for r in summary if r["task"] == task and r["method"] == method],
                                key=lambda r: r["alpha"])
                style = "o-" if method == "group_lasso_path_refit" else "x--"
                ax.plot([r["cost_mean"] for r in points], [r[metric + "_mean"] for r in points],
                        style, color=color, label=label, alpha=.85, markersize=5)
            for method, marker, label in [("all_checks", "*", "All checks"), ("free_only", "s", "Free only")]:
                r = next(r for r in summary if r["task"] == task and r["method"] == method)
                ax.scatter(r["cost_mean"], r[metric + "_mean"], marker=marker,
                           s=85, color="#424242", label=label, zorder=5)
            ax.set_xlabel("Cost per person (proxy units)")
            if col == 0:
                ax.set_ylabel("Test AUROC (higher is better)" if row == 0 else "Test BCE (lower is better)")
            ax.grid(alpha=.2)
            if row == 0:
                ax.set_title(title)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .035), ncol=4, frameon=False)
    fig.suptitle("Synthetic Group Lasso pilot | means across repeated splits\n"
                 "Each point selected on validation data at a preset alpha; no test-based selection", fontsize=12, y=.97)
    fig.savefig(output / "cost_performance.png", dpi=180)
    fig.savefig(output / "cost_performance.svg")
    plt.close(fig)


def write_report(output, summary, path, selected, manifest):
    lines = ["# Group Lasso 合成实验报告", "",
             "本轮是合成数据、非重叠检查组及临时费用的机制实验，不是真实疾病或真实医疗价格结果。", "",
             "## 实验设置", "",
             f"每个任务 {manifest['samples_per_task']} 人；每次按类别分层划分 60% 训练、20% 验证、20% 测试。",
             f"随机种子：{manifest['seeds']}。这是重复留出实验，不是交叉验证或独立队列复现。", "",
             "20 个特征：2 个免费基础特征、6 个付费检查各覆盖 3 个特征。检查价格依次为 10、20、35、60、90、140 proxy_unit，总价 355。",
             "任务 A 的信号来自免费信息及 g0、g1（费用 30）；任务 B 来自免费信息及 g2、g4（费用 125）；任务 C 只来自免费信息。", "",
             "Group Lasso 使用组权重 sqrt(组大小)、训练侧标准化、平均 BCE、固定 ridge=1e-4；截距不惩罚，基础信息不受组惩罚。",
             "λ 从训练侧基础模型导出的 λ_max 的 1.05 倍到 0.005 倍，再加入 0。活动组容差为 1e-6。",
             "路径支持集加上基础信息和全检查边界候选；各候选统一用训练数据重新拟合 ridge logistic 模型。费用仅参与验证方案选择，没有伪装成 Group Lasso 系数拟合时的实际账单。", "",
             f"预设 α 网格：{ALPHAS}。验证数据按 BCE + α×Cost 选方案，所有决定完成后才评估测试数据。α=0 是无费用偏好的消融。",
             "枚举参照覆盖全部 64 套检查组合，使用完全相同的重拟合模型；它是有限候选内的验证最优参照，不是总体风险最优或真实疾病最优证明。", "",
             "## 结果", "",
             "下表均为重复实验的测试指标与费用均值；完整 α 网格及标准差见 summary.csv。仅展示预先设定网格的 0、0.001、0.005，不据测试成绩推荐 α。", "",
             "| 任务 | 方法 | α | 费用 | 检查数 | 测试 BCE | 测试 AUROC | 测试 AP |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    names = dict(zip(TASKS, ["A 便宜信号", "B 较贵信号", "C 免费信号"]))
    for task in TASKS:
        for r in summary:
            if r["task"] != task or (r["alpha"] is not None and r["alpha"] not in [0., .001, .005]):
                continue
            name = {"group_lasso_path_refit": "Group Lasso + 重拟合", "enumerated_refit": "64 组合枚举", "all_checks": "全部检查", "free_only": "仅免费信息"}[r["method"]]
            alpha = "—" if r["alpha"] is None else f"{r['alpha']:g}"
            lines.append(f"| {names[task]} | {name} | {alpha} | {r['cost_mean']:.1f} | {r['n_groups_mean']:.1f} | {r['test_bce_mean']:.4f} | {r['test_auroc_mean']:.4f} | {r['test_ap_mean']:.4f} |")
    gaps = []
    matched = 0
    for r in selected:
        if r["method"] == "group_lasso_path_refit":
            other = next(o for o in selected if o["method"] == "enumerated_refit" and
                         (o["task"], o["seed"], o["alpha"]) == (r["task"], r["seed"], r["alpha"]))
            gaps.append(r["validation_objective"] - other["validation_objective"])
            matched += r["mask_id"] == other["mask_id"]
    lines += ["", "![费用与预测表现](cost_performance.png)", "", "## 数值验证与适用范围", "",
              f"- 路径模型共 {len(path)} 个，全部达到 KKT 残差 ≤1e-7；最大残差 {max(r['kkt_residual'] for r in path):.3g}。",
              f"- 与枚举参照在 {len(gaps)} 个任务/种子/α 配置中有 {matched} 个选择同一方案；路径的最大验证目标差为 {max(gaps):.6f}。",
              "- 有限 λ 路径可能漏掉有价值的组合；真实费用影响最终方案选择，不保证候选生成已经充分考虑价格。",
              "- 本轮没有实现重叠 Group Lasso、真实数据字段映射或组门控。重叠收费只做独立的覆盖/计费逻辑测试，不作为重叠模型已实现的证据。",
              "- 合成任务的关系是线性 logistic 生成；本轮不能回答非线性互补信号、真实疾病准确率或医院价格下的节省幅度。",
              "- 数学定义文档保持不变。", "", "## 复现与文件", "",
              "从项目根目录运行；输出目录必须为空，下面使用新的目录以保留既有结果：", "", "```powershell",
              f".venv\\Scripts\\python.exe -m run_experiment.run_group_lasso_pilot --output {output.with_name(output.name + '_rerun').as_posix()} --samples {manifest['samples_per_task']} --seeds " + " ".join(map(str, manifest['seeds'])) + f" --lambdas {manifest['lambda_count_positive']}",
              "```", "", "- manifest.json：环境版本、配置和代码哈希。",
              "- path.csv：λ 路径、活动检查组、KKT 与迭代次数。",
              "- candidates.csv：全部 64 组合的验证得分；不包含全候选测试排名。",
              "- selected.csv / summary.csv：已选方案的逐次结果、重复实验均值与标准差。",
              "- models_*.npz：模型参数、训练侧缩放参数和样本划分。",
              "- synthetic_*.npz：本轮合成数据及真实生成系数。",
              "- cost_performance.png / .svg：独立可导出的结果图。"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=3600)
    parser.add_argument("--seeds", type=int, nargs="+", default=[51966, 51967, 51968])
    parser.add_argument("--lambdas", type=int, default=40)
    args = parser.parse_args()
    if args.samples < 100 or args.lambdas < 2 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Need >=100 samples, >=2 positive lambdas, and unique seeds")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory is nonempty; choose a new path to preserve existing results")
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    manifest = {"data_kind": "synthetic_only", "cost_unit": "proxy_unit", "samples_per_task": args.samples,
                "seeds": args.seeds, "lambda_count_positive": args.lambdas, "alphas": ALPHAS,
                "ridge": RIDGE, "group_weights": "sqrt(group_size)", "groups": GROUPS,
                "costs": COSTS.tolist(), "free_features": FREE, "truth_groups": TRUTH,
                "split": [0.6, 0.2, 0.2], "activity_tolerance": 1e-6, "kkt_tolerance": 1e-7,
                "python": platform.python_version(), "numpy": np.__version__,
                "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                "test_selection": False, "cost_in_candidate_generation": False}
    code_paths = [Path(__file__), Path("medfs/selectors/group_lasso.py"),
                  Path("run_experiment/cost_aware_formulation_v1.md")]
    manifest["sha256"] = {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths}
    path_rows, candidates, selected = [], [], []
    with threadpool_limits(limits=1):
        for seed in args.seeds:
            X, labels, truth = make_data(args.samples, seed)
            np.savez_compressed(args.output / f"synthetic_{seed}.npz", X=X, labels=labels, truth_coef=truth)
            for index, task in enumerate(TASKS):
                rows = run_one(X, labels[:, index], task, seed, args.output, args.lambdas)
                path_rows.extend(rows[0])
                candidates.extend(rows[1])
                selected.extend(rows[2])
                print(f"Completed {task}, seed={seed}, path supports={len(set(r['mask_id'] for r in rows[0]))}", flush=True)
        summary = aggregate(selected)
        write_csv(args.output / "path.csv", path_rows)
        write_csv(args.output / "candidates.csv", candidates)
        write_csv(args.output / "selected.csv", selected)
        write_csv(args.output / "summary.csv", summary)
        make_plot(args.output, summary)
    manifest["elapsed_seconds"] = time.perf_counter() - start
    write_json(args.output / "manifest.json", manifest)
    write_report(args.output, summary, path_rows, selected, manifest)
    print(f"Saved report: {args.output / 'report.md'} ({manifest['elapsed_seconds']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
