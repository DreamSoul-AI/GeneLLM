"""Nonlinear mechanism check for a cost-weighted, input-group-sparse MLP.

All data and prices are synthetic. A linear control and a pure interaction use
the same legal features, split, scaling, and labels across compared models.
Costs enter MLP training through group weights, and actual invoices enter
validation selection. No test labels are used by training or model selection.
"""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
import sklearn
from scipy.special import expit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from medfs.selectors.group_lasso import fit_group_logistic
from medfs.selectors.group_lasso_mlp import fit_group_mlp, group_norms
from run_experiment.run_group_lasso_pilot import (
    COSTS, FREE, GROUPS, aggregate, metrics, plan_details, write_csv, write_json,
)

TASKS = ("linear_control", "nonlinear_interaction")
LAMBDAS = [0., .001, .003, .01, .03, .1]
ALPHAS = [0., .0005, .001, .003, .01]
RIDGE = 1e-3


def make_data(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 20))
    a, b = X[:, GROUPS[0][0]], X[:, GROUPS[1][0]]
    logits = np.column_stack((1.7 * a - 1.3 * b, 4. * a * b))
    probabilities = expit(logits)
    labels = rng.binomial(1, probabilities)
    return X, labels, probabilities


def choose(candidates, alpha):
    return min(candidates, key=lambda r: (r["val_bce"] + alpha * r["cost"],
                                          r["cost"], r["candidate_id"]))


def run_one(X, y, task, seed, output, hidden, iterations):
    train_idx, rest = train_test_split(np.arange(len(y)), test_size=.4, random_state=seed, stratify=y)
    val_idx, test_idx = train_test_split(rest, test_size=.5, random_state=seed + 101, stratify=y[rest])
    scaler = StandardScaler().fit(X[train_idx])
    train, val = scaler.transform(X[train_idx]), scaler.transform(X[val_idx])
    y_train, y_val = y[train_idx], y[val_idx]
    models, candidates, traces = {}, [], {}
    arrays = {"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
              "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_}

    def add(name, family, lam, selected, model, fit=None, warmup_iterations=0):
        features, cost = plan_details(selected)
        models[name] = model
        if isinstance(model, np.ndarray):
            logits = model[0] + val @ model[1:]
            arrays[name + "_linear"] = model
        else:
            logits = model.decision_function(val)
            for key, value in zip(("W1", "b1", "W2", "b2"), model.arrays()):
                arrays[name + "_" + key] = value
        if fit is not None:
            traces[name] = fit.history.tolist()
        candidates.append({"task": task, "seed": seed, "candidate_id": name, "family": family,
                           "lambda": lam, "selected_groups": json.dumps(selected),
                           "selected_features": json.dumps(features), "cost": cost,
                           "n_groups": len(selected), "n_features": len(features),
                           "train_objective": None if fit is None else fit.objective,
                           "stationarity": None if fit is None else fit.stationarity,
                           "converged": None if fit is None else fit.converged,
                           "iterations": None if fit is None else fit.n_iter,
                           "warmup_iterations": warmup_iterations,
                           **{"val_" + k: v for k, v in metrics(y_val, logits).items()}})

    # Dense training starts all paths away from the all-zero stationary state,
    # which would otherwise hide signals with no marginal association (XOR).
    dense = fit_group_mlp(train, y_train, GROUPS, 0, hidden=hidden, ridge=RIDGE,
                          seed=seed + 1000, max_iter=iterations)
    add("mlp_all", "mlp_all", 0, list(range(len(GROUPS))), dense.params, dense)
    free = fit_group_mlp(train, y_train, GROUPS, 0, hidden=hidden, ridge=RIDGE,
                         seed=seed + 1000, max_iter=iterations, allowed_features=FREE)
    add("mlp_free", "mlp_free", 0, [], free.params, free)
    for mode in ("uniform", "cost_weighted"):
        weights = np.sqrt([len(g) for g in GROUPS])
        if mode == "cost_weighted":
            weights *= COSTS / np.median(COSTS[COSTS > 0])
        initial = dense.params
        for j, lam in enumerate(LAMBDAS[1:], start=1):
            fit = fit_group_mlp(train, y_train, GROUPS, lam, weights=weights,
                                hidden=hidden, ridge=RIDGE, seed=seed + 1000,
                                max_iter=iterations, initial=initial)
            initial = fit.params
            selected = np.flatnonzero(group_norms(fit.params, GROUPS) > 0).tolist()
            add(f"mlp_{mode}_{j}", f"mlp_{mode}", lam, selected, fit.params, fit,
                warmup_iterations=dense.n_iter)

    for name, features, selected in [("logistic_all", list(range(X.shape[1])), list(range(len(GROUPS)))),
                                      ("logistic_free", FREE, [])]:
        fitted = fit_group_logistic(train[:, features], y_train, [], 0, ridge=RIDGE, tol=1e-7)
        if not fitted.converged:
            raise RuntimeError("Logistic boundary model did not converge")
        v = np.r_[fitted.intercept, fitted.coef]
        params = np.zeros(X.shape[1] + 1)
        params[0], params[1 + np.asarray(features)] = v[0], v[1:]
        add(name, name, 0, selected, params)
    # The logistic comparator uses the same normalized cost weights and lambda
    # grid, and the same all-layer-equivalent coefficient ridge as the MLP.
    weights = np.sqrt([len(g) for g in GROUPS]) * COSTS / np.median(COSTS[COSTS > 0])
    initial = None
    for j, lam in enumerate(LAMBDAS):
        fit = fit_group_logistic(train, y_train, GROUPS, lam, weights=weights,
                                 ridge=RIDGE, initial=initial, tol=1e-7)
        if not fit.converged:
            raise RuntimeError("Logistic comparator did not converge")
        initial = fit
        selected = [g for g, idx in enumerate(GROUPS) if np.linalg.norm(fit.coef[idx]) > 0]
        add(f"logistic_cost_weighted_{j}", "logistic_cost_weighted", lam, selected,
            np.r_[fit.intercept, fit.coef])

    # All candidate selection is frozen before the held-out test matrix/labels
    # enter scoring. There is no logistic refit of selected MLP inputs.
    choices = []
    for method in ("mlp_uniform", "mlp_cost_weighted", "logistic_cost_weighted"):
        bounds = {"mlp_all", "mlp_free"} if method.startswith("mlp") else {"logistic_all", "logistic_free"}
        pool = [r for r in candidates if r["family"] == method or r["family"] in bounds]
        choices.extend((method, alpha, choose(pool, alpha)) for alpha in ALPHAS)
    choices.extend((method, None, next(r for r in candidates if r["family"] == method))
                   for method in ("mlp_all", "mlp_free", "logistic_all", "logistic_free"))
    test = scaler.transform(X[test_idx])
    selected_rows = []
    for method, alpha, row in choices:
        model = models[row["candidate_id"]]
        logits = model[0] + test @ model[1:] if isinstance(model, np.ndarray) else model.decision_function(test)
        selected_rows.append({**row, "method": method, "alpha": alpha,
                              "validation_objective": None if alpha is None else row["val_bce"] + alpha * row["cost"],
                              **{"test_" + k: v for k, v in metrics(y[test_idx], logits).items()}})
    np.savez_compressed(output / f"models_{task}_{seed}.npz", **arrays)
    write_json(output / f"traces_{task}_{seed}.json", traces)
    return candidates, selected_rows


def make_plot(output, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    colors = {"mlp_cost_weighted": "#1261A0", "mlp_uniform": "#E07B39", "logistic_cost_weighted": "#666666"}
    for ax, task in zip(axes, TASKS):
        for method, color in colors.items():
            rows = sorted([r for r in summary if r["task"] == task and r["method"] == method], key=lambda r: r["alpha"])
            ax.plot([r["cost_mean"] for r in rows], [r["test_auroc_mean"] for r in rows], "o-", color=color,
                    label={"mlp_cost_weighted": "Cost-weighted MLP", "mlp_uniform": "Uniform-group MLP",
                           "logistic_cost_weighted": "Cost-weighted logistic"}[method], markersize=5)
        ax.set_title(task.replace("_", " ").capitalize())
        ax.set_xlabel("Cost per person (proxy units)")
        ax.grid(alpha=.2)
        ax.set_ylim(.4, 1)
    axes[0].set_ylabel("Test AUROC (mean across seeds)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, .005))
    fig.suptitle("MLP input-group selection: nonlinear mechanism check\nPreset alpha values; validation selection only", fontsize=12)
    fig.tight_layout(rect=(0, .08, 1, .88))
    fig.savefig(output / "cost_performance.png", dpi=170)
    fig.savefig(output / "cost_performance.svg")
    plt.close(fig)


def write_report(output, summary, candidates, manifest):
    lines = ["# MLP＋成本加权 Group Lasso：非线性机制实验", "",
             "日期：2026-09-09。全部为合成数据和临时价格，不代表真实疾病效果或真实医疗收费。", "",
             "## 模型与实验", "",
             f"主预测器为输入 20 → tanh 隐藏层 {manifest['hidden']} → sigmoid 输出的 MLP，各层参数一起训练。",
             "付费检查对应第一层的完整输入连接块，按 Frobenius 范数组稀疏；整组精确为零时移除检查，无旁路和额外门控。",
             "训练目标：平均 BCE + λ Σ_g w_g ‖W1[G_g,:]‖_F + ridge/2 × (‖W1‖_F² + ‖W2‖₂²)。",
             "成本权重为 sqrt(组大小) × c_g / median(正价格)，普通组惩罚对照仅为 sqrt(组大小)。输入缩放只拟合训练集。",
             f"ridge={RIDGE}，偏置不惩罚；全批量近端梯度和回溯步长。每次最多 {manifest['iterations']} 步，停止残差 1e-4。",
             "各层权重均有 L2 正则，限制输入层缩小、后层放大的参数尺度补偿。MLP 是非凸问题，达到驻点也不是全局最优。", "",
             f"每任务 {manifest['samples']} 条记录，种子 {manifest['seeds']}，按标签分层 60%/20%/20% 训练/验证/测试。",
             "20 个独立标准正态特征；2 个免费特征和 6 个付费组各 3 个特征，价格 10、20、35、60、90、140，总价 355 proxy_unit。",
             "线性对照的真实 logit 为 1.7×x2−1.3×x5；非线性任务为 4×x2×x5，再以 sigmoid 概率抽样标签。",
             "两个任务的信号组均为 g0、g1（费用 30）。交互任务中单个信号特征没有总体边际关联，适合检验组合信号；生成机制仅用于审计，不供训练。",
             "先从随机初始化训练全输入 MLP，再从同一全输入模型分别沿两类递增 λ 路径继续训练。这样避免从全零模型出发漏掉纯交互信号。",
             f"预设 λ 网格为 {LAMBDAS}；α 网格为 {ALPHAS}。初始种子固定，沿路径承接上一 λ 的模型；这是机制设置，未做大规模架构或超参数搜索。",
             "MLP 路径模型直接输出预测，不再改用 logistic 重拟合。免费/全检查 MLP 作为边界参照；logistic 也在相同数据上重新运行。",
             "logistic 路径及免费/全检查 logistic 边界均使用 ridge=1e-3，控制正则系数设置。",
             "费用通过训练权重影响参数和选组；实际账单仍按所选检查全价计。验证侧按 BCE+α×实际费用选择 λ/候选，测试只在全部选择完成后评价。",
             "λ 是代理惩罚强度，α 是实际费用的评价权重，二者没有强行等同；没有费用上限。", "",
             "## 结果", "", "展示预设 α=0.001 及全输入/免费基线；不是按测试结果选择 α。完整均值、标准差见 summary.csv。", "",
             "| 任务 | 方法 | 费用均值 | 测试 BCE | 测试 AUROC | 测试 AP |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in summary:
        if row["alpha"] in (.001, None):
            lines.append(f"| {row['task']} | {row['method']} | {row['cost_mean']:.1f} | {row['test_bce_mean']:.4f} | {row['test_auroc_mean']:.4f} | {row['test_ap_mean']:.4f} |")
    fits = [r for r in candidates if r["stationarity"] is not None]
    lines += ["", "![结果图](cost_performance.png)", "", "## 限制与复现", "",
              f"- {len(fits)} 个 MLP 拟合中 {sum(r['converged'] for r in fits)} 个达到 1e-4 驻点残差阈值；其他拟合用满迭代次数，明确不标作收敛。",
              "- 全输入 MLP 在本配置下出现明显过拟合，不能由非线性表达能力推出泛化必然更好；需同时检查独立测试表现和训练诊断。",
              "- 目标下降轨迹、驻点残差、实际迭代数及各层参数全部保存，不能沿用 logistic 的凸优化保证。",
              "- 未枚举所有 MLP 子集，也未证明选择的是最低实际费用方案。成本组范数只是代理，不能把其数值写成账单。",
              "- 这轮同时检查非线性表达和价格权重影响，不预设成本权重必然优于普通组权重；只报告实际结果。",
              "- 真实受试者数据、真实检查价格映射及重叠检查仍未接入。问题与数学定义文件不变。", "",
              "```powershell",
              f".venv/Scripts/python.exe -m run_experiment.run_group_lasso_mlp_pilot --output results/group_lasso_mlp_pilot_rerun --samples {manifest['samples']} --hidden {manifest['hidden']} --iterations {manifest['iterations']} --seeds " + " ".join(map(str, manifest['seeds'])),
              "```", "", "synthetic_*.npz 保存生成数据；models_*.npz 保存全部模型、缩放参数和划分；traces_*.json 保存训练目标轨迹。",
              "candidates.csv 仅含训练诊断和验证分数；selected.csv/summary.csv 含已冻结选择的测试结果；manifest.json 记录环境和代码哈希。", "",
              "运行 `.venv/Scripts/python.exe -m run_experiment.verify_group_lasso_mlp_pilot results/group_lasso_mlp_pilot_20260909` 可独立复核所有候选的验证分数、选择后的测试分数、收费与输入不变性。", "",
              "方法参考：[Feng & Simon, Sparse-Input Neural Networks](https://arxiv.org/abs/1711.07592)。本实现为检查级、价格加权的组惩罚原型，不声称提出新的神经网络特征选择方法。"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2400)
    parser.add_argument("--hidden", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--seeds", type=int, nargs="+", default=[51966, 51967, 51968])
    args = parser.parse_args()
    if args.samples < 100 or args.hidden < 1 or args.iterations < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Need >=100 samples, positive width/iterations, and unique seeds")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory is nonempty; use a new path")
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    manifest = {"date": "2026-09-09", "data_kind": "synthetic_only", "cost_unit": "proxy_unit",
                "samples": args.samples, "seeds": args.seeds, "hidden": args.hidden, "activation": "tanh",
                "iterations": args.iterations, "ridge_all_weight_matrices": RIDGE,
                "lambdas": LAMBDAS, "alphas": ALPHAS, "groups": GROUPS, "costs": COSTS.tolist(),
                "free_features": FREE, "truth_groups": [0, 1], "test_selection": False,
                "cost_in_mlp_training": True, "group_weight": "sqrt(size) * cost / median(positive costs)",
                "activity": "exact zero after proximal update", "stationarity_tolerance": 1e-4,
                "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                "sklearn": sklearn.__version__, "blas_threads": 1}
    files = ["run_experiment/run_group_lasso_mlp_pilot.py", "medfs/selectors/group_lasso_mlp.py",
             "run_experiment/run_group_lasso_pilot.py", "medfs/selectors/group_lasso.py",
             "run_experiment/cost_aware_formulation_v1.md"]
    manifest["sha256"] = {f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}
    candidates, selected = [], []
    with threadpool_limits(limits=1):
        for seed in args.seeds:
            X, labels, probabilities = make_data(args.samples, seed)
            np.savez_compressed(args.output / f"synthetic_{seed}.npz", X=X, labels=labels, probabilities=probabilities)
            for t, task in enumerate(TASKS):
                c, s = run_one(X, labels[:, t], task, seed, args.output, args.hidden, args.iterations)
                candidates.extend(c)
                selected.extend(s)
                print(f"Completed {task}, seed={seed}; {len(c)} candidates", flush=True)
        summary = aggregate(selected)
        write_csv(args.output / "candidates.csv", candidates)
        write_csv(args.output / "selected.csv", selected)
        write_csv(args.output / "summary.csv", summary)
        make_plot(args.output, summary)
    manifest["elapsed_seconds"] = time.perf_counter() - start
    write_json(args.output / "manifest.json", manifest)
    write_report(args.output, summary, candidates, manifest)
    print(f"Saved {args.output / 'report.md'} ({manifest['elapsed_seconds']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
