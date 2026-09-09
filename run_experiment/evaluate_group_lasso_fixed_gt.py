"""Post-hoc fixed-GT evaluation of the saved two-task synthetic MLP pilot.

No training, parameter selection or test re-ranking occurs here. The source
pilot has independent inputs, signal in x2/x5, unique providers g0/g1, and
positive prices. Those assumptions make its unique minimum-cost sufficient
plan {g0, g1}. This evaluator must not be generalized to arbitrary datasets
with substitutes, overlapping coverage, or multiple optimal plans.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from run_experiment.run_group_lasso_pilot import write_csv, write_json


def score_plan(selected, truth, costs):
    selected, truth = set(selected), set(truth)
    costs = np.asarray(costs, dtype=float)
    if costs.ndim != 1 or not np.isfinite(costs).all() or (costs <= 0).any():
        raise ValueError("This unique-GT evaluator requires strictly positive prices")
    if any(not isinstance(g, (int, np.integer)) or g < 0 or g >= len(costs) for g in selected | truth):
        raise ValueError("Invalid check id")
    missing, extra = sorted(truth - selected), sorted(selected - truth)
    sufficient = not missing
    cost, optimum = sum(costs[g] for g in selected), sum(costs[g] for g in truth)
    return {"gt_groups": json.dumps(sorted(truth)), "gt_cost": float(optimum),
            "information_sufficient": sufficient, "exact_gt": selected == truth,
            "missing_groups": json.dumps(missing), "extra_groups": json.dumps(extra),
            "n_missing": len(missing), "n_extra": len(extra), "actual_cost": float(cost),
            "unnecessary_spend": float(sum(costs[g] for g in extra)),
            # Incomplete plans do not earn a negative 'cost regret' reward.
            "excess_cost_if_sufficient": float(cost - optimum) if sufficient else None}


def evaluate(source, output):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output directory; prior results are immutable")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["sha256"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Historical source hash mismatch: {name}")
    if manifest["data_kind"] != "synthetic_only" or manifest["truth_groups"] != [0, 1]:
        raise ValueError("This audit is specific to the known two-task synthetic generator")
    groups, costs = manifest["groups"], manifest["costs"]
    if groups != [list(range(2 + 3 * g, 5 + 3 * g)) for g in range(6)] or costs != [10., 20., 35., 60., 90., 140.]:
        raise ValueError("Unexpected coverage or prices for the asserted unique GT")
    truth = [0, 1]
    tables = {}
    for filename in ("candidates.csv", "selected.csv"):
        with (source / filename).open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if set(r["task"] for r in rows) != {"linear_control", "nonlinear_interaction"}:
            raise ValueError("Unexpected task: prove its sufficient information GT separately")
        tables[filename] = []
        for row in rows:
            scores = score_plan(json.loads(row["selected_groups"]), truth, costs)
            if float(row["cost"]) != scores["actual_cost"]:
                raise ValueError("Saved invoice does not match the selected checks")
            renamed = {("legacy_alpha" if k == "alpha" else k): v for k, v in row.items()}
            tables[filename].append({**renamed, **scores})

    summary = []
    selected = tables["selected.csv"]
    for task, method, alpha in sorted({(r["task"], r["method"], r["legacy_alpha"]) for r in selected}):
        rows = [r for r in selected if (r["task"], r["method"], r["legacy_alpha"]) == (task, method, alpha)]
        sufficient = [r for r in rows if r["information_sufficient"]]
        summary.append({"task": task, "method": method, "legacy_alpha": alpha,
                        "repeats": len(rows), "exact_gt_count": sum(r["exact_gt"] for r in rows),
                        "exact_gt_rate": float(np.mean([r["exact_gt"] for r in rows])),
                        "sufficient_count": len(sufficient),
                        "mean_missing_groups": float(np.mean([r["n_missing"] for r in rows])),
                        "mean_extra_groups": float(np.mean([r["n_extra"] for r in rows])),
                        "mean_cost": float(np.mean([r["actual_cost"] for r in rows])),
                        "mean_unnecessary_spend": float(np.mean([r["unnecessary_spend"] for r in rows])),
                        "mean_excess_cost_among_sufficient": float(np.mean([r["excess_cost_if_sufficient"] for r in sufficient])) if sufficient else None})
    coverage = []
    candidates = tables["candidates.csv"]
    for task, seed in sorted({(r["task"], r["seed"]) for r in candidates}):
        for family in ("mlp_uniform", "mlp_cost_weighted", "logistic_cost_weighted"):
            bounds = {"mlp_all", "mlp_free"} if family.startswith("mlp") else {"logistic_all", "logistic_free"}
            rows = [r for r in candidates if r["task"] == task and r["seed"] == seed and (r["family"] == family or r["family"] in bounds)]
            coverage.append({"task": task, "seed": seed, "family": family,
                             "candidate_count": len(rows), "contains_exact_gt": any(r["exact_gt"] for r in rows),
                             "sufficient_candidate_count": sum(r["information_sufficient"] for r in rows)})

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "candidate_gt.csv", candidates)
    write_csv(output / "selected_gt.csv", selected)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "candidate_coverage.csv", coverage)
    metadata = {"date": "2026-09-09", "evaluation": "fixed minimum-cost sufficient information GT",
                "source": source.as_posix(), "gt_groups": truth, "gt_cost": 30,
                "gt_signal_features": [2, 5], "acquired_features": sorted(set(groups[0] + groups[1])),
                "training_rerun": False, "gt_used_to_choose_hyperparameters": False,
                "source_sha256": {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                                  for name in ("candidates.csv", "selected.csv")},
                # Git may normalize JSON line endings; hash its canonical content.
                "source_manifest_canonical_json_sha256": hashlib.sha256(
                    json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest(),
                "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "definition_sha256": hashlib.sha256(Path("run_experiment/cost_aware_formulation_v2.md").read_bytes()).hexdigest()}
    write_json(output / "manifest.json", metadata)
    write_report(source, output, summary, selected, coverage)
    return {"candidate_rows": len(candidates), "selected_rows": len(selected),
            "summary_rows": len(summary), "coverage_rows": len(coverage)}


def write_report(source, output, summary, selected, coverage):
    lines = ["# 固定 GT 复核：最低费用充分检查集", "",
             "当前定义：[数学定义 v2](../../run_experiment/cost_aware_formulation_v2.md)。",
             "本报告重新评价已有合成实验输出，不重训、不按 GT 改选模型，也不宣称已经完成新的调参方法。", "",
             "## 固定答案与评价口径", "",
             "两个任务的信号特征都是 x2、x5，输入相互独立；它们分别由 g0、g1 独占提供。根据生成条件概率，缺少任一个特征都会丢失信息。两组总价 30，其他检查价格为正，所以最低费用充分方案唯一且固定。",
             "g0、g1 两个套餐提供 6 个指标，其中附带的 4 个无关指标不算额外购买检查。",
             "主评价是是否恢复 {g0,g1}，同时记录遗漏、额外检查和额外费用。费用差仅在方案充分时作为可比较的额外开销；不充分的零费用方案标为失败，不获得省钱奖励。",
             "表中 legacy_alpha 是历史验证评分 BCE+alpha×Cost 的方法参数，不改变 GT；lambda 是训练组惩罚参数。历史分数低不等于满足当前定义。", "",
             "## 原先展示的参数设置：legacy_alpha=0.001", "",
             "保留此前展示的参数位置，避免根据本次 GT 复核追选最有利的参数。每个任务有 3 次重复。", "",
             "| 任务 | 方法 | 完全恢复 GT | 信息充分 | 平均漏选组数 | 平均多选组数 | 平均费用 |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary:
        if row["legacy_alpha"] == "0.001":
            lines.append(f"| {row['task']} | {row['method']} | {row['exact_gt_count']}/{row['repeats']} | {row['sufficient_count']}/{row['repeats']} | {row['mean_missing_groups']:.2f} | {row['mean_extra_groups']:.2f} | {row['mean_cost']:.2f} |")
    lines += ["", "成本加权 MLP 的具体结果：", "",
              "| 任务 | 种子 | 选中检查 | 费用 | 完全恢复 | 遗漏 | 额外检查 |", "| --- | ---: | --- | ---: | --- | --- | --- |"]
    for row in selected:
        if row["method"] == "mlp_cost_weighted" and row["legacy_alpha"] == "0.001":
            lines.append(f"| {row['task']} | {row['seed']} | {row['selected_groups']} | {row['actual_cost']:g} | {row['exact_gt']} | {row['missing_groups']} | {row['extra_groups']} |")
    lines += ["", "## 候选生成与方案选择的诊断", "",
              "只检查每次训练保存的候选是否包含 GT，不用 GT 替换历史选择；这用于区分候选缺失与选择规则失误。", "",
              "| 任务 | 方法 | 候选中存在 GT 的重复次数 |", "| --- | --- | ---: |"]
    for task in ("linear_control", "nonlinear_interaction"):
        for family in ("mlp_uniform", "mlp_cost_weighted", "logistic_cost_weighted"):
            rows = [r for r in coverage if r["task"] == task and r["family"] == family]
            lines.append(f"| {task} | {family} | {sum(r['contains_exact_gt'] for r in rows)}/{len(rows)} |")
    lines += ["", "## 如何解释结果", "",
              "- 本次成功指检查方案充分且费用最优；不单凭 AUROC 高或费用低下结论。信息充分的判定来自本合成生成器的已知分布，不从模型拟合优劣推断。",
              "- 找回 GT 不等于预测器已训练到最优；原实验的 MLP 驻点与过拟合限制仍然存在。",
              "- 所有 legacy_alpha 设置和边界基线均保存在 selected_gt.csv/summary.csv，失败结果没有删除。不同参数位置不是独立的重复实验，不把它们混合计算总体成功率。",
              "- 只适用于当前有唯一最优方案的两个合成任务。真实数据、重叠获取路线或等价特征需要重新确定充分性与全部等价最优方案，不能直接套一个特征名单。",
              "- 下一步需要设计不使用测试标签或 GT 调参的有限样本方案选择规则；不能继续把加权评分的最低值当作研究目标达成的证明。", "",
              "## 复现", "", "```powershell",
              f".venv/Scripts/python.exe -m run_experiment.evaluate_group_lasso_fixed_gt --source {source.as_posix()} --output results/group_lasso_fixed_gt_rerun",
              "```", "", "本次只增加派生评价文件，原始训练参数、数据、验证选择及测试指标保持不变。"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/group_lasso_mlp_pilot_20260909"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.source, args.output), indent=2))
