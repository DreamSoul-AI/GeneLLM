# 固定 GT 复核：最低费用充分检查集

当前定义：[数学定义 v2](../../run_experiment/cost_aware_formulation_v2.md)。
本报告重新评价已有合成实验输出，不重训、不按 GT 改选模型，也不宣称已经完成新的调参方法。

## 固定答案与评价口径

两个任务的信号特征都是 x2、x5，输入相互独立；它们分别由 g0、g1 独占提供。根据生成条件概率，缺少任一个特征都会丢失信息。两组总价 30，其他检查价格为正，所以最低费用充分方案唯一且固定。
g0、g1 两个套餐提供 6 个指标，其中附带的 4 个无关指标不算额外购买检查。
主评价是是否恢复 {g0,g1}，同时记录遗漏、额外检查和额外费用。费用差仅在方案充分时作为可比较的额外开销；不充分的零费用方案标为失败，不获得省钱奖励。
表中 legacy_alpha 是历史验证评分 BCE+alpha×Cost 的方法参数，不改变 GT；lambda 是训练组惩罚参数。历史分数低不等于满足当前定义。

## 原先展示的参数设置：legacy_alpha=0.001

保留此前展示的参数位置，避免根据本次 GT 复核追选最有利的参数。每个任务有 3 次重复。

| 任务 | 方法 | 完全恢复 GT | 信息充分 | 平均漏选组数 | 平均多选组数 | 平均费用 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| linear_control | logistic_cost_weighted | 3/3 | 3/3 | 0.00 | 0.00 | 30.00 |
| linear_control | mlp_cost_weighted | 3/3 | 3/3 | 0.00 | 0.00 | 30.00 |
| linear_control | mlp_uniform | 2/3 | 3/3 | 0.00 | 0.33 | 50.00 |
| nonlinear_interaction | logistic_cost_weighted | 0/3 | 0/3 | 2.00 | 0.00 | 0.00 |
| nonlinear_interaction | mlp_cost_weighted | 2/3 | 3/3 | 0.00 | 0.33 | 41.67 |
| nonlinear_interaction | mlp_uniform | 1/3 | 3/3 | 0.00 | 0.67 | 61.67 |

成本加权 MLP 的具体结果：

| 任务 | 种子 | 选中检查 | 费用 | 完全恢复 | 遗漏 | 额外检查 |
| --- | ---: | --- | ---: | --- | --- | --- |
| linear_control | 51966 | [0, 1] | 30 | True | [] | [] |
| nonlinear_interaction | 51966 | [0, 1] | 30 | True | [] | [] |
| linear_control | 51967 | [0, 1] | 30 | True | [] | [] |
| nonlinear_interaction | 51967 | [0, 1, 2] | 65 | False | [] | [2] |
| linear_control | 51968 | [0, 1] | 30 | True | [] | [] |
| nonlinear_interaction | 51968 | [0, 1] | 30 | True | [] | [] |

## 候选生成与方案选择的诊断

只检查每次训练保存的候选是否包含 GT，不用 GT 替换历史选择；这用于区分候选缺失与选择规则失误。

| 任务 | 方法 | 候选中存在 GT 的重复次数 |
| --- | --- | ---: |
| linear_control | mlp_uniform | 2/3 |
| linear_control | mlp_cost_weighted | 3/3 |
| linear_control | logistic_cost_weighted | 3/3 |
| nonlinear_interaction | mlp_uniform | 1/3 |
| nonlinear_interaction | mlp_cost_weighted | 2/3 |
| nonlinear_interaction | logistic_cost_weighted | 0/3 |

## 如何解释结果

- 本次成功指检查方案充分且费用最优；不单凭 AUROC 高或费用低下结论。信息充分的判定来自本合成生成器的已知分布，不从模型拟合优劣推断。
- 找回 GT 不等于预测器已训练到最优；原实验的 MLP 驻点与过拟合限制仍然存在。
- 所有 legacy_alpha 设置和边界基线均保存在 selected_gt.csv/summary.csv，失败结果没有删除。不同参数位置不是独立的重复实验，不把它们混合计算总体成功率。
- 只适用于当前有唯一最优方案的两个合成任务。真实数据、重叠获取路线或等价特征需要重新确定充分性与全部等价最优方案，不能直接套一个特征名单。
- 下一步需要设计不使用测试标签或 GT 调参的有限样本方案选择规则；不能继续把加权评分的最低值当作研究目标达成的证明。

## 复现

```powershell
.venv/Scripts/python.exe -m run_experiment.evaluate_group_lasso_fixed_gt --source results/group_lasso_mlp_pilot_20260909 --output results/group_lasso_fixed_gt_rerun
```

本次只增加派生评价文件，原始训练参数、数据、验证选择及测试指标保持不变。
