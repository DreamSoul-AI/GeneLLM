# Group Lasso：疾病级低成本特征选择

本分支独立保存 Group Lasso 的实现、问题定义、方法说明及合成实验。当前目标是：**对每个疾病，在完整保留疾病预测信息的方案中找到费用最低的检查集，并训练对应预测模型。** 正确答案固定，α 等超参数只影响方法，不改变 GT；不设置费用预算上限。

主预测模型为 **MLP＋成本加权 Group Lasso**：各层共同学习非线性预测关系，检查价格通过第一层输入连接块的组惩罚参与训练。现有代码中的 `BCE + α × Cost` 是历史验证选择启发式，不是研究目标或充分性证明；当前按固定 GT 重新评价其输出。

MLP 的训练目标连续但非凸；组范数是实际费用的代理，实际账单按选中的检查全价计算。logistic 的旧实现与结果保留为线性基线。

本轮全部使用合成数据和临时价格，不包含受试者数据、临床成本工作簿、会议资料或凭据。

## 内容

- [当前问题与数学定义 v2](run_experiment/cost_aware_formulation_v2.md)
- [方法与实验设计](run_experiment/cost_aware_methods_and_experiments.md)
- [MLP Group Lasso 求解器](medfs/selectors/group_lasso_mlp.py)
- [MLP 非线性实验入口](run_experiment/run_group_lasso_mlp_pilot.py)
- [固定 GT 复核报告](results/group_lasso_fixed_gt_20260909/report.md)
- [历史 MLP 训练报告](results/group_lasso_mlp_pilot_20260909/report.md)
- [独立结果核验](run_experiment/verify_group_lasso_mlp_pilot.py)
- [MLP 测试](tests/test_group_lasso_mlp.py)、[logistic 测试](tests/test_group_lasso.py)和 [固定 GT 评价测试](tests/test_fixed_gt_evaluation.py)，共 28 项
- [旧 logistic 求解器](medfs/selectors/group_lasso.py)及 [旧实验报告](results/group_lasso_pilot_20260907/report.md)

## 环境和复现

使用 Python 3.12；下面命令使用 uv，在仓库根目录运行：

```powershell
git clone --branch experiments/group-lasso-pilot --single-branch https://github.com/DreamSoul-AI/GeneLLM.git GeneLLM-group-lasso
cd GeneLLM-group-lasso
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m run_experiment.evaluate_group_lasso_fixed_gt --output results/group_lasso_fixed_gt_rerun
# 复现历史模型训练及其数值核验：
.venv/Scripts/python.exe -m run_experiment.run_group_lasso_mlp_pilot --output results/group_lasso_mlp_pilot_rerun
.venv/Scripts/python.exe -m run_experiment.verify_group_lasso_mlp_pilot results/group_lasso_mlp_pilot_rerun
# 复现历史 logistic 基线：
.venv/Scripts/python.exe -m run_experiment.run_group_lasso_pilot --output results/group_lasso_pilot_rerun
```

Linux/macOS 将 `.venv/Scripts/python.exe` 替换为 `.venv/bin/python`。脚本拒绝覆盖非空结果目录；新实验应使用新的输出路径。

## 当前 MLP 实验

单隐藏层 tanh MLP（20→24→1），每个合成任务 2400 条记录、3 个种子、60%/20%/20% 分层划分。线性对照与双特征乘积任务使用相同的检查结构和价格；各模型在同一批数据和划分上比较。

两个任务的最低费用充分方案均为 **g0＋g1，费用 30**。沿用此前展示的历史 α=0.001，成本加权 MLP 的固定 GT 结果为：

| 任务 | 完全恢复 GT | 信息充分 | 具体问题 |
| --- | ---: | ---: | --- |
| 线性对照 | 3/3 | 3/3 | 三次均选中 g0＋g1。 |
| 非线性交互 | 2/3 | 3/3 | 一次多选 g2，费用为 65。 |

因此非线性任务的平均 AUROC 约 0.88、费用约 41.7，不能替代恢复 GT 的结论。全部历史参数位置的成功与失败均见 [固定 GT 复核](results/group_lasso_fixed_gt_20260909/report.md)；这次没有用 GT 改选参数或重新训练。真实数据中的充分性近似与调参方法仍需设计。

72 个 MLP 拟合中 3 个达到预设 1e-4 驻点残差，其余达到训练步数上限；全输入 MLP 出现过拟合。保存全部训练轨迹及数值诊断，不把固定训练次数标成收敛。当前不包含重叠检查实现或真实受试者数据。

## 历史 logistic 实验（2026-09-07）

三个合成任务各有 3600 条记录，使用 3 个随机种子；每次分层划分 60% 训练、20% 验证、20% 测试。下面展示预设 `α=0.001` 下三次重复的均值：

| 合成场景 | 全检查费用 → 选中费用 | 测试 AUROC：全检查 → 选中方案 |
| --- | ---: | ---: |
| 有效信息来自便宜检查 | 355 → 30 | 0.8878 → 0.8891 |
| 有效信息来自较贵检查 | 355 → 125 | 0.8639 → 0.8658 |
| 免费信息已足够 | 355 → 0 | 0.8421 → 0.8422 |

费用单位为 proxy_unit。369 个路径模型达到 KKT 残差 ≤1e-7；与全部 64 组合的枚举参照相比，72 个配置中 67 个选出相同方案。有限路径会漏掉候选，不能据此宣称全局最低成本或真实疾病诊断效果。

保存的 NPZ 文件全部是该脚本生成的合成数据、模型和数据划分；CSV、PNG/SVG 和 JSON 文件包含结果、配置及核验记录。复现时可核对 `candidates.csv`、`selected.csv`、`summary.csv` 和 `path.csv`。

[数学定义 v1](run_experiment/cost_aware_formulation_v1.md)仅作为历史版本按原字节保留，以维持旧实验来源哈希；当前研究目标和评价口径以 v2 为准。

本分支是一次独立的 Group Lasso 工作快照；GeneLLM 原有分支的代码未纳入本实验。
