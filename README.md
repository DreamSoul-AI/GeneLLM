# Group Lasso：疾病级低成本特征选择

本分支独立保存 Group Lasso 的实现、问题定义、方法说明及第一轮合成实验。每个疾病学习一套统一检查方案及对应预测模型，以平均 BCE 与实际检查费用的加权和比较方案。

当前实现为非重叠检查组的 logistic Group Lasso。正则化路径产生候选，各候选统一重拟合，再按验证数据上的 `BCE + α × Cost` 选择。费用参与方案选择，尚未直接加入 Group Lasso 的系数拟合项。

本轮全部使用合成数据和临时价格，不包含受试者数据、临床成本工作簿、会议资料或凭据。

## 内容

- [问题与数学定义](run_experiment/cost_aware_formulation_v1.md)
- [方法与实验设计](run_experiment/cost_aware_methods_and_experiments.md)
- [Group Lasso 求解器](medfs/selectors/group_lasso.py)
- [实验入口](run_experiment/run_group_lasso_pilot.py)
- [11 项测试](tests/test_group_lasso.py)
- [完整实验报告](results/group_lasso_pilot_20260907/report.md)
- [结果图](results/group_lasso_pilot_20260907/cost_performance.png)

## 环境和复现

使用 Python 3.12；下面命令使用 uv，在仓库根目录运行：

```powershell
git clone --branch experiments/group-lasso-pilot --single-branch https://github.com/DreamSoul-AI/GeneLLM.git GeneLLM-group-lasso
cd GeneLLM-group-lasso
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m run_experiment.run_group_lasso_pilot --output results/group_lasso_pilot_rerun
```

Linux/macOS 将 `.venv/Scripts/python.exe` 替换为 `.venv/bin/python`。脚本拒绝覆盖非空结果目录；新实验应使用新的输出路径。

## 已保存结果

三个合成任务各有 3600 条记录，使用 3 个随机种子；每次分层划分 60% 训练、20% 验证、20% 测试。下面展示预设 `α=0.001` 下三次重复的均值：

| 合成场景 | 全检查费用 → 选中费用 | 测试 AUROC：全检查 → 选中方案 |
| --- | ---: | ---: |
| 有效信息来自便宜检查 | 355 → 30 | 0.8878 → 0.8891 |
| 有效信息来自较贵检查 | 355 → 125 | 0.8639 → 0.8658 |
| 免费信息已足够 | 355 → 0 | 0.8421 → 0.8422 |

费用单位为 proxy_unit。369 个路径模型达到 KKT 残差 ≤1e-7；与全部 64 组合的枚举参照相比，72 个配置中 67 个选出相同方案。有限路径会漏掉候选，不能据此宣称全局最低成本或真实疾病诊断效果。

保存的 NPZ 文件全部是该脚本生成的合成数据、模型和数据划分；CSV、PNG/SVG 和 JSON 文件包含结果、配置及核验记录。复现时可核对 `candidates.csv`、`selected.csv`、`summary.csv` 和 `path.csv`。

本分支是一次独立的 Group Lasso 工作快照；GeneLLM 原有分支的代码未纳入本实验。
