import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np

# 使用说明：在 src 目录下运行此脚本
# e.g. python process_exp_result/plot_result.py --file_name result_split_processed_merged

# ===== 添加命令行参数解析器 =====
parser = argparse.ArgumentParser(description="Process and summarize test results.")
parser.add_argument('--file_name', type=str, required=True,
                    help="Name of the Excel file (without .xlsx extension), e.g., result_split_processed")
args = parser.parse_args()
 
file_name = args.file_name 

# ===== 路径拼接 =====

in_file_path = os.path.join(".", "process_exp_result","processed_result", "processed", f"{file_name}.xlsx")


# 读取你的合并结果文件
df = pd.read_excel(in_file_path)  # 替换为你实际的文件路径

# 修改 df 的 values, 需要乘上 100
df['test_MCC_best'] = df['test_MCC_best'] * 100
df['test_F1_best'] = df['test_F1_best'] * 100

# 添加 label 字段
df['label'] = df['task_name_test'] + '_' + df['subset_name_test']


# 准备横轴位置
x = np.arange(len(df))
width = 0.35

# 初始化数据列
bar1_values = []
bar2_values = []
bar_labels = []

for idx, row in df.iterrows():
    if row['task_name_test'] == 'virus':
        bar1_values.append(row['test_F1_best'])
        bar2_values.append(row['test_F1_DNABert2'])
        bar_labels.append('F1')
    else:
        bar1_values.append(row['test_MCC_best'])
        bar2_values.append(row['test_MCC_DNABert2'])
        bar_labels.append('MCC')

# 画图
fig, ax = plt.subplots(figsize=(12, 6))

bar1 = ax.bar(x - width/2, bar1_values, width, label='Experiment', color='steelblue')
bar2 = ax.bar(x + width/2, bar2_values, width, label='DNABert2', color='orange')

# 横轴设置
ax.set_ylabel('Metric Value (%)')
ax.set_title(file_name)
ax.set_xticks(x)
ax.set_xticklabels(df['label'], rotation=45, ha='right')
ax.legend()

# 给是 F1 的 bar 加上文字标签
shown_mcc_label = False  # 控制只显示一个 MCC

for i in range(len(df)):
    label = bar_labels[i]
    
    if label == 'F1':
        # 所有 F1 都显示
        ax.text(x[i] - width/2, bar1_values[i] + 0.01, 'F1', ha='center', va='bottom', fontsize=8)
        ax.text(x[i] + width/2, bar2_values[i] + 0.01, 'F1', ha='center', va='bottom', fontsize=8)
    
    elif label == 'MCC' and not shown_mcc_label:
        # 只显示第一个 MCC 标签
        ax.text(x[i] - width/2, bar1_values[i] + 0.01, 'MCC', ha='center', va='bottom', fontsize=8)
        ax.text(x[i] + width/2, bar2_values[i] + 0.01, 'MCC', ha='center', va='bottom', fontsize=8)
        shown_mcc_label = True


plt.tight_layout()

# 保存图像（可选）
out_fig_path = os.path.join(".", "process_exp_result", "processed_result", "plots", f"{file_name}_result_comparison.png")
plt.savefig(out_fig_path, dpi=300)

plt.show()

