import pandas as pd
import os
import argparse



# 使用说明：在 src 目录下运行此脚本
# e.g. python process_exp_result/merge_results.py --file_name result_split_processed

# ===== 添加命令行参数解析器 =====
parser = argparse.ArgumentParser(description="Process and summarize test results.")
parser.add_argument('--file_name', type=str, required=True,
                    help="Name of the Excel file (without .xlsx extension), e.g., result_split_processed")
args = parser.parse_args()
 
file_name = args.file_name 

# ===== 路径拼接 =====

in_file_path_exp = os.path.join(".", "process_exp_result","processed_result", "processed", f"{file_name}.xlsx")
in_file_path_DNABert2 = os.path.join(".", "process_exp_result","processed_result", "processed", "DNABert2_results.xlsx")

# 读取 Excel 文件，生成 DataFrame
df_exp = pd.read_excel(in_file_path_exp)
df_DNABert2 = pd.read_excel(in_file_path_DNABert2)

merged_df = df_exp.merge(
    df_DNABert2[['task_name_test', 'subset_name_test', 'test_MCC', 'test_F1']],
    on=['task_name_test', 'subset_name_test'],
    how='left',  # 保留实验表格中的所有记录
    suffixes=('', '_DNABert2')  # 避免字段冲突
)

# rename columns
merged_df = merged_df.rename(columns={
    'test_MCC': 'test_MCC_DNABert2',
    'test_F1': 'test_F1_DNABert2'
})

# 输出保存到新的 Excel 文件，index=False 不保存行索引
out_file_path = os.path.join(".", "process_exp_result", "processed_result", "processed", f"{file_name}_merged.xlsx")
merged_df.to_excel(out_file_path, index=False)

print(f"处理完成，结果已保存到：{out_file_path}")
