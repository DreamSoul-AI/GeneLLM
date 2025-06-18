import pandas as pd
import os

# 定义输入文件路径（相对路径示例）
file_name = "result_all_all_embedding_word"

in_file_path = os.path.join(".", "processed_result", "raw", f"{file_name}.xlsx")

# 读取 Excel 文件，生成 DataFrame
df = pd.read_excel(in_file_path)

# 按照 task_name_test 和 subset_name_test 两个字段分组
grouped = df.groupby(["task_name_test", "subset_name_test"])

# 对每个分组执行多个聚合操作：
# - test_F1 字段聚合为列表（把该组所有 test_F1 放进 list）
# - test_F1 字段求最大值（max）
# - test_MCC 字段聚合为列表
# - test_MCC 字段求最大值
# 返回一个多级列名的 DataFrame
result = grouped.agg({
    "test_F1": [lambda x: list(x), "max"],
    "test_MCC": [lambda x: list(x), "max"]
}).reset_index()  # reset_index 让 groupby keys 成为普通列

# 多级列名（MultiIndex）变成普通列名，赋予新的有意义名字
result.columns = [
    "task_name_test",   # group key1
    "subset_name_test", # group key2
    "test_F1_list",     # test_F1 聚合为 list
    "test_F1_best",     # test_F1 最大值
    "test_MCC_list",    # test_MCC 聚合为 list
    "test_MCC_best"     # test_MCC 最大值
]

# 输出保存到新的 Excel 文件，index=False 不保存行索引
out_file_path = os.path.join(".", "processed_result", "processed", f"{file_name}_processed.xlsx")
result.to_excel(out_file_path, index=False)

print(f"处理完成，结果已保存到：{out_file_path}")
