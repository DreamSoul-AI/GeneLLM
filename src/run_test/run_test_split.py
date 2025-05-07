import os
import subprocess

# 定义基础路径
base_dir = '.'  # 指 src
output_dir = os.path.join(base_dir, 'output', 'test_result')  # ./output/test_result
script_path = os.path.join(base_dir, 'output', 'script', 'test_mouse_split_1.sh')
output_file = os.path.join(output_dir, 'test_split.txt') # ./output/test_result/test_split.txt

# 创建 test_result 文件夹（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 运行 shell 脚本并将输出追加保存到文件中
with open(output_file, 'a') as outfile:
    subprocess.run(['bash', script_path], stdout=outfile, stderr=subprocess.STDOUT)