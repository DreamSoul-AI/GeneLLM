import re
import os

base_dir = '.'  # 指 src
output_dir = os.path.join(base_dir, 'output', 'test_result')  # ./output/test_result
script_path = os.path.join(base_dir, 'output', 'script', 'train_mouse_split_1.sh')
output_file = os.path.join(output_dir, 'test_split.txt') # ./output/test_result/test_split.txt

# 读取文件
with open(output_file, 'r') as f:
    content = f.read()

# 按每个 Experiment block 分段（用 "Experiment:" 开头分隔）
experiment_blocks = re.split(r'(?=Experiment:)', content)
experiment_blocks = experiment_blocks[1:] # 去掉第0个空的 block

# 存储结果
results = []

# 处理每个 block
for block in experiment_blocks:
    exp_match = re.search(r'Experiment:\s*(\S+)', block)
    test_match = re.search(
        r'Model:.*?test\(.*?\)\s+Loss:\s*([\d.]+)\s+Accuracy:\s*([\d.]+)\s+F1:\s*([\d.]+)\s+MCC:\s*([-.\d]+)',
        block
    )

    if exp_match and test_match:
        experiment_name = exp_match.group(1)
        test_loss = test_match.group(1)
        test_accuracy = test_match.group(2)
        test_f1 = test_match.group(3)
        test_mcc = test_match.group(4)
        
        results.append({
            'Experiment': experiment_name,
            'Loss': test_loss,
            'Accuracy': test_accuracy,
            'F1': test_f1,
            'MCC': test_mcc
        })

# 打印结果
for res in results:
    print(f"Experiment: {res['Experiment']}")
    print(f"  Test Loss: {res['Loss']}")
    print(f"  Test Accuracy: {res['Accuracy']}")
    print(f"  Test F1: {res['F1']}")
    print(f"  Test MCC: {res['MCC']}")
    print('-' * 40)