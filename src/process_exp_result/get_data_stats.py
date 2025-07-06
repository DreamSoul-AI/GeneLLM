import sys
import os
import pandas as pd

# in directory src
# Example usage: python process_exp_result/get_data_stats.py 
# get data statistics for all tasks and subsets: task_name, subset_name, DNA_seq_length, train_samples, valid_samples, test_samples

# add src to sys.path, so that we can import modules from it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # os.path.dirname(__file__) is the directory of this file, i.e. /src/process_exp_result
from module import load


def get_data_stats(task_name, subset_name, data_path_root = os.path.join('data', 'GUE', 'processed')):
    """
    Get data statistics for a specific task and subset.
    """
    data_path = os.path.join(data_path_root, task_name, subset_name)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path {data_path} does not exist.")
    
    train = load(os.path.join(data_path, 'train'))
    valid = load(os.path.join(data_path, 'valid'))
    test = load(os.path.join(data_path, 'test'))
    
    stats = {
        'task_name': task_name,
        'subset_name': subset_name,
        'DNA_seq_length': len(train[1][0]),
        'train_samples': len(train[0]),
        'valid_samples': len(valid[0]),
        'test_samples': len(test[0])
    }

    return stats



task_names = ['EMP', 'mouse', 'promcore', 'prom300', 'splice', 'tf', 'virus']
subset_names = {
        'EMP': ['H3', 'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K9ac', 'H3K14ac', 'H3K36me3', 'H3K79me3', 'H4', 'H4ac'],
        # 'mouse': ['0', '1', '2', '3', '4'],
        'mouse': ['Ch12Nrf2Iggrab', 'Ch12Znf384hpa004051Iggrab', 'MelJundIggrab', 'MelMafkDm2p5dStd', 'MelNelfeIggrab'],
        'promcore': ['all', 'notata', 'tata'],
        'prom300': ['all', 'notata', 'tata'],
        'splice': ['reconstructed'],
        # 'tf': ['0', '1', '2', '3', '4'],
        'tf': ['wgEncodeEH000552', 'wgEncodeEH000606', 'wgEncodeEH001546', 'wgEncodeEH001776', 'wgEncodeEH002829'],
        'virus': ['covid'],
    }


# 收集所有统计信息
all_stats = []
for task_name in task_names:
    for subset_name in subset_names[task_name]:
        stats = get_data_stats(task_name, subset_name)
        all_stats.append(stats)
        
# 转为 DataFrame
df = pd.DataFrame(all_stats)

# 保存为 Excel 文件或 CSV
output_dir = os.path.join('.','process_exp_result' , 'processed_result','data_stats', 'dataset_stats.xlsx')
df.to_excel(output_dir, index=False)
