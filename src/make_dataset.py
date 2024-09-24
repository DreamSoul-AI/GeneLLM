import os
import torch
from torchvision import transforms
from config import cfg
from dataset import make_dataset, make_data_loader, process_dataset, Compose
from module import save, Stats, makedir_exist_ok, process_control

if __name__ == "__main__":
    stats_path = os.path.join('output', 'stats')
    dim = 1
    data_names = ['GUE']
    task_names = ['EMP', 'mouse', 'prom', 'splice', 'tf', 'virus']
    subsets = {
        'EMP': ['H3', 'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K9ac', 'H3K14ac', 'H3K36me3', 'H3K79me3', 'H4', 'H4ac'],
        'mouse': ['0', '1', '2', '3', '4'],
        'prom': ['prom_300_all', 'prom_300_notata', 'prom_300_tata', 'prom_core_all', 'prom_core_notata',
                 'prom_core_tata'],
        'splice': ['reconstructed'],
        'tf': ['0', '1', '2', '3', '4'],
        'virus': ['covid'],
    }
    cfg['seed'] = 0
    cfg['tag'] = 'make_dataset'
    process_control()
    with torch.no_grad():
        for data_name in data_names:
            for task_name in task_names:
                for subset in subsets[task_name]:
                    dataset = make_dataset(data_name, task_name=task_name, subset=subset)
                    process_dataset(dataset)
                    cfg['step'] = 0
                    data_loader = make_data_loader(dataset, cfg[cfg['tag']]['optimizer']['batch_size'], shuffle=False)
                    print(data_name, task_name, subset)
