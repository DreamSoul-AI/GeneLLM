import os
import torch
import shutil
from torchvision import transforms
from config import cfg
from dataset import make_dataset, make_data_loader, process_dataset, Compose
from module import save, Stats, makedir_exist_ok, process_control


def move_raw_datasets():
    # Define source and destination directories
    source_dir = './data/GUE/raw/prom'
    dest_dirs = {
        'prom_300': './data/GUE/raw/prom300',
        'prom_core': './data/GUE/raw/promcore'
    }

    # Ensure destination directories exist
    for dest in dest_dirs.values():
        os.makedirs(dest, exist_ok=True)

    # Mapping of source folder names to new folder names
    name_mapping = {
        'prom_300_all': ('prom_300', 'all'),
        'prom_300_notata': ('prom_300', 'notata'),
        'prom_300_tata': ('prom_300', 'tata'),
        'prom_core_all': ('prom_core', 'all'),
        'prom_core_notata': ('prom_core', 'notata'),
        'prom_core_tata': ('prom_core', 'tata')
    }

    # Check and move folders
    if os.path.exists(source_dir):
        for folder in os.listdir(source_dir):
            src_path = os.path.join(source_dir, folder)
            if folder in name_mapping and os.path.isdir(src_path):
                category, new_name = name_mapping[folder]
                dest_path = os.path.join(dest_dirs[category], new_name)
                if not os.path.exists(dest_path):
                    shutil.move(src_path, dest_path)
                    print(f"Moved {folder} to {dest_path}")
                else:
                    print(f"Already moved: {dest_path}")

        os.rmdir(source_dir)
        print(f"Removed empty source directory: {source_dir}")
    return


if __name__ == "__main__":
    stats_path = os.path.join('output', 'stats')
    dim = 1
    data_names = ['GUE']
    task_names = ['EMP', 'mouse', 'promcore', 'prom300', 'splice', 'tf', 'virus']
    subsets = {
        'EMP': ['H3', 'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K9ac', 'H3K14ac', 'H3K36me3', 'H3K79me3', 'H4', 'H4ac'],
        'mouse': ['0', '1', '2', '3', '4'],
        'promcore': ['all', 'notata', 'tata'],
        'prom300': ['all', 'notata', 'tata'],
        'splice': ['reconstructed'],
        'tf': ['0', '1', '2', '3', '4'],
        'virus': ['covid'],
    }
    cfg['seed'] = 0
    cfg['tag'] = 'make_dataset'
    process_control()
    move_raw_datasets()
    with torch.no_grad():
        for data_name in data_names:
            for task_name in task_names:
                for subset in subsets[task_name]:
                    dataset = make_dataset(data_name, task_name=task_name, subset=subset)
                    process_dataset(dataset)
                    cfg['step'] = 0
                    data_loader = make_data_loader(dataset, cfg[cfg['tag']]['optimizer']['batch_size'], shuffle=False)
                    print(data_name, task_name, subset)
