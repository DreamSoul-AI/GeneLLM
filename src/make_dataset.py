import shutil
import os
import torch
from transformers import AutoTokenizer, AutoModel
from config import cfg
from dataset import make_dataset, make_data_loader, process_dataset
from module import save, to_device, process_control


def move_prom_datasets():
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


def move_mouse_datasets():
    # Define source and destination directories
    source_dir = './data/GUE/raw/mouse'
    # Mapping of source folder names to new folder names
    name_mapping = {
        '0': 'Ch12Nrf2Iggrab',
        '1': 'Ch12Znf384hpa004051Iggrab',
        '2': 'MelJundIggrab',
        '3': 'MelMafkDm2p5dStd',
        '4': 'MelNelfeIggrab'
    }

    # Check and move folders
    for name, mapped_name in name_mapping.items():
        src_path = os.path.join(source_dir, name)
        if os.path.exists(src_path):
            dest_path = os.path.join(source_dir, mapped_name)
            if not os.path.exists(dest_path):
                shutil.move(src_path, dest_path)
                print(f"Moved {name} to {dest_path}")
            else:
                print(f"Already moved: {dest_path}")
            print(f"Removed empty source directory: {source_dir}")
    return


def move_tf_datasets():
    # Define source and destination directories
    source_dir = './data/GUE/raw/tf'
    # Mapping of source folder names to new folder names
    name_mapping = {
        '0': 'wgEncodeEH000552',
        '1': 'wgEncodeEH000606',
        '2': 'wgEncodeEH001546',
        '3': 'wgEncodeEH001776',
        '4': 'wgEncodeEH002829'
    }

    # Check and move folders
    for name, mapped_name in name_mapping.items():
        src_path = os.path.join(source_dir, name)
        if os.path.exists(os.path.join(src_path)):
            src_path = os.path.join(source_dir, name)
            dest_path = os.path.join(source_dir, mapped_name)
            if not os.path.exists(dest_path):
                shutil.move(src_path, dest_path)
                print(f"Moved {name} to {dest_path}")
            else:
                print(f"Already moved: {dest_path}")
            print(f"Removed empty source directory: {source_dir}")
    return


def make_embeddings():
    model_name = 'intfloat/multilingual-e5-large-instruct'
    cache_dir = os.path.join('output', 'cache')
    cache_tokenizer_path = os.path.join(cache_dir, model_name, 'tokenizer')
    cache_model_path = os.path.join(cache_dir, model_name, 'model')
    local_files_only = {'tokenizer': False, 'model': False}
    for key in local_files_only:
        if os.path.exists(os.path.join(cache_dir, model_name, key)):
            local_files_only[key] = True
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_tokenizer_path,
                                              local_files_only=local_files_only['tokenizer'])  # local_files_only, if True, will only load from local files
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_model_path,
                                      local_files_only=local_files_only['model'])
    if torch.cuda.is_available():
        model = model.to('cuda')

    task_names = ['EMP', 'mouse', 'promcore', 'prom300', 'splice', 'tf', 'virus']
    base_folder = os.path.join('data', 'GUE', 'description_embedding')
    description_path = os.path.join(".", "dataset", "description")
    for task_name in task_names:
        task_file_path = os.path.join(description_path, f"{task_name}.txt")
        with open(task_file_path, 'r') as f:
            description_i = f.read().replace('\n', '')
        input = tokenizer(description_i, return_tensors='pt')
        if torch.cuda.is_available():
            input = to_device(input, 'cuda')
        with torch.no_grad():
            output = model(**input)
            output = to_device(output, 'cpu')
        save(output, os.path.join(base_folder, task_name))
    return


if __name__ == "__main__":
    stats_path = os.path.join('output', 'stats')
    dim = 1
    data_names = ['GUE']
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
    cfg['seed'] = 0
    cfg['tag'] = 'make_dataset'
    process_control()
    move_prom_datasets()
    move_mouse_datasets()
    move_tf_datasets()
    make_embeddings()
    with torch.no_grad():
        for data_name in data_names:
            for task_name in task_names:
                for subset_name in subset_names[task_name]:
                    dataset = make_dataset(data_name, task_name=task_name, subset_name=subset_name)
                    process_dataset(dataset)
                    cfg['step'] = 0
                    data_loader = make_data_loader(dataset, cfg[cfg['tag']]['optimizer']['batch_size'], shuffle=False)
                    print(data_name, task_name, subset_name)
                    print(cfg['num_samples'])
