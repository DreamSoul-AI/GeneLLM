import dataset
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from dataset.utils import Compose
from module import apply_recursively
from config import cfg


def make_dataset(data_name, verbose=True, **kwargs):
    if verbose:
        print('fetching data {}...'.format(data_name))
    root = os.path.join('data', data_name)    # 根据 data_name 获取数据的根目录
    if data_name in ['GUE']:
        task_name = kwargs['task_name']
        subset_name = kwargs['subset_name']
        # 根据 task_name 来准备不同的数据集
        if task_name == 'all':
            task_names = list(cfg['subset_names'].keys())
            dataset_ = {'train': [], 'valid': [], 'test': []}
            for task_name_i in task_names:
                subset_names = cfg['subset_names'][task_name_i]
                for subset_name_i in subset_names:
                    dataset_train = dataset.GUE(root=root, task_name=task_name_i, subset_name=subset_name_i,
                                                split='train')
                    dataset_valid = dataset.GUE(root=root, task_name=task_name_i, subset_name=subset_name_i,
                                                split='valid')
                    dataset_test = dataset.GUE(root=root, task_name=task_name_i, subset_name=subset_name_i,
                                               split='test')
                    dataset_['train'].append(dataset_train)
                    dataset_['valid'].append(dataset_valid)
                    dataset_['test'].append(dataset_test)
        else: # for the case e.g. EMP_all_index
            if subset_name == 'all':
                dataset_ = {'train': [], 'valid': [], 'test': []}
                subset_names = cfg['subset_names'][task_name]
                for subset_name_i in subset_names:
                    dataset_train = dataset.GUE(root=root, task_name=task_name, subset_name=subset_name_i,
                                                split='train')
                    dataset_valid = dataset.GUE(root=root, task_name=task_name, subset_name=subset_name_i,
                                                split='valid')
                    dataset_test = dataset.GUE(root=root, task_name=task_name, subset_name=subset_name_i,
                                               split='test')
                    dataset_['train'].append(dataset_train)
                    dataset_['valid'].append(dataset_valid)
                    dataset_['test'].append(dataset_test)
            elif isinstance(subset_name, list):
                dataset_ = {'train': [], 'valid': [], 'test': []}
                subset_names = cfg['subset_name']
                for subset_name_i in subset_names:
                    dataset_train = dataset.GUE(root=root, task_name=task_name, subset_name=subset_name_i,
                                                split='train')
                    dataset_valid = dataset.GUE(root=root, task_name=task_name, subset_name=subset_name_i,
                                                split='valid')
                    dataset_test = dataset.GUE(root=root, task_name=task_name, subset_name=subset_name_i,
                                               split='test')
                    dataset_['train'].append(dataset_train)
                    dataset_['valid'].append(dataset_valid)
                    dataset_['test'].append(dataset_test)
            else:
                dataset_ = {}
                dataset_['train'] = dataset.GUE(root=root, task_name=task_name, subset_name=subset_name,
                                                split='train')
                dataset_['valid'] = dataset.GUE(root=root, task_name=task_name, subset_name=subset_name,
                                                split='valid')
                dataset_['test'] = dataset.GUE(root=root, task_name=task_name, subset_name=subset_name,
                                               split='test')
    else:
        raise ValueError('Not valid dataset name')
    if verbose:
        print('data ready')
    return dataset_


def input_collate(input):
    def add_(input_, key=None):
        split_names = key.split('.')
        current = batch
        for split_name in split_names[:-1]:
            if split_name not in current:
                current[split_name] = {}
            current = current[split_name]
        if split_names[-1] not in current:
            current[split_names[-1]] = input_.unsqueeze(0)
        else:
            current[split_names[-1]] = torch.cat([current[split_names[-1]], input_.unsqueeze(0)], dim=0)
        return

    batch = {}
    apply_condition = lambda x: isinstance(x, torch.Tensor)
    identity_condition = lambda x: isinstance(x, (str, type(None)))
    for i in range(len(input)):
        input_i = input[i]
        apply_recursively(add_, input_i, apply_condition=apply_condition, identity_condition=identity_condition)
    return batch


def make_data_collate(collate_mode):
    if collate_mode == 'dict':
        return input_collate
    elif collate_mode == 'default':
        return default_collate
    else:
        raise ValueError('Not valid collate mode')


def make_data_loader(dataset, batch_size, num_steps=None, step=0, step_period=1, pin_memory=True,
                     num_workers=0, collate_mode='dict', seed=0, shuffle=True):
    data_loader = {}
    for k in dataset:
        if k == 'train' and num_steps is not None:
            num_samples = batch_size[k] * (num_steps - step) * step_period
            if num_samples > 0:
                generator = torch.Generator()
                generator.manual_seed(seed)
                sampler = torch.utils.data.RandomSampler(dataset[k], replacement=False, num_samples=num_samples,
                                                         generator=generator)
                data_loader[k] = DataLoader(dataset=dataset[k], batch_size=batch_size[k], sampler=sampler,
                                            pin_memory=pin_memory, num_workers=num_workers,
                                            collate_fn=make_data_collate(collate_mode),
                                            worker_init_fn=np.random.seed(seed))
        else:
            if k == 'train':
                data_loader[k] = DataLoader(dataset=dataset[k], batch_size=batch_size[k], shuffle=shuffle,
                                            pin_memory=pin_memory, num_workers=num_workers,
                                            collate_fn=make_data_collate(collate_mode),
                                            worker_init_fn=np.random.seed(seed))
            else:
                data_loader[k] = DataLoader(dataset=dataset[k], batch_size=batch_size[k], shuffle=False,
                                            pin_memory=pin_memory, num_workers=num_workers,
                                            collate_fn=make_data_collate(collate_mode),
                                            worker_init_fn=np.random.seed(seed))
    return data_loader


def process_dataset(dataset, tokenizer=None, merge_test=True):
    def tokenize_transform(tokenizer, max_length):
        def transform(input):
            tokenized = tokenizer(
                input['data'],
                return_tensors="pt",
                padding="max_length",
                max_length=max_length,
                truncation=True,
            )
            tokenized['input_ids'] = tokenized['input_ids'].squeeze(0)
            tokenized['attention_mask'] = tokenized['attention_mask'].squeeze(0)
            del tokenized['token_type_ids']
            input = {**input, **tokenized}  # 将 tokenized 的结果合并到原来的 input 中
            return input

        return transform

    def dataset_index_transform(input):
        if cfg['task_name'] == 'all':
            input['dataset_idx'] = torch.tensor(cfg['dataset_indices'][(input['task_idx'].item(),
                                                                        input['subset_idx'].item())]) # 给每一个 (task, subtask) e.g. (EMP, xxx) 一个数字 id。
        else:
            if cfg['subset_name'] == 'all':
                # 每一个 input 都有一个 subset_idx，表示它属于哪个 （task, subset）
                #  recall: 每一个 dataset 都有一个 index, (task, subset) -> index, 这个 index 是在 cfg['dataset_indices'] 中定义的。
                input['dataset_idx'] = input['subset_idx']  
            else:
                input['dataset_idx'] = torch.tensor(0)
        return input

    processed_dataset = dataset


    if isinstance(processed_dataset['train'], list):
        if cfg['model']['num_targets'] == 1:
            data_size = [cfg['model']['task_max_length'][processed_dataset['train'][0].task_name]] # task max length
            target_size = processed_dataset['train'][0].target_size  # 模型输出的类别个数
        else:
            data_size = {}
            target_size = {}
            for k in processed_dataset: 
                for i in range(len(processed_dataset[k])):
                    if processed_dataset[k][i].task_name not in data_size:
                        data_size[processed_dataset[k][i].task_name] = \
                            [cfg['model']['task_max_length'][processed_dataset[k][i].task_name]]
                    if processed_dataset[k][i].task_name not in target_size:
                        target_size[processed_dataset[k][i].task_name] = processed_dataset[k][i].target_size
        if tokenizer is not None:
            for k in processed_dataset:  # processed_dataset = {'train': [dataset1, dataset2, ...], 'valid': [dataset1, dataset2, ...], 'test': [dataset1, dataset2, ...]}
                for i in range(len(processed_dataset[k])):
                    processed_dataset[k][i].transform = Compose([
                        dataset_index_transform,
                        tokenize_transform(tokenizer, cfg['model']['max_length'])])  # processed_dataset[k][i] is a dataset object. 这里是 set 每一个 dataset 的 transform
        processed_dataset['train'] = torch.utils.data.ConcatDataset(processed_dataset['train']) # 把多个训练子数据集（dataset 对象）合并成一个大的训练数据集。processed_dataset['train']本来是 a list of dataset。concat 后就变成一个
        if merge_test:
            processed_dataset['valid'] = torch.utils.data.ConcatDataset(processed_dataset['valid']) # 把 processed_dataset['valid'] 也给 merge 了
            processed_dataset['test'] = torch.utils.data.ConcatDataset(processed_dataset['test']) # 把 processed_dataset['test'] 也给 merge 了
            cfg['num_samples'] = {}
            for k in processed_dataset:
                processed_dataset[k].data_size = data_size # data_size is task max length
                processed_dataset[k].target_size = target_size # target_size is the number of classes
                cfg['num_samples'][k] = len(processed_dataset[k]) # 每个数据集的样本数
        else:
            cfg['num_samples'] = {}
            for k in processed_dataset:
                if k == 'train':
                    processed_dataset[k].data_size = data_size
                    processed_dataset[k].target_size = target_size
                    cfg['num_samples'][k] = len(processed_dataset[k])
                else:
                    cfg['num_samples'][k] = []
                    for i in range(len(processed_dataset[k])):
                        processed_dataset[k][i].data_size = data_size
                        processed_dataset[k][i].target_size = target_size
                        cfg['num_samples'][k].append(len(processed_dataset[k]))
        cfg['model']['data_size'] = processed_dataset['train'].data_size
        cfg['model']['target_size'] = processed_dataset['train'].target_size
    else:
        cfg['num_samples'] = {k: len(processed_dataset[k]) for k in processed_dataset}
        cfg['model']['data_size'] = processed_dataset['train'].data_size
        cfg['model']['target_size'] = processed_dataset['train'].target_size
        if tokenizer is not None:
            for k in processed_dataset:
                processed_dataset[k].transform = Compose([
                    dataset_index_transform,
                    tokenize_transform(tokenizer, cfg['model']['max_length'])])

    # 有些配置参数需要根据数据集的大小来调整，比如 batch size 和 num_steps。
    if 'num_epochs' in cfg and cfg['num_epochs'] is not None:
        if cfg['batch_size'] > len(processed_dataset['train']):
            cfg['batch_size'] = len(processed_dataset['train'])
            cfg[cfg['tag']]['optimizer']['batch_size'] = {'train': cfg['batch_size'],
                                                          'test': cfg[cfg['tag']]['optimizer']['test_batch_ratio'] *
                                                                  cfg['batch_size']}
        cfg['num_steps'] = int(np.ceil(len(processed_dataset['train']) / cfg['batch_size'])) * cfg['num_epochs'] # 计算 update model weights 的总次数
        cfg['eval_period'] = int(np.ceil(len(processed_dataset['train']) / cfg['batch_size'])) # 每一个 epoch evaluate 一次模型。这里只不过是换算成每个 epoch 的 step 数量。
        cfg[cfg['tag']]['optimizer']['num_steps'] = cfg['num_steps']
    return processed_dataset
