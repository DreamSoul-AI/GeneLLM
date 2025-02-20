import dataset
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from module import apply_recursively
from config import cfg


def make_dataset(data_name, verbose=True, **kwargs):
    if verbose:
        print('fetching data {}...'.format(data_name))
    root = os.path.join('data', data_name)
    if data_name in ['GUE']:
        task_name = kwargs['task_name']
        subset_name = kwargs['subset_name']
        if task_name == -1:
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
        else:
            if subset_name == '-1':
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
                data_loader[k] = DataLoader(dataset=dataset[k], batch_size=batch_size[k], shuffle=True,
                                            pin_memory=pin_memory, num_workers=num_workers,
                                            collate_fn=make_data_collate(collate_mode),
                                            worker_init_fn=np.random.seed(seed))
    return data_loader


def process_dataset(dataset, tokenizer=None):
    def tokenize_transform(tokenizer, max_length):
        def transform(example):
            tokenized = tokenizer(
                example,
                return_tensors="pt",
                padding="max_length",
                max_length=max_length,
                truncation=False,
            )
            tokenized['input_ids'] = tokenized['input_ids'].squeeze(0)
            tokenized['attention_mask'] = tokenized['attention_mask'].squeeze(0)
            del tokenized['token_type_ids']
            return tokenized

        return transform

    processed_dataset = dataset

    if isinstance(processed_dataset['train'], list):
        if cfg['model']['num_targets'] == 1:
            data_size = processed_dataset['train'][0].data_size
            target_size = processed_dataset['train'][0].target_size
        else:
            data_size = {} # TODO: this seems has error
            target_size = {}
            for k in processed_dataset:
                for i in range(len(processed_dataset[k])):
                    if processed_dataset[k][i].task_name not in data_size:
                        data_size[processed_dataset[k][i].task_name] = processed_dataset[k][i].data_size
                    if processed_dataset[k][i].task_name not in target_size:
                        target_size[processed_dataset[k][i].task_name] = processed_dataset[k][i].target_size
        if tokenizer is not None:
            for k in processed_dataset:
                for i in range(len(processed_dataset[k])):
                    processed_dataset[k][i].transform = tokenize_transform(tokenizer, cfg['model']['max_length'])
        processed_dataset['train'] = torch.utils.data.ConcatDataset(processed_dataset['train'])
        processed_dataset['valid'] = torch.utils.data.ConcatDataset(processed_dataset['valid'])
        processed_dataset['test'] = torch.utils.data.ConcatDataset(processed_dataset['test'])
        for k in processed_dataset:
            processed_dataset[k].data_size = data_size
            processed_dataset[k].target_size = target_size
        cfg['num_samples'] = {k: len(processed_dataset[k]) for k in processed_dataset}
        cfg['model']['data_size'] = processed_dataset['train'].data_size
        cfg['model']['target_size'] = processed_dataset['train'].target_size
    else:
        cfg['num_samples'] = {k: len(processed_dataset[k]) for k in processed_dataset}
        cfg['model']['data_size'] = processed_dataset['train'].data_size
        cfg['model']['target_size'] = processed_dataset['train'].target_size
        if tokenizer is not None:
            for k in processed_dataset:
                processed_dataset[k].transform = tokenize_transform(tokenizer, cfg['model']['max_length'])

    if 'num_epochs' in cfg:
        cfg['num_steps'] = int(np.ceil(len(processed_dataset['train']) / cfg['batch_size'])) * cfg['num_epochs']
        cfg['eval_period'] = int(np.ceil(len(processed_dataset['train']) / cfg['batch_size']))
        cfg[cfg['tag']]['optimizer']['num_steps'] = cfg['num_steps']

    # data_size = dataset_['train'][0].data_size
    # target_size = dataset_['train'][0].target_size
    # dataset_['train'] = torch.utils.data.ConcatDataset(dataset_['train'])
    # dataset_['valid'] = torch.utils.data.ConcatDataset(dataset_['valid'])
    # dataset_['test'] = torch.utils.data.ConcatDataset(dataset_['test'])
    # for split in dataset_:
    #     dataset_[split].data_size = data_size
    #     dataset_[split].target_size = target_size
    return processed_dataset
