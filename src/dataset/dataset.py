import dataset
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from config import cfg


def make_dataset(data_name, verbose=True, **kwargs):
    if verbose:
        print('fetching data {}...'.format(data_name))
    root = os.path.join('data', data_name)
    if data_name in ['GUE']:
        task_name = kwargs['task_name']
        subset = kwargs['subset']
        if subset == '-1':
            subset_names = cfg['subset_names'][task_name]
            dataset_ = {'train': [], 'valid': [], 'test': []}
            for subset in subset_names:
                dataset_train = dataset.GUE(root=root, task_name=task_name, subset=subset, split='train')
                dataset_valid = dataset.GUE(root=root, task_name=task_name, subset=subset, split='valid')
                dataset_test = dataset.GUE(root=root, task_name=task_name, subset=subset, split='test')
                dataset_['train'].append(dataset_train)
                dataset_['valid'].append(dataset_valid)
                dataset_['test'].append(dataset_test)
            # data_size = dataset_['train'][0].data_size
            # target_size = dataset_['train'][0].target_size
            # dataset_['train'] = torch.utils.data.ConcatDataset(dataset_['train'])
            # dataset_['valid'] = torch.utils.data.ConcatDataset(dataset_['valid'])
            # dataset_['test'] = torch.utils.data.ConcatDataset(dataset_['test'])
            # for split in dataset_:
            #     dataset_[split].data_size = data_size
            #     dataset_[split].target_size = target_size
        else:
            dataset_ = {}
            dataset_['train'] = dataset.GUE(root=root, task_name=task_name, subset=subset, split='train')
            dataset_['valid'] = dataset.GUE(root=root, task_name=task_name, subset=subset, split='valid')
            dataset_['test'] = dataset.GUE(root=root, task_name=task_name, subset=subset, split='test')
    else:
        raise ValueError('Not valid dataset name')
    if verbose:
        print('data ready')
    return dataset_


def input_collate(input):
    first = input[0]
    batch = {}
    for k, v in first.items():
        if v is not None:
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in input])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in input]))
            elif isinstance(v, str):
                batch[k] = [f[k] for f in input]
            else:
                batch[k] = torch.tensor([f[k] for f in input])
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
        if tokenizer is not None:
            data_size = processed_dataset['train'][0].data_size
            target_size = processed_dataset['train'][0].target_size
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
            cfg['model']['data_size'] = dataset['train'].data_size
            cfg['model']['target_size'] = dataset['train'].target_size
        else:
            cfg['model']['data_size'] = dataset['train'][0].data_size
            cfg['model']['target_size'] = dataset['train'][0].target_size
    else:
        if tokenizer is not None:
            for k in processed_dataset:
                processed_dataset[k].transform = tokenize_transform(tokenizer, cfg['model']['max_length'])
        cfg['num_samples'] = {k: len(processed_dataset[k]) for k in processed_dataset}
        cfg['model']['data_size'] = dataset['train'].data_size
        cfg['model']['target_size'] = dataset['train'].target_size

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
