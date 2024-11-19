import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from module import check_exists, save, load
from .utils import make_classes_counts


class GUE(Dataset):
    data_name = 'GUE'

    def __init__(self, root, task_name, subset, split, transform=None):
        self.root = os.path.expanduser(root)
        self.task_name = task_name
        self.subset = subset
        self.split = split
        self.transform = transform
        if not check_exists(self.processed_folder):
            self.process()
        self.id, self.data, self.target = load(os.path.join(self.processed_folder, self.split))
        self.other = {}
        self.classes_counts = make_classes_counts(self.target)
        self.data_shape, self.target_size, self.classes_to_label = load(os.path.join(self.processed_folder, 'meta'))

    def __getitem__(self, index):
        id, data, target = torch.tensor(self.id[index]), self.data[index], torch.tensor(
            self.target[index])

        print(data)
        exit()



        input = {'id': id, 'data': data, 'target': target}
        other = {k: torch.tensor(self.other[k][index]) for k in self.other}
        input = {**input, **other}
        if self.transform is not None:
            input = self.transform(input['data'])
        return input

    def __len__(self):
        return len(self.data)

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed', self.task_name, self.subset)

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw', self.task_name, self.subset)

    def process(self):
        if not check_exists(self.raw_folder):
            self.download()
        train_set, valid_set, test_set, meta = self.make_data()
        save(train_set, os.path.join(self.processed_folder, 'train'))
        save(valid_set, os.path.join(self.processed_folder, 'valid'))
        save(test_set, os.path.join(self.processed_folder, 'test'))
        save(meta, os.path.join(self.processed_folder, 'meta'))
        return

    def download(self):
        raise NotImplementedError

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nTask name:{}\nSubset:{}\nSplit: {}\nTransforms: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.task_name, self.subset, self.split,
            self.transform.__repr__())
        return fmt_str

    def make_data(self):
        def make_data_split(df):
            # load data from the disk
            if df.shape[1] == 2:
                # data is in the format of [text, label]
                texts = df.iloc[:, 0].tolist()
                labels = np.array(df.iloc[:, 1].astype(int).tolist()).astype(np.int64)
            elif df.shape[1] == 3:
                # data is in the format of [text1, text2, label]
                texts = df.iloc[:, [0, 1]].values.tolist()  # Get the first two columns
                labels = np.array(df.iloc[:, 2].astype(int).tolist()).astype(np.int64) # Get the third column and convert to int
            else:
                raise ValueError("Data format not supported.")
            id = np.arange(len(texts), dtype=np.int64)
            data, target = texts, labels
            return id, data, target

        train_df = pd.read_csv(os.path.join(self.raw_folder, 'train.csv'), delimiter=',')
        valid_df = pd.read_csv(os.path.join(self.raw_folder, 'dev.csv'), delimiter=',')
        test_df = pd.read_csv(os.path.join(self.raw_folder, 'test.csv'), delimiter=',')

        train_id, train_data, train_target = make_data_split(train_df)
        valid_id, valid_data, valid_target = make_data_split(valid_df)
        test_id, test_data, test_target = make_data_split(test_df)
        data_shape = [len(train_data[0])]
        classes = np.unique(train_target)
        classes_to_labels = {classes[i]: i for i in range(len(classes))}
        target_size = len(classes)
        return (train_id, train_data, train_target), (valid_id, valid_data, valid_target), \
                (test_id, test_data, test_target), (data_shape, target_size, classes_to_labels)
