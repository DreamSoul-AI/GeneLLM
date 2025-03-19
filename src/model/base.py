import torch
import torch.nn as nn
from module import filter_args
from .loss import make_loss


class Base(nn.Module):
    def __init__(self, model, hidden_size, target_size, num_datasets, num_targets, task_names, subset_names, task_name,
                 subset_name, freeze, dataset_embedding_mode):
        super().__init__()
        self.model = model
        self.hidden_size = hidden_size
        self.target_size = target_size
        self.num_datasets = num_datasets
        self.num_targets = num_targets
        self.task_names = task_names
        self.subset_names = subset_names
        self.task_name = task_name
        self.subset_name = subset_name
        if freeze:
            self.freeze(self.model)
        self.dataset_embedding_mode = dataset_embedding_mode
        self.dataset_embedding = self.make_dataset_embedding()

        if num_targets == 1:
            self.output_proj = nn.Linear(hidden_size, target_size)
        else:
            output_proj = []
            for i in range(num_targets):
                target_size_i = target_size[task_names[i]]
                output_proj.append(nn.Linear(hidden_size, target_size_i))
            self.output_proj = nn.ModuleList(output_proj)
        self.loss = make_loss

    def freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False
        return

    def make_dataset_embedding(self):
        if self.dataset_embedding_mode == 'index':
            dataset_embedding = nn.Embedding(self.num_datasets, self.hidden_size)
        else:
            dataset_embedding = None
        return dataset_embedding

    def forward(self, **input):
        output = {}
        # https://github.com/mosaicml/examples/blob/main/examples/benchmarks/bert/src/bert_layers.py
        with torch.no_grad():
            valid_input = filter_args(self.model.forward, input)
            encoder_outputs, pooled_output = self.model(**valid_input)
        decoder_input = pooled_output

        if self.dataset_embedding is not None:
            decoder_input += self.dataset_embedding(input['dataset_idx'])

        if self.num_targets == 1:
            output['pred'] = self.output_proj(decoder_input)
            output['loss'] = self.loss(output['pred'], input['target'])
        else:
            loss = 0
            unique_task_idx = torch.unique(input['task_idx'])
            output['pred'] = {}
            output['loss_task'] = {}
            for i in range(len(unique_task_idx)):
                task_idx = unique_task_idx[i].item()
                mask_i = input['task_idx'] == task_idx
                output_i = self.output_proj[task_idx](decoder_input[mask_i])
                target_i = input['target'][mask_i]
                output['pred'][task_idx] = target_i
                loss_i = self.loss(output_i, target_i, reduction='sum')
                loss += loss_i
            output['loss'] = loss / len(encoder_outputs)
            if not self.training and 'test_task_idx' in input:
                output['pred'] = output['pred'][input['test_task_idx']]
        return output


def base(model, cfg):
    hidden_size = cfg[cfg['model_name']]['hidden_size']
    target_size = cfg['target_size']
    num_datasets = cfg['num_datasets']
    num_targets = cfg['num_targets']
    task_names = cfg['task_names']
    subset_names = cfg['subset_names']
    task_name = cfg['task_name']
    subset_name = cfg['subset_name']
    freeze = cfg['freeze']
    dataset_embedding_mode = cfg['dataset_embedding_mode']
    model = Base(model, hidden_size, target_size, num_datasets, num_targets, task_names, subset_names, task_name,
                 subset_name, freeze, dataset_embedding_mode)
    return model
