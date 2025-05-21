import os
import torch
import torch.nn as nn
from module import filter_args, load
from .loss import make_loss


class DataEmbedding(nn.Module):
    def __init__(self, embedding_mode, task_name, task_names, num_datasets, hidden_size):
        super().__init__()
        self.embedding_mode = embedding_mode
        self.task_name = task_name
        self.task_names = task_names
        self.hidden_size = hidden_size
        self.num_datasets = num_datasets

        if self.embedding_mode == 'index':
            self.dataset_embedding = nn.Embedding(num_datasets, hidden_size)
        elif self.embedding_mode == 'word' and self.task_name == 'all':
            embedding = []
            for task_name in self.task_names:
                embedding_i = load(os.path.join('data', 'GUE', 'description_embedding', task_name))['pooler_output']
                embedding.append(embedding_i)
            embedding = torch.cat(embedding, dim=0)
            word_embedding_size = embedding.size(-1)
            task_embedding = nn.Embedding(len(self.task_names), word_embedding_size)
            task_embedding.weight.data.copy_(embedding.data)
            task_embedding.weight.requires_grad = False
            self.task_embedding = nn.Sequential(task_embedding, nn.Linear(word_embedding_size, hidden_size))
        else:
            self.dataset_embedding = None

    def forward(self, dataset_idx, task_idx=None):
        if self.embedding_mode == 'index':
            data_embedding = self.dataset_embedding(dataset_idx)
        elif self.embedding_mode == 'word' and self.task_name == 'all':
            data_embedding = self.task_embedding(task_idx)
        else:
            data_embedding = 0
        return data_embedding


class BertBase(nn.Module):
    def __init__(self, model, hidden_size, target_size, num_datasets, num_targets, task_names, subset_names, task_name,
                 subset_name, freeze, embedding_mode):
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
        self.freeze = freeze
        if self.freeze:
            self.freeze(self.model)
        self.embedding_mode = embedding_mode
        self.dataset_embedding = DataEmbedding(self.embedding_mode, self.task_name, self.task_names, self.num_datasets,
                                               self.hidden_size)
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

    def forward(self, **input):
        output = {}
        # https://github.com/mosaicml/examples/blob/main/examples/benchmarks/bert/src/bert_layers.py
        valid_input = filter_args(self.model.forward, input)
        if self.freeze:
            with torch.no_grad():
                encoder_outputs, pooled_output = self.model(**valid_input)
        else:
            encoder_outputs, pooled_output = self.model(**valid_input)
        # decoder_input = encoder_outputs.mean(dim=1)
        decoder_input = pooled_output

        if self.embedding_mode != 'none':
            dataset_embedding = self.dataset_embedding(input['dataset_idx'], input['task_idx'])
            decoder_input = decoder_input + dataset_embedding

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
                output['pred'][task_idx] = output_i
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
    embedding_mode = cfg['embedding_mode']
    if cfg['model_name'] in ['dnabert2']:
        model = BertBase(model, hidden_size, target_size, num_datasets, num_targets, task_names, subset_names,
                         task_name,
                         subset_name, freeze, embedding_mode)
    else:
        model = LanguageModelBase(model, hidden_size, target_size, num_datasets, num_targets, task_names, subset_names,
                                  task_name, subset_name, freeze, embedding_mode)
    return model
