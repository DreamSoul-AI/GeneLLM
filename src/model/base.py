import torch
import torch.nn as nn
from module import filter_args
from .loss import make_loss


class Base(nn.Module):
    def __init__(self, model, hidden_size, target_size, num_targets, task_names, subset_names, freeze):
        super().__init__()
        self.model = model
        self.num_targets = num_targets
        self.task_names = task_names
        self.subset_names = subset_names
        if freeze:
            self.freeze(self.model)
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
        # print(input)
        # print(input['input_ids'].size())
        # print(input['attention_mask'].size())
        # https://github.com/mosaicml/examples/blob/main/examples/benchmarks/bert/src/bert_layers.py
        with torch.no_grad():
            valid_input = filter_args(self.model.forward, input)
            print(valid_input)
            encoder_outputs, pooled_output = self.model(**valid_input)
        if self.num_targets == 1:
            output['pred'] = self.output_proj(pooled_output)
        else:
            unique_task_idx = torch.unique(input['task_idx'])
            for i in range(len(unique_task_idx)):
                task_idx = unique_task_idx[i]
            print(unique_task_idx, input['task_idx'])
            exit()
        # print(output['pred'][0].size(), output['pred'][1].size())
        # print(output['pred'][0][:, 0, :])  # Embeddings of the `[CLS]` token
        # print(output['pred'][1])  # Sequence embeddings
        # print(input.keys())
        output['loss'] = self.loss(output, input)
        # print(output['loss'])
        # exit()
        return output


def base(model, cfg):
    hidden_size = cfg[cfg['model_name']]['hidden_size']
    target_size = cfg['target_size']
    num_targets = cfg['num_targets']
    task_names = cfg['task_names']
    subset_names = cfg['subset_names']
    freeze = cfg['freeze']
    model = Base(model, hidden_size, target_size, num_targets, task_names, subset_names, freeze)
    return model
