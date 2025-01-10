import torch
import torch.nn as nn
from .loss import make_loss


class Base(nn.Module):
    def __init__(self, model, hidden_size, target_size, freeze):
        super().__init__()
        self.model = model
        if freeze:
            self.freeze(self.model)
        self.output_proj = nn.Linear(hidden_size, target_size)
        self.loss = make_loss

    def freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False
        return

    def forward(self, **input):
        output = {}
        print(input['input_ids'].size())
        print(input['attention_mask'].size())
        # https://github.com/mosaicml/examples/blob/main/examples/benchmarks/bert/src/bert_layers.py
        encoder_outputs, pooled_output = self.model(**input)
        output['pred'] = self.output_proj(pooled_output)
        # print(output['pred'][0].size(), output['pred'][1].size())
        # print(output['pred'][0][:, 0, :])  # Embeddings of the `[CLS]` token
        # print(output['pred'][1])  # Sequence embeddings
        # print(input.keys())
        output['loss'] = self.loss(output, input)
        print(output['loss'])
        exit()
        return output


def base(model, cfg):
    hidden_size = cfg[cfg['model_name']]['hidden_size']
    target_size = cfg['target_size']
    freeze = cfg['freeze']
    model = Base(model, hidden_size, target_size, freeze)
    return model
