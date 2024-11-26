import torch
import torch.nn as nn
from .loss import make_loss


class Base(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss = make_loss

    def forward(self, **input):
        output = {}
        print(input['input_ids'].size())
        print(input['attention_mask'].size())
        output['pred'] = self.model(**input)
        print(output['pred'][0].size(), output['pred'][1].size())
        print(output['pred'][0][:, 0, :])  # Embeddings of the `[CLS]` token
        print(output['pred'][1])  # Sequence embeddings

        exit()
        output['loss'] = self.loss(output, input)
        return output


def base(model):
    model = Base(model)
    return model
