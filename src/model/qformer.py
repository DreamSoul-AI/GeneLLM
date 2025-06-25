import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

# https://github.com/salesforce/LAVIS/blob/506965b9c4a18c1e565bd32acaccabe0198433f7/lavis/models/blip2_models/blip2_qformer.py#L27
class QFormer(nn.Module):
    def __init__(self, bert_model_name, num_query_tokens, hidden_size, encoder_width, cross_attention_freq):
        super().__init__()
        # TODO: check if exists, download
        config = BertConfig.from_pretrained(bert_model_name)
        config.add_cross_attention = True
        config.cross_attention_freq = cross_attention_freq
        config.encoder_width = encoder_width
        self.bert = BertModel(config)
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, hidden_size))

    def forward(self, encoder_hidden_states, encoder_attention_mask=None):
        batch_size = encoder_hidden_states.size(0)
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(
                encoder_hidden_states.size()[:-1], dtype=torch.bool, device=encoder_hidden_states.device
            )
        outputs = self.bert(
            inputs_embeds=query_tokens,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True,
        )
        return outputs.last_hidden_state  # [B, Q, H]


def qformer(cfg):
    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/blip_2/configuration_blip_2.py
    # https://github.com/salesforce/LAVIS/blob/main/lavis/models/blip2_models/blip2_qformer.py
    bert_model_name = cfg['qformer']['bert_model_name']
    num_query_tokens = cfg['qformer']['num_query_tokens']
    hidden_size = cfg['qformer']['hidden_size']
    encoder_width = cfg['qformer']['encoder_width']
    cross_attention_freq = cfg['qformer']['cross_attention_freq']
    model = QFormer(bert_model_name, num_query_tokens, hidden_size, encoder_width, cross_attention_freq)
    return model
