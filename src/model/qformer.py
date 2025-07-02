import os
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

# https://github.com/salesforce/LAVIS/blob/506965b9c4a18c1e565bd32acaccabe0198433f7/lavis/models/blip2_models/blip2_qformer.py#L27
class QFormer(nn.Module):
    def __init__(self, backbone, num_query_tokens, hidden_size):
        super().__init__()
        self.backbone = backbone
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, hidden_size))

    def forward(self, encoder_hidden_states, encoder_attention_mask=None):
        batch_size = encoder_hidden_states.size(0)
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(
                encoder_hidden_states.size()[:-1], dtype=torch.bool, device=encoder_hidden_states.device
            )
        output = self.backbone(
            inputs_embeds=query_tokens,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=True,
            return_dict=True,
        )
        return output.last_hidden_state


def qformer(cfg):
    if cfg['model_source'] == 'huggingface':
        from huggingface_hub import snapshot_download as hf_snapshot_download
        identifier = 'google-bert'
    elif cfg['model_source'] == 'modelscope':
        from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download
        identifier = 'AI-ModelScope'
    else:
        raise ValueError('Not valid model source')

    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/blip_2/configuration_blip_2.py
    # https://github.com/salesforce/LAVIS/blob/main/lavis/models/blip2_models/blip2_qformer.py
    bert_model_name = cfg['qformer']['bert_model_name']
    num_query_tokens = cfg['qformer']['num_query_tokens']
    hidden_size = cfg['qformer']['hidden_size']
    encoder_width = cfg['qformer']['encoder_width']
    cross_attention_freq = cfg['qformer']['cross_attention_freq']
    model_name = '{}/{}'.format(identifier, bert_model_name)
    snapshot_path = os.path.join('output', 'snapshot', identifier, bert_model_name)
    if not os.path.exists(os.path.join(snapshot_path)):
        if cfg['model_source'] == 'huggingface':
            hf_snapshot_download(repo_id=model_name, local_files_only=False, local_dir=snapshot_path)
        elif cfg['model_source'] == 'modelscope':
            snapshot_path = os.path.join('output', 'snapshot', identifier, bert_model_name)
            ms_snapshot_download(repo_id=model_name, local_files_only=False, local_dir=snapshot_path)
        else:
            raise ValueError('Not valid model source')

    config = BertConfig.from_pretrained(snapshot_path, is_decoder=True)
    config.add_cross_attention = True
    config.cross_attention_freq = cross_attention_freq
    config.encoder_width = encoder_width
    backbone = BertModel.from_pretrained(snapshot_path, config=config)
    model = QFormer(backbone, num_query_tokens, hidden_size)
    return model
