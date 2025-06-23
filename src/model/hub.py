import os
import torch
import torch.nn as nn


def make_model_generate(cfg):
    if cfg['model_source'] == 'huggingface':
        from huggingface_hub import snapshot_download as hf_snapshot_download
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
        )
    elif cfg['model_source'] == 'modelscope':
        from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download
        from modelscope import (
            AutoModelForCausalLM,
            AutoTokenizer,
        )
    else:
        raise ValueError('Not valid model source')

    for identifier in cfg['hub_model_identifier']:
        if identifier in cfg['model_name']:
            model_name = '{}/{}'.format(identifier, cfg['model_name'])
            break

    cfg['snapshot_path'] = os.path.join('output', 'snapshot', identifier, cfg['model_name'])
    snapshot_path = cfg['snapshot_path']

    if not os.path.exists(os.path.join(snapshot_path)):
        if cfg['model_source'] == 'huggingface':
            hf_snapshot_download(repo_id=model_name, local_files_only=False, local_dir=snapshot_path)
        elif cfg['model_source'] == 'modelscope':
            ms_snapshot_download(repo_id=model_name, local_files_only=False, local_dir=snapshot_path)
        else:
            raise ValueError('Not valid model source')
    tokenizer = AutoTokenizer.from_pretrained(snapshot_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=snapshot_path)
    return model, tokenizer
