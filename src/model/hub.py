import os
import torch
import torch.nn as nn


def make_model_generate(cfg):
    if cfg['model_source'] == 'transformer':
        # huggingface-cli download --model Qwen/Qwen2.5-7B-Instruct --local_dir Qwen/Qwen2.5-7B-Instruct
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
        )
    elif cfg['model_source'] == 'modelscope':
        # modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir Qwen/Qwen2.5-7B-Instruct
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

    cache_dir = os.path.join('output', 'cache')
    cache_tokenizer_path = os.path.join(cache_dir, model_name, 'tokenizer')
    cache_model_path = os.path.join(cache_dir, model_name, 'model')
    local_files_only = {'model_config': False, 'generation_config': False, 'tokenizer': False, 'model': False}
    for key in local_files_only:
        if os.path.exists(os.path.join(cache_dir, model_name, key)):
            local_files_only[key] = True
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_tokenizer_path,
                                              local_files_only=local_files_only['tokenizer'])
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name,
                                                 cache_dir=cache_model_path,
                                                 local_files_only=local_files_only['model'])
    return model, tokenizer
