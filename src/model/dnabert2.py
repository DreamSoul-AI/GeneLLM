import torch
import os
from transformers import AutoTokenizer
from transformers.models.bert.configuration_bert import BertConfig
from transformers import AutoModel, AutoModelForSequenceClassification, AutoConfig


def dnabert2(cfg):
    model_name_or_path = 'zhihan1996/DNABERT-2-117M'
    cache_dir = os.path.join('output', 'cache')
    cache_config_path = os.path.join(cache_dir, cfg['model_name'], 'config')
    cache_tokenizer_path = os.path.join(cache_dir, cfg['model_name'], 'tokenizer')
    # https://huggingface.co/zhihan1996/DNABERT-2-117M/tree/main
    cache_model_path = os.path.join(cache_dir, cfg['model_name'], 'model')
    local_files_only = {'config': False, 'tokenizer': False, 'model': False}
    for key in local_files_only:
        if os.path.exists(os.path.join(cache_dir, cfg['model_name'], key)): # if already downloaded,use local files only for tokenizer, config and model
            local_files_only[key] = True
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True,
                                              cache_dir=cache_tokenizer_path,
                                              model_max_length=cfg['task_max_length'],
                                              local_files_only=local_files_only['tokenizer'],
                                              use_fast=True, padding_side=cfg['padding_side'])
    config = BertConfig.from_pretrained(model_name_or_path,
                                        cache_dir=cache_config_path, local_files_only=local_files_only['config'])
    config.torch_dtype = cfg['torch_dtype']
    model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, config=config,
                                      cache_dir=cache_model_path, local_files_only=local_files_only['model'],
                                      torch_dtype=cfg['torch_dtype'])
    return model, tokenizer
