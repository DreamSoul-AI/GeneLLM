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
        if os.path.exists(os.path.join(cache_dir, cfg['model_name'], key)):
            local_files_only[key] = True
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True,
                                              cache_dir=cache_tokenizer_path,
                                              model_max_length=cfg['task_max_length'],
                                              local_files_only=local_files_only['tokenizer'],
                                              use_fast=True, padding_side=cfg['padding_side'])

    # model = AutoModelForSequenceClassification.from_pretrained(
    #     cache_model_path,  # Folder containing `pytorch_model.bin` and `config.json`
    #     trust_remote_code=True  # Optional, only for custom models
    # )
    # print(model)
    # exit()

    config = BertConfig.from_pretrained(model_name_or_path,
                                        cache_dir=cache_config_path, local_files_only=local_files_only['config'])
    config.torch_dtype = cfg['torch_dtype']
    # model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, trust_remote_code=True,
    #                                                            config=config,
    #                                                            cache_dir=cache_model_path,
    #                                                            local_files_only=local_files_only['model'],
    #                                                            torch_dtype=cfg['torch_dtype'])
    # print(model)
    # exit()
    model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, config=config,
                                      cache_dir=cache_model_path, local_files_only=local_files_only['model'],
                                      torch_dtype=cfg['torch_dtype'])
    # config = AutoConfig.from_pretrained("bert-base-uncased", num_labels=2)
    # model = AutoModelForSequenceClassification.from_config(config)
    # state_dict = torch.load(cache_model_path, map_location='cpu')
    # print(state_dict)
    # exit()
    # model.load_state_dict(state_dict)
    # print(model)
    # model = AutoModelForSequenceClassification.from_pretrained(cache_model_path, trust_remote_code=True, config=config)
    return model, tokenizer
