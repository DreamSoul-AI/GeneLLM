import os
from transformers import AutoTokenizer
from transformers.models.bert.configuration_bert import BertConfig
from transformers import AutoModel


def dnabert2(cfg):
    cache_dir = os.path.join('output', 'cache')
    cache_tokenizer_path = os.path.join(cache_dir, cfg['model_name'], 'tokenizer')
    cache_config_path = os.path.join(cache_dir, cfg['model_name'], 'config')
    cache_model_path = os.path.join(cache_dir, cfg['model_name'], 'model')
    if not os.path.exists(os.path.join(cache_dir, cfg['model_name'])):
        local_files_only = False
    else:
        local_files_only = True
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True,
                                              cache_dir=cache_tokenizer_path, local_files_only=local_files_only)
    # Load the configuration and model
    config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M", cache_dir=cache_config_path,
                                        local_files_only=local_files_only)
    model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, config=config,
                                      cache_dir=cache_model_path, local_files_only=local_files_only)
    model.tokenizer = tokenizer
    return model
