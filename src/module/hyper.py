import torch

from config import cfg


def process_control():
    """
    把 cfg 这个实验配置参数改得更容易实用；
    然后再加入训练时候需要用的 hyper parameters
    """
    cfg['data_name'] = cfg['control']['data_name']
    cfg['task_name'] = cfg['control']['task_name']
    cfg['subset_name'] = str(cfg['control']['subset_name'])
    cfg['model_name'] = cfg['control']['model_name']
    cfg['freeze'] = int(cfg['control']['freeze'])
    cfg['embedding_mode'] = cfg['control']['embedding_mode']

    # virus may be 8
    # batch_size = {'EMP': 8, 'mouse': 8, 'promcore': 8, 'prom300': 8, 'splice': 8, 'tf': 8, 'virus': 32}
    batch_size = {'EMP': 32, 'mouse': 32, 'promcore': 32, 'prom300': 32, 'splice': 32, 'tf': 32, 'virus': 128,
                  'all': 32}
    cfg['batch_size'] = batch_size[cfg['task_name']]
    cfg['step_period'] = 1
    cfg['num_steps'] = 30
    cfg['eval_period'] = 30
    cfg['eval'] = {}
    cfg['eval']['num_steps'] = 30
    # promcore/300 all, notata = 4
    num_epochs = {'EMP': 3, 'mouse': 5, 'promcore': 10, 'prom300': 10, 'splice': 5, 'tf': 3, 'virus': 8, 'all': 10}
    cfg['num_epochs'] = num_epochs[cfg['task_name']]
    # cfg['num_epochs'] = None  # for test

    cfg['collate_mode'] = 'dict'

    cfg['task_names'] = ['EMP', 'mouse', 'promcore', 'prom300', 'splice', 'tf', 'virus']
    cfg['subset_names'] = {
        'EMP': ['H3', 'H3K4me1', 'H3K4me2', 'H3K4me3', 'H3K9ac', 'H3K14ac', 'H3K36me3', 'H3K79me3', 'H4', 'H4ac'],
        # 'mouse': ['0', '1', '2', '3', '4'],
        'mouse': ['Ch12Nrf2Iggrab', 'Ch12Znf384hpa004051Iggrab', 'MelJundIggrab', 'MelMafkDm2p5dStd', 'MelNelfeIggrab'],
        'promcore': ['all', 'notata', 'tata'],
        'prom300': ['all', 'notata', 'tata'],
        'splice': ['reconstructed'],
        # 'tf': ['0', '1', '2', '3', '4'],
        'tf': ['wgEncodeEH000552', 'wgEncodeEH000606', 'wgEncodeEH001546', 'wgEncodeEH001776', 'wgEncodeEH002829'],
        'virus': ['covid'],
    }

    # 用 dataset indices 来表示数据集的索引，其实就是给每一个 dataset 一个二维的 id (task_index, subset_index),
    #   然后再把这个二维的 integer id 给 map 成一个一维的 integer id
    # task_index：跟上面 cfg['subset_task_namesnames'] 定义的顺序是一样的，i.e.['EMP', 'mouse', 'promcore', 'prom300', 'splice', 'tf', 'virus'] -> [0, 1, 2, 3, 4, 5, 6]
    # subset_index: 以此类推，跟上面 cfg['subset_names'] 定义的顺序是一样的，e.g. 对于 mouse: ['Ch12Nrf2Iggrab', 'Ch12Znf384hpa004051Iggrab', 'MelJundIggrab', 'MelMafkDm2p5dStd', 'MelNelfeIggrab'] -> [0, 1, 2, 3, 4]
    dataset_indices = {}
    index = 0
    for task_name in cfg['subset_names']:
        task_index = cfg['task_names'].index(task_name)
        for subset_name in cfg['subset_names'][task_name]:
            subset_index = cfg['subset_names'][task_name].index(subset_name)
            index_name = (task_index, subset_index)
            dataset_indices[index_name] = index
            index += 1
    cfg['dataset_indices'] = dataset_indices

    cfg['model'] = {}
    cfg['model']['model_source'] = 'transformer'
    cfg['model']['hub_model_identifier'] = ['Qwen']

    cfg['model']['model_name'] = cfg['model_name']
    cfg['model']['task_names'] = cfg['task_names']
    cfg['model']['subset_names'] = cfg['subset_names']
    cfg['model']['task_name'] = cfg['task_name']
    cfg['model']['subset_name'] = cfg['subset_name']

    cfg['model']['dnabert2'] = {'hidden_size': 768}
    cfg['model']['padding_side'] = 'right'
    cfg['model']['torch_dtype'] = torch.float32
    # https://github.com/MAGICS-LAB/DNABERT_2/blob/main/finetune/scripts/run_dnabert2.sh
    task_max_length = {'EMP': 128, 'mouse': 30, 'promcore': 20, 'prom300': 70, 'splice': 80, 'tf': 30, 'virus': 256}
    cfg['model']['task_max_length'] = task_max_length
    if cfg['task_name'] == 'all':
        cfg['model']['max_length'] = max(list(cfg['model']['task_max_length'].values()))
        cfg['model']['num_datasets'] = sum(len(v) for v in cfg['subset_names'].values())
        cfg['model']['num_targets'] = len(cfg['task_names'])
    else:
        if cfg['subset_name'] == 'all':
            cfg['model']['num_datasets'] = len(cfg['subset_names'][cfg['task_name']])
        else:
            cfg['model']['num_datasets'] = 1
        cfg['model']['num_targets'] = 1  # 模型输出的类别数量
        cfg['model']['max_length'] = cfg['model']['task_max_length'][cfg['task_name']]
    cfg['model']['freeze'] = cfg['freeze'] == 1
    cfg['model']['embedding_mode'] = cfg['embedding_mode']

    # https://github.com/MAGICS-LAB/DNABERT_2/blob/main/finetune/train.py
    # https://github.com/MAGICS-LAB/DNABERT_2/blob/main/finetune/scripts/run_dnabert2.sh
    tag = cfg['tag']
    cfg[tag] = {}
    cfg[tag]['optimizer'] = {}
    cfg[tag]['optimizer']['optimizer_name'] = 'AdamW'
    cfg[tag]['optimizer']['lr'] = 3e-5
    cfg[tag]['optimizer']['momentum'] = 0.9
    cfg[tag]['optimizer']['betas'] = (0.9, 0.999)
    cfg[tag]['optimizer']['weight_decay'] = 0.01
    cfg[tag]['optimizer']['nesterov'] = True
    cfg[tag]['optimizer']['test_batch_ratio'] = 4
    cfg[tag]['optimizer']['batch_size'] = {'train': cfg['batch_size'],
                                           'valid': cfg[tag]['optimizer']['test_batch_ratio'] * cfg['batch_size'],
                                           'test': cfg[tag]['optimizer']['test_batch_ratio'] * cfg['batch_size']}
    cfg[tag]['optimizer']['step_period'] = cfg['step_period']
    cfg[tag]['optimizer']['num_steps'] = cfg['num_steps']
    cfg[tag]['optimizer']['scheduler_name'] = 'LinearAnnealingLR'
    # cfg[tag]['optimizer']['scheduler_name'] = 'None'
    cfg[tag]['optimizer']['num_warmup_steps'] = 50
    return
