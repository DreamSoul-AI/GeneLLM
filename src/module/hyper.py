from config import cfg


def process_control():
    cfg['data_name'] = cfg['control']['data_name']
    cfg['task_name'] = cfg['control']['task_name']
    cfg['subset_name'] = str(cfg['control']['subset_name'])
    cfg['model_name'] = cfg['control']['model_name']

    cfg['batch_size'] = 8
    cfg['step_period'] = 1
    cfg['num_steps'] = 30
    cfg['eval_period'] = 30
    cfg['eval'] = {}
    cfg['eval']['num_steps'] = 30
    # cfg['num_epochs'] = 5
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

    cfg['model'] = {}
    cfg['model']['model_name'] = cfg['model_name']
    cfg['model']['task_names'] = cfg['task_names']
    cfg['model']['subset_names'] = cfg['subset_names']
    cfg['model']['dnabert2'] = {'hidden_size': 768}
    # TODO: this inconsistent with data size
    max_length = {'EMP': 128, 'mouse': 30, 'promcore': 20, 'prom300': 70, 'splice': 80, 'tf': 30, 'virus': 256}
    cfg['model']['padding_side'] = 'right'
    cfg['model']['freeze'] = False
    if cfg['task_name'] == -1:
        cfg['model']['max_length'] = max(list(max_length.values()))
    else:
        cfg['model']['max_length'] = max_length[cfg['task_name']]
    if cfg['task_name'] == -1:
        cfg['model']['num_targets'] = len(cfg['task_names'])
    else:
        cfg['model']['num_targets'] = 1

    # https://github.com/MAGICS-LAB/DNABERT_2/blob/main/finetune/train.py
    # https://github.com/MAGICS-LAB/DNABERT_2/blob/main/finetune/scripts/run_dnabert2.sh
    tag = cfg['tag']
    cfg[tag] = {}
    cfg[tag]['optimizer'] = {}
    cfg[tag]['optimizer']['optimizer_name'] = 'AdamW'
    cfg[tag]['optimizer']['lr'] = 1e-3
    cfg[tag]['optimizer']['momentum'] = 0.9
    cfg[tag]['optimizer']['betas'] = (0.9, 0.999)
    cfg[tag]['optimizer']['weight_decay'] = 0.01
    cfg[tag]['optimizer']['nesterov'] = True
    cfg[tag]['optimizer']['batch_size'] = {'train': cfg['batch_size'], 'valid': cfg['batch_size'],
                                           'test': cfg['batch_size']}
    cfg[tag]['optimizer']['step_period'] = cfg['step_period']
    cfg[tag]['optimizer']['num_steps'] = cfg['num_steps']
    # cfg[tag]['optimizer']['scheduler_name'] = 'LinearAnnealingLR'
    cfg[tag]['optimizer']['scheduler_name'] = 'None'
    # cfg[tag]['optimizer']['num_warmup_steps'] = 50
    return
