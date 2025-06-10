import argparse
import itertools
import os

parser = argparse.ArgumentParser(description='config')
parser.add_argument('--run', default='train', type=str)
parser.add_argument('--init_gpu', default=0, type=int)
parser.add_argument('--num_gpus', default=1, type=int)
parser.add_argument('--init_seed', default=0, type=int)
parser.add_argument('--round', default=4, type=int)
parser.add_argument('--experiment_step', default=1, type=int)
parser.add_argument('--num_experiments', default=1, type=int)
parser.add_argument('--resume_mode', default=0, type=int)
parser.add_argument('--split_round', default=65535, type=int)
parser.add_argument('--task_name', default=None, type=str)
parser.add_argument('--subset_name', default=None, type=str)
parser.add_argument('--mode', default=None, type=str)
args = vars(parser.parse_args())


def make_controls(script_name, init_seeds, num_experiments, resume_mode, control_name):
    control_names = []
    for i in range(len(control_name)):
        control_names.extend(list('_'.join(x) for x in itertools.product(*control_name[i])))
    control_names = [control_names]
    controls = script_name + init_seeds + num_experiments + resume_mode + control_names
    controls = list(itertools.product(*controls))
    return controls


def main():
    run = args['run']
    init_gpu = args['init_gpu']
    num_gpus = args['num_gpus']
    round = args['round']
    experiment_step = args['experiment_step']
    init_seed = args['init_seed']
    num_experiments = args['num_experiments']
    resume_mode = args['resume_mode']
    split_round = args['split_round']
    task_name = args['task_name']
    subset_name = args['subset_name']
    mode = args['mode']

    script_path = os.path.join('output', 'script')
    if num_gpus > 0:
        gpu_ids = [','.join(str(i) for i in list(range(x, x + 1))) for x in
                   list(range(init_gpu, init_gpu + num_gpus))]
    init_seeds = [list(range(init_seed, init_seed + num_experiments, experiment_step))]
    num_experiments = [[experiment_step]]
    resume_mode = [[resume_mode]]
    filename = '{}_{}_{}'.format(run, task_name, subset_name)

    subset_names = {
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

    script_name = [['{}_model.py'.format(run)]]
    data_name = ['GUE']
    task_name = [task_name]
    if subset_name == 'split':
        subset_name = subset_names[task_name[0]]
    else:
        subset_name = [subset_name]
    if mode == 'base':
        model_name = ['dnabert2']
        freeze = ['0']
        embedding_mode = ['none']
        control_name = [[data_name, task_name, subset_name, model_name, freeze, embedding_mode]]
    elif mode == 'embedding':
        model_name = ['dnabert2']
        freeze = ['0']
        if task_name[0] == 'all':
            embedding_mode = ['index', 'word']
        else:
            embedding_mode = ['index']
        control_name = [[data_name, task_name, subset_name, model_name, freeze, embedding_mode]]
    else:
        raise ValueError('Not valid mode')
    controls = make_controls(script_name, init_seeds, num_experiments, resume_mode, control_name)

    # the following is for generating the bash script for each experiment
    s = '#!/bin/bash\n'  # this s is the bash script
    j = 1
    k = 1
    for i in range(len(controls)):
        controls[i] = list(controls[i])
        if num_gpus > 0:
            s = s + 'CUDA_VISIBLE_DEVICES=\"{}\" python {} --init_seed {} --num_experiments {} ' \
                    '--resume_mode {} --device cuda ' \
                    '--control_name {}&\n'.format(gpu_ids[i % len(gpu_ids)], *controls[i])
        else:
            s = s + 'python {} --init_seed {} --num_experiments {} ' \
                    '--resume_mode {} --device cpu --control_name {}&\n'.format(*controls[i])
        if i % round == round - 1:
            s = s[:-2] + '\nwait\n'
            if j % split_round == 0:
                print(s)
                if not os.path.exists(script_path):
                    os.makedirs(script_path)
                run_file = open(os.path.join(script_path, '{}_{}.sh'.format(filename, k)), 'w', newline='\n')
                run_file.write(s)
                run_file.close()
                s = '#!/bin/bash\n'
                k = k + 1
            j = j + 1
    if s != '#!/bin/bash\n':
        if s[-5:-1] != 'wait':
            s = s + 'wait\n'
        print(s)
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        run_file = open(os.path.join(script_path, '{}_{}.sh'.format(filename, k)), 'w', newline='\n')
        run_file.write(s)
        run_file.close()
    return


if __name__ == '__main__':
    main()
