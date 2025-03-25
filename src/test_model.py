import argparse
import os
import torch
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from dataset import make_dataset, make_data_loader, process_dataset
from metric import make_logger
from model import make_core, make_model
from module import save, resume, to_device, process_control

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)


def main():
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        tag_list = [str(seeds[i]), cfg['control_name']]
        cfg['tag'] = '_'.join([x for x in tag_list if x])
        process_control()
        print('Experiment: {}'.format(cfg['tag']))
        runExperiment()
    return


def runExperiment():
    cfg['seed'] = int(cfg['tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    cfg['path'] = os.path.join('output', 'exp')
    cfg['tag_path'] = os.path.join(cfg['path'], cfg['tag'])
    cfg['checkpoint_path'] = os.path.join(cfg['tag_path'], 'checkpoint')
    cfg['best_path'] = os.path.join(cfg['tag_path'], 'best')
    dataset = make_dataset(cfg['data_name'], task_name=cfg['task_name'], subset_name=cfg['subset_name'])
    core, tokenizer = make_core(cfg['model'])
    dataset = process_dataset(dataset, tokenizer, merge_test=False)
    model = make_model(core, tokenizer, cfg['model'])
    result = resume(cfg['best_path'])
    if result is None:
        raise ValueError('No valid model, please train model first')
    cfg['step'] = result['cfg']['step']
    model = model.to(cfg['device'])
    model.load_state_dict(result['model'])
    if isinstance(dataset['test'], list):
        for i in range(len(dataset['test'])):
            task_name = dataset['test'][i].task_name
            subset_name = dataset['test'][i].subset_name
            task_idx = dataset['test'][i].task_idx
            tag_i = '{}_{}_{}'.format(cfg['tag'], task_name, subset_name)
            cfg['result_path'] = os.path.join('output', 'result', tag_i)
            cfg['logger_path'] = os.path.join('output', 'logger', 'test', 'runs', tag_i)
            dataset_i = {'test': dataset['test'][i]}
            data_loader = make_data_loader(dataset_i, cfg[cfg['tag']]['optimizer']['batch_size'])
            test_logger = make_logger(cfg['logger_path'], split=['train', 'valid', 'test'], data_name=cfg['data_name'],
                                      task_name=cfg['task_name'], run_mode='test')
            test('valid', data_loader['valid'], model, test_logger, task_name, subset_name, task_idx)
            test('test', data_loader['test'], model, test_logger, task_name, subset_name, task_idx)
            result = resume(cfg['checkpoint_path'])
            result = {'cfg': cfg, 'logger': {'train': result['logger'], 'test': test_logger.state_dict()}}
            save(result, cfg['result_path'])
    else:
        task_name = dataset['test'].task_name
        subset_name = dataset['test'].subset_name
        task_idx = dataset['test'].task_idx
        tag = '{}_{}_{}'.format(cfg['tag'], task_name, subset_name)
        cfg['result_path'] = os.path.join('output', 'result', tag)
        cfg['logger_path'] = os.path.join('output', 'logger', 'test', 'runs', tag)

        data_loader = make_data_loader(dataset, cfg[cfg['tag']]['optimizer']['batch_size'])
        test_logger = make_logger(cfg['logger_path'], split=['train', 'valid', 'test'], data_name=cfg['data_name'],
                                  task_name=cfg['task_name'],run_mode='test')
        test('valid', data_loader['valid'], model, test_logger, task_name, subset_name, task_idx)
        test('test', data_loader['test'], model, test_logger, task_name, subset_name, task_idx)
        result = resume(cfg['checkpoint_path'])
        result = {'cfg': cfg, 'logger': {'train': result['logger'], 'test': test_logger.state_dict()}}
        save(result, cfg['result_path'])
    return


def test(subset, data_loader, model, logger, task_name, subset_name, task_idx):
    with torch.no_grad():
        model.train(False)
        for i, input in enumerate(data_loader):
            input_size = len(input[list(input.keys())[0]])
            input = to_device(input, cfg['device'])
            input['test_task_idx'] = task_idx
            output = model(**input)
            evaluation = logger.evaluate(subset, 'batch', input, output)
            logger.append(evaluation, subset, input_size)
            logger.add(subset, input, output)
        evaluation = logger.evaluate(subset, 'full')
        logger.append(evaluation, subset, input_size)
        info = {'info': ['Model: {}({}, {})'.format(cfg['tag'], task_name, subset_name),
                         'Test Epoch (): {}({:.0f}%)'.format(subset, cfg['step'] // cfg['eval_period'], 100.)]}
        logger.append(info, subset)
        print(logger.write(subset))
        logger.save(True)
    return


if __name__ == "__main__":
    main()
