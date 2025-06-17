import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, matthews_corrcoef


def make_metric(split, **kwargs):
    data_name = kwargs['data_name']
    task_name = kwargs['task_name']
    run_mode = kwargs['run_mode']
    metric_name = {k: [] for k in split}
    if data_name in ['GUE']:
        best_direction = 'down'
        best_metric_name = 'Loss'
        if run_mode == 'train':
            for k in metric_name:
                if k == 'train':
                    if task_name == 'all' or isinstance(task_name, list):
                        metric_name[k].extend(['Loss'])
                    else:
                        metric_name[k].extend(['Loss', 'Accuracy'])
                else:
                    if task_name == 'all' or isinstance(task_name, list):
                        metric_name[k].extend(['Loss'])
                    else:
                        metric_name[k].extend(['Loss', 'Accuracy', 'F1', 'MCC'])
        else:
            for k in metric_name:
                if k == 'train':
                    metric_name[k].extend(['Loss', 'Accuracy'])
                else:
                    metric_name[k].extend(['Loss', 'Accuracy', 'F1', 'MCC'])
    else:
        raise ValueError('Not valid data name')
    metric = Metric(metric_name, best_direction, best_metric_name)
    return metric


class BaseMetric:
    def __init__(self):
        super().__init__()

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class Loss(BaseMetric):
    def __call__(self, loss):
        with torch.no_grad():
            loss = loss.item()
        return loss


class Accuracy(BaseMetric):
    def __init__(self, topk=1):
        super().__init__()
        self.topk = topk

    def __call__(self, pred, target):
        with torch.no_grad():
            if target.dtype != torch.int64:
                target = (target.topk(1, -1, True, True)[1]).view(-1)
            batch_size = torch.numel(target)
            if pred.dtype != torch.int64:
                pred_k = pred.topk(self.topk, -1, True, True)[1]
                correct_k = pred_k.eq(target.unsqueeze(-1).expand_as(pred_k)).float().sum()
            else:
                correct_k = pred.eq(target).float().sum()
            acc = (correct_k * (100.0 / batch_size)).item()
        return acc


class F1(BaseMetric):
    def __call__(self, pred, target):
        with torch.no_grad():
            if pred.dtype != torch.int64:
                pred = pred.argmax(dim=-1)
            f1 = f1_score(target.cpu().numpy(), pred.cpu().numpy(), average='macro', zero_division=0)
        return f1


class MCC(BaseMetric):
    def __call__(self, pred, target):
        with torch.no_grad():
            if pred.dtype != torch.int64:
                pred = pred.argmax(dim=-1)
            mcc = matthews_corrcoef(target.cpu().numpy(), pred.cpu().numpy())
        return mcc


class MSE(BaseMetric):
    def __call__(self, pred, target):
        with torch.no_grad():
            mse = F.mse_loss(pred, target).item()
        return mse


class RMSE(BaseMetric):
    def __call__(self, pred, target):
        with torch.no_grad():
            rmse = F.mse_loss(pred, target).sqrt().item()
        return rmse


class Metric:
    def __init__(self, metric_name, best_direction, best_metric_name):
        self.metric_name = metric_name
        self.best_direction, self.best_metric_name = best_direction, best_metric_name
        self.metric, self.mode, self.mode_keys = self.make_metric(metric_name)
        self.full_mode_keys = self.make_full_mode(self.mode, self.mode_keys)
        self.reset()

    def make_metric(self, metric_name):
        metric = {}
        mode = {}
        mode_keys = {}
        for split in metric_name:
            metric[split] = {}
            mode[split] = {}
            mode_keys[split] = {}
            for metric_name_i in metric_name[split]:
                metric[split][metric_name_i] = eval('{}()'.format(metric_name_i))
                mode_keys[split][metric_name_i] = {'input': set(), 'output': set()}
                if metric_name_i in ['Loss']:
                    mode[split][metric_name_i] = 'batch'
                    mode_keys[split][metric_name_i]['output'].add('loss')
                elif metric_name_i in ['Accuracy', 'MSE', 'MAE', 'MBE', 'MPE']:
                    mode[split][metric_name_i] = 'batch'
                    mode_keys[split][metric_name_i]['input'].add('target')
                    mode_keys[split][metric_name_i]['output'].add('pred')
                elif metric_name_i in ['RMSE', 'R2', 'Correlation', 'ResidualMean', 'ResidualStd', 'ResidualSkewness',
                                       'ResidualKurtosis', 'F1', 'MCC']:
                    mode[split][metric_name_i] = 'full'
                    mode_keys[split][metric_name_i]['input'].add('target')
                    mode_keys[split][metric_name_i]['output'].add('pred')
                else:
                    raise ValueError('Not valid metric name')
        return metric, mode, mode_keys

    def make_init_best(self):
        if self.best_direction == 'up':
            init_best = -float('inf')
        elif self.best_direction == 'down':
            init_best = float('inf')
        else:
            raise ValueError('Not valid best direction')
        return init_best

    def make_full_mode(self, mode, mode_keys):
        full_mode_keys = {}
        for split in mode:
            full_mode_keys[split] = {'input': set(), 'output': set()}
            for metric_name_i in mode[split]:
                if mode[split][metric_name_i] == 'full':
                    full_mode_keys[split]['input'].update(mode_keys[split][metric_name_i]['input'])
                    full_mode_keys[split]['output'].update(mode_keys[split][metric_name_i]['output'])
        return full_mode_keys

    def add(self, split, input, output):
        with torch.no_grad():
            for key in self.full_mode_keys[split]['input']:
                if key not in self.buffer['input']:
                    self.buffer['input'][key] = input[key]
                else:
                    self.buffer['input'][key] = torch.cat([self.buffer['input'][key], input[key]], dim=0)
            for key in self.full_mode_keys[split]['output']:
                if key not in self.buffer['output']:
                    self.buffer['output'][key] = output[key]
                else:
                    self.buffer['output'][key] = torch.cat([self.buffer['output'][key], output[key]], dim=0)
        return

    def evaluate(self, split, mode, input=None, output=None, metric_name=None):
        metric_name = self.metric_name if metric_name is None else metric_name
        evaluation = {}
        if mode == 'batch':
            for metric_name_i in metric_name[split]:
                if self.mode[split][metric_name_i] == mode:
                    input_ = {key: input[key] for key in self.mode_keys[split][metric_name_i]['input']}
                    output_ = {key: output[key] for key in self.mode_keys[split][metric_name_i]['output']}
                    evaluation[metric_name_i] = self.metric[split][metric_name_i](**input_, **output_)
        elif mode == 'full':
            for metric_name_i in metric_name[split]:
                if self.mode[split][metric_name_i] == mode:
                    input_ = {key: self.buffer['input'][key] for key in
                              self.mode_keys[split][metric_name_i]['input']}
                    output_ = {key: self.buffer['output'][key] for key in
                               self.mode_keys[split][metric_name_i]['output']}
                    evaluation[metric_name_i] = self.metric[split][metric_name_i](**input_, **output_)
            self.reset_buffer()
        else:
            raise ValueError('Not valid mode')
        return evaluation

    def compare(self, val, if_update):
        if self.best_direction == 'down':
            compared = self.best > val
        elif self.best_direction == 'up':
            compared = self.best < val
        else:
            raise ValueError('Not valid best direction')
        if compared and if_update:
            self.best = val
        return compared

    def reset(self):
        self.reset_best()
        self.reset_buffer()
        return

    def reset_best(self):
        self.best = self.make_init_best()
        return

    def reset_buffer(self):
        self.buffer = {'input': {}, 'output': {}}
        return

    def load_state_dict(self, state_dict):
        self.best_metric_name = state_dict['best_metric_name']
        self.best_direction = state_dict['best_direction']
        self.reset_best()
        return

    def state_dict(self):
        return {'best_metric_name': self.best_metric_name, 'best_direction': self.best_direction}
