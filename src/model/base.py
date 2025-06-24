import os
import torch
import torch.nn as nn
from module import filter_args, load
from .loss import make_loss
from .qformer import QFormer


class DataEmbedding(nn.Module):
    def __init__(self, embedding_mode, task_name, task_names, num_datasets, hidden_size):
        super().__init__()
        self.embedding_mode = embedding_mode
        self.task_name = task_name
        self.task_names = task_names
        self.hidden_size = hidden_size
        self.num_datasets = num_datasets

        if self.embedding_mode == 'index':
            self.dataset_embedding = nn.Embedding(num_datasets,
                                                  hidden_size)  # 给每一个 sebsets 加一个 index embedding。 e.g.EMP_all_index, EMP 里面有10个 subsets, 为每一个 subset 分配一个 embedding vector(size 也是768)
            nn.init.normal_(self.dataset_embedding.weight, mean=0.0, std=1e-4)  # init embedding vector
        elif self.embedding_mode == 'word' and self.task_name == 'all':
            embedding = []
            for task_name in self.task_names:
                embedding_i = load(os.path.join('data', 'GUE', 'description_embedding', task_name))['pooler_output']
                embedding.append(embedding_i)
            embedding = torch.cat(embedding, dim=0)
            word_embedding_size = embedding.size(-1)
            task_embedding = nn.Embedding(len(self.task_names), word_embedding_size)
            task_embedding.weight.data.copy_(embedding.data)
            task_embedding.weight.requires_grad = False
            self.task_embedding = nn.Sequential(task_embedding, nn.Linear(word_embedding_size, hidden_size))
        else:
            self.dataset_embedding = None

    def forward(self, dataset_idx, task_idx=None):
        if self.embedding_mode == 'index':
            data_embedding = self.dataset_embedding(dataset_idx)
        elif self.embedding_mode == 'word' and self.task_name == 'all':
            data_embedding = self.task_embedding(task_idx)
        else:
            data_embedding = 0
        return data_embedding


class BertBase(nn.Module):
    def __init__(self, model, hidden_size, target_size, num_datasets, num_targets, task_names, subset_names, task_name,
                 subset_name, freeze, embedding_mode):  # 这里的 model 就是 core model, 就是 DNABert2 用的那个 BERT。
        super().__init__()
        self.model = model
        self.hidden_size = hidden_size
        self.target_size = target_size
        self.num_datasets = num_datasets
        self.num_targets = num_targets
        self.task_names = task_names
        self.subset_names = subset_names
        self.task_name = task_name
        self.subset_name = subset_name
        self.freeze = freeze
        if self.freeze == 1:
            self.freeze(self.model)
        self.embedding_mode = embedding_mode
        self.dataset_embedding = DataEmbedding(self.embedding_mode, self.task_name, self.task_names, self.num_datasets,
                                               self.hidden_size)
        if num_targets == 1:  # 这个就是分类头的数量
            self.output_proj = nn.Linear(hidden_size, target_size)
        else:
            output_proj = []
            for i in range(num_targets):
                target_size_i = target_size[task_names[i]]
                output_proj.append(nn.Linear(hidden_size, target_size_i))
            self.output_proj = nn.ModuleList(output_proj)
        self.loss = make_loss  # 定义 loss 的计算图。

    def freeze(self, model):
        for param in model.parameters():
            param.requires_grad = False
        return

    def forward(self, **input):
        output = {}
        # https://github.com/mosaicml/examples/blob/main/examples/benchmarks/bert/src/bert_layers.py
        # 用 filter_args 函数筛选 input 字典，只保留 self.model.forward 方法所需要的参数。
        valid_input = filter_args(self.model.forward, input)
        if self.freeze:
            with torch.no_grad():
                encoder_outputs, pooled_output = self.model(**valid_input)
        else:
            encoder_outputs, pooled_output = self.model(**valid_input)
        # decoder_input = encoder_outputs.mean(dim=1)
        decoder_input = pooled_output

        if self.embedding_mode != 'none':
            dataset_embedding = self.dataset_embedding(input['dataset_idx'], input['task_idx'])
            decoder_input = decoder_input + dataset_embedding  # 这里的 + 是 element-wise add

        if self.num_targets == 1:
            output['pred'] = self.output_proj(decoder_input)
            output['loss'] = self.loss(output['pred'], input['target'])
        else:
            loss = 0
            unique_task_idx = torch.unique(input['task_idx'])
            output['pred'] = {}
            output['loss_task'] = {}
            for i in range(len(unique_task_idx)):
                task_idx = unique_task_idx[i].item()
                mask_i = input['task_idx'] == task_idx
                output_i = self.output_proj[task_idx](decoder_input[mask_i])
                target_i = input['target'][mask_i]
                output['pred'][task_idx] = output_i
                loss_i = self.loss(output_i, target_i, reduction='sum')
                loss += loss_i
            output['loss'] = loss / len(encoder_outputs)
            if not self.training and 'test_task_idx' in input:
                output['pred'] = output['pred'][input['test_task_idx']]
        return output


class LLMBase(nn.Module):
    def __init__(self, gene_encoder, llm, hidden_size, target_size,
                 num_datasets, num_targets, task_names, subset_names,
                 task_name, subset_name, freeze, embedding_mode, num_query_tokens):
        super().__init__()
        self.gene_encoder = gene_encoder
        self.llm = llm
        self.hidden_size = hidden_size
        self.target_size = target_size
        self.num_datasets = num_datasets
        self.num_targets = num_targets
        self.task_names = task_names
        self.subset_names = subset_names
        self.task_name = task_name
        self.subset_name = subset_name
        self.freeze = freeze
        self.embedding_mode = embedding_mode
        self.num_query_tokens = num_query_tokens
        if self.freeze >= 1:
            for p in llm.parameters():
                p.requires_grad = False
            if self.freeze > 1:
                for p in gene_encoder.parameters():
                    p.requires_grad = False
        self.qformer = QFormer(
            num_query_tokens=num_query_tokens,
            hidden_size=hidden_size,
            encoder_width=gene_encoder.config.hidden_size
        )
        llm_hidden_size = llm.config.hidden_size
        self.gene_proj = nn.Linear(hidden_size, llm_hidden_size)
        self.loss = make_loss
        exit()

    def forward(self, **input):
        # Step 1: Encode gene sequence
        gene_input = filter_args(self.gene_encoder.forward, input)
        with torch.no_grad() if self.freeze else torch.enable_grad():
            gene_output = self.gene_encoder(**gene_input)[0]  # [B, L, H]

        # Step 2: Q-Former attends to gene features
        q_output = self.qformer(gene_output)  # [B, Q, H_qformer]
        q_output_proj = self.vision_proj(q_output)  # [B, Q, H_lm]

        if self.embedding_mode != 'none':
            dataset_embedding = self.dataset_embedding(input['dataset_idx'], input.get('task_idx'))
            q_output_proj = q_output_proj + dataset_embedding.unsqueeze(1)

        # Step 3: LLM (or just average)
        if self.llm:
            with torch.no_grad() if self.freeze else torch.enable_grad():
                llm_output = self.llm(inputs_embeds=q_output_proj)
                pooled = llm_output.last_hidden_state.mean(dim=1)
        else:
            pooled = q_output_proj.mean(dim=1)

        # Step 4: Classification
        output = {}
        if self.num_targets == 1:
            output['pred'] = self.output_proj(pooled)
            output['loss'] = self.loss(output['pred'], input['target'])
        else:
            loss = 0
            output['pred'] = {}
            output['loss_task'] = {}
            unique_task_idx = torch.unique(input['task_idx'])
            for i in unique_task_idx:
                i = i.item()
                mask = input['task_idx'] == i
                pred_i = self.output_proj[i](pooled[mask])
                loss += self.loss(pred_i, input['target'][mask], reduction='sum')
                output['pred'][i] = pred_i
            output['loss'] = loss / pooled.size(0)
            if not self.training and 'test_task_idx' in input:
                output['pred'] = output['pred'][input['test_task_idx']]
        return output


def base(cfg, gene_encoder, llm_encoder=None):
    """
    Create a base model (a computation graph) based on the configuration. 这里就是在 core model 的基础上，根据 cfg 的值来决定是否需要添加其他的模块，比如分类头，或者其他的任务相关的模块。
    """
    hidden_size = cfg['dnabert2']['hidden_size']
    target_size = cfg['target_size']
    num_datasets = cfg['num_datasets']
    num_targets = cfg['num_targets']
    task_names = cfg['task_names']
    subset_names = cfg['subset_names']
    task_name = cfg['task_name']
    subset_name = cfg['subset_name']
    freeze = cfg['freeze']
    embedding_mode = cfg['embedding_mode']
    if cfg['model_name'] in ['dnabert2']:
        model = BertBase(gene_encoder, hidden_size, target_size, num_datasets, num_targets, task_names, subset_names,
                         task_name,
                         subset_name, freeze, embedding_mode)  # 定义 BertBase 这个 model,i.e. a computation graph
    else:
        num_query_tokens = cfg['num_query_tokens']
        # https://github.com/salesforce/LAVIS/blob/main/lavis/models/blip2_models/blip2_qformer.py
        model = LLMBase(gene_encoder, llm_encoder, hidden_size, target_size,
                        num_datasets, num_targets,
                        task_names, subset_names,
                        task_name, subset_name, freeze, embedding_mode, num_query_tokens)
    return model
