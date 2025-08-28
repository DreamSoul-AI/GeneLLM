import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from module import filter_args, load
from .loss import make_loss


class DataEmbedding(nn.Module):
    def __init__(self, embedding_mode, task_name, task_names, num_datasets, hidden_size):
        super().__init__()
        self.embedding_mode = embedding_mode
        self.task_name = task_name
        self.task_names = task_names
        self.hidden_size = hidden_size
        self.num_datasets = num_datasets

        if self.embedding_mode == 'index':
            # 给每一个 sebsets 加一个 index embedding。 e.g.EMP_all_index, EMP 里面有10个 subsets, 为每一个 subset 分配一个 embedding vector(size 也是768)
            self.dataset_embedding = nn.Embedding(num_datasets, hidden_size) # 这里其实就是给每一个 datasets 一个 embedding 表示 
            nn.init.normal_(self.dataset_embedding.weight, mean=0.0, std=1e-4)  # init embedding vector
        elif self.embedding_mode == 'word':
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
        elif self.embedding_mode == 'word':
            data_embedding = self.task_embedding(task_idx)
        else:
            data_embedding = 0
        return data_embedding


class BertBase(nn.Module):
    """
    这个 BertBase 其实就是在 DNABert2 这个模型的基础上，加了一个或者多个分类头，和 dataset embedding。
    所以，可以看出来一些如何在已有模型上去继续构建更大模型的做法，就是这样，用一个 nn.Module 去包裹已有的模型，然后在 forward 里面调用已有模型的 forward 方法，得到输出后，再接上其他的模块，形成一个新的计算图。
    """

    def __init__(self, model, gene_tokenizer, hidden_size, target_size, num_datasets, num_targets, task_names,
                 subset_names, task_name, subset_name, freeze, embedding_mode):
        super().__init__()
        self.model = model
        self.tokenizer = {'gene': gene_tokenizer}
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
            for p in self.model.parameters():
                p.requires_grad = False
        self.embedding_mode = embedding_mode
        self.dataset_embedding = DataEmbedding(self.embedding_mode, self.task_name, self.task_names, self.num_datasets,
                                               self.hidden_size)
        if num_targets == 1:  # 这个就是分类头的数量
            self.output_proj = nn.Linear(hidden_size, target_size)
        else:
            output_proj = []
            task_idx_mapping = {}
            for i in range(num_targets):
                target_size_i = target_size[self.task_name[i]]
                task_idx_mapping[self.task_names.index(self.task_name[i])] = i
                output_proj.append(nn.Linear(hidden_size, target_size_i))
            self.task_idx_mapping = task_idx_mapping
            self.output_proj = nn.ModuleList(output_proj)
        self.loss = make_loss  # 定义 loss 的计算图。

    def forward(self, **input):
        output = {}
        # https://github.com/mosaicml/examples/blob/main/examples/benchmarks/bert/src/bert_layers.py
        # 用 filter_args 函数筛选 input 字典，只保留 self.model.forward 方法所需要的参数。
        # 注意，这里是 self.model.forward 这个函数，而不是 self.forward!
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
            output['pred'] = {}
            output['loss_task'] = {}
            unique_task_idx = torch.unique(input['task_idx'])
            for i in range(len(unique_task_idx)):
                task_idx = unique_task_idx[i].item()
                mask_i = input['task_idx'] == task_idx
                task_idx = self.task_idx_mapping[task_idx] # revised order
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
    def __init__(self, gene_encoder, gene_tokenizer, qformer, llm, llm_tokenizer, hidden_size, target_size,
                 num_datasets, num_targets, task_names, subset_names,
                 task_name, subset_name, freeze, embedding_mode):
        super().__init__()
        self.gene_encoder = gene_encoder
        self.qformer = qformer
        self.llm = llm
        self.tokenizer = {'gene': gene_tokenizer, 'llm': llm_tokenizer}
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
        if self.freeze >= 1:
            for p in llm.parameters():
                p.requires_grad = False
            if self.freeze > 1:
                for p in gene_encoder.parameters():
                    p.requires_grad = False

        # TODO: change to data transform
        # instruction_token = []
        # for task_name in self.task_names:
        #     instruction_token_i = load(os.path.join('data', 'GUE', 'instruction_token', task_name))
        #     instruction_token.append({'input_ids': instruction_token_i})
        #
        # instruction_token = llm_tokenizer.pad(
        #     instruction_token,
        #     padding='longest',
        #     return_tensors='pt'
        # )
        # self.register_buffer('instruction_token_input_ids', instruction_token['input_ids'])
        # self.register_buffer('instruction_token_attention_mask', instruction_token['attention_mask'])

        llm_hidden_size = llm.config.hidden_size
        self.gene_proj = nn.Linear(hidden_size, llm_hidden_size)

        if num_targets > 1:
            task_idx_mapping = {}
            for i in range(num_targets):
                task_idx_mapping[self.task_names.index(self.task_name[i])] = i
            self.task_idx_mapping = task_idx_mapping

        self.loss = make_loss

    def forward(self, **input):
        print(input.keys())
        # Step 1: Encode gene sequence
        gene_input = filter_args(self.gene_encoder.forward, input)
        print(gene_input.keys())
        exit()

        if self.freeze > 1:
            with torch.no_grad():
                encoder_outputs, _ = self.gene_encoder(**gene_input)  # [B, L, H]
        else:
            encoder_outputs, _ = self.gene_encoder(**gene_input)
        # Step 2: Q-Former attends to gene features
        q_output = self.qformer(encoder_outputs)  # [B, Q, H_qformer]
        q_output_proj = self.gene_proj(q_output)  # [B, Q, H_lm]
        q_output_proj = F.normalize(q_output_proj, dim=-1)
        print(q_output_proj.shape)
        print(input.keys())
        exit()

        batch_task_idx = task_idx.new_zeros(
            (len(input['task_idx']), self.instruction_token_input_ids.size(-1)))
        unique_task_idx = torch.unique(input['task_idx'])
        for i in range(len(unique_task_idx)):
            task_idx = unique_task_idx[i].item()
            mask_i = input['task_idx'] == task_idx
            task_idx = self.task_idx_mapping[task_idx]
            batch_task_idx[mask_i] = task_idx
        input_ids = self.instruction_token_input_ids[batch_task_idx]
        attention_mask = self.instruction_token_attention_mask[batch_task_idx]

        exit()


        # TODO: add instruction token as input for decoder, needs to check original implementation
        # Step 3: LLM (or just average)
        # decoder_input_ids = text_tokens.input_ids.clone()
        # decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        # labels = decoder_input_ids.masked_fill(
        #     decoder_input_ids == self.tokenizer.pad_token_id, -100
        # )

        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
        attention_mask = torch.cat([self.instruction_token_attention_mask, text_tokens.attention_mask], dim=1)
        lm_output = self.Qformer(
            decoder_input_ids,
            attention_mask=attention_mask,
            past_key_values=q_output_proj,  # TODO: this needs verification
            return_dict=True,
            labels=labels,
        )

        loss_lm = lm_output.loss
        return output


def base(cfg, gene_encoder, gene_tokenizer, qformer=None, llm=None, llm_tokenizer=None):
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
        # 定义 BertBase 这个 model,i.e. a computation graph
        model = BertBase(gene_encoder, gene_tokenizer, hidden_size, target_size, num_datasets, num_targets, task_names,
                         subset_names,  task_name,  subset_name, freeze, embedding_mode)
    else:
        # https://github.com/salesforce/LAVIS/blob/main/lavis/models/blip2_models/blip2_vicuna_instruct.py
        model = LLMBase(gene_encoder, gene_tokenizer, qformer, llm, llm_tokenizer, hidden_size, target_size,
                        num_datasets, num_targets, task_names, subset_names,
                        task_name, subset_name, freeze, embedding_mode)
    return model
