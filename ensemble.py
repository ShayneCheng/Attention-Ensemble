from torch.utils.data import Dataset

import math
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.parameter import Parameter
import os
import time
import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings

warnings.filterwarnings("ignore")
import gc

gc.enable()
import math
import json
import time
import random
import multiprocessing

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from sklearn import model_selection
from string import punctuation

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torch.optim as optim
from torch.utils.data import (
    Dataset, DataLoader,
    SequentialSampler, RandomSampler
)
from torch.utils.data.distributed import DistributedSampler

try:
    from apex import amp

    APEX_INSTALLED = True
except ImportError:
    APEX_INSTALLED = False

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    logging,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
)


class EnsembleDataset(Dataset):
    def __init__(self, train_ensemble_features, mode='train'):
        super(EnsembleDataset, self).__init__()
        self.features = train_ensemble_features
        self.mode = mode

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        feature = self.features[item]
        if self.mode == 'train':
            return {
                'sequence_output': feature['sequence_output'],

                'attention_mask': torch.tensor(feature['attention_mask'], dtype=torch.long),
                'pooling_mask': feature['pooling_mask'],
                'question_bert_len': torch.tensor(feature['question_bert_len'], dtype=torch.long),
                'text_bert_len': torch.tensor(feature['text_bert_len'], dtype=torch.long),

                'start_position': torch.tensor(feature['start_position'], dtype=torch.long),
                'end_position': torch.tensor(feature['end_position'], dtype=torch.long),
                'example_id': feature["example_id"]
            }
        else:
            raise Exception("暂不支持其他mode！")
            # return {
            #     'sequence_output': feature['sequence_output'],
            #
            #     'attention_mask': torch.tensor(feature['attention_mask'], dtype=torch.long),
            #     'pooling_mask': feature['pooling_mask'],
            #     'question_bert_len': torch.tensor(feature['question_bert_len'], dtype=torch.long),
            #     'text_bert_len': torch.tensor(feature['text_bert_len'], dtype=torch.long),
            #
            #     # label 算loss
            #     'start_position': torch.tensor(feature['start_position'], dtype=torch.long),
            #     'end_position': torch.tensor(feature['end_position'], dtype=torch.long),
            #
            #     # text 算jaccard
            #
            #
            #
            # }


class EnsembleLinear(nn.Module):
    def __init__(self, *shape, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(EnsembleLinear, self).__init__()
        assert len(shape) == 3
        self.ensemble_num, self.in_features, self.out_features = shape
        self.weight = Parameter(torch.empty((self.ensemble_num, self.in_features, self.out_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty((self.ensemble_num, self.out_features), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):  # bs, ensemble_num, max_len, in_features
        assert len(input.shape) == 4
        bs, ensemble_num, max_len, in_features = input.shape
        if self.bias is not None:
            return torch.matmul(input, self.weight) + self.bias.unsqueeze(1).repeat(1, max_len, 1)
        return torch.matmul(input, self.weight)

    def extra_repr(self) -> str:
        return 'ensemble_num={}, in_features={}, out_features={}, bias={}'.format(
            self.ensemble_num, self.in_features, self.out_features, self.bias is not None
        )


class EnsembleModel(nn.Module):
    def __init__(self, config, ensemble_num=5, linear_bias=False):
        super(EnsembleModel, self).__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.ensemble_num = ensemble_num

        self.LinearQ = nn.Linear(self.hidden_size, self.hidden_size, bias=linear_bias)
        self.LinearK = EnsembleLinear(self.ensemble_num, self.hidden_size, self.hidden_size, bias=linear_bias)
        self.LinearV = EnsembleLinear(self.ensemble_num, self.hidden_size, self.hidden_size, bias=linear_bias)

        self.qa_outputs = nn.Linear(self.hidden_size, 2)

    #     self._init_weights(self.LinearQ)
    #     self._init_weights(self.LinearK)
    #     self._init_weights(self.qa_outputs)
    #
    # def _init_weights(self, module):
    #     if isinstance(module, nn.Linear):
    #         module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    #         if module.bias is not None:
    #             module.bias.data.zero_()

    def forward(
            self,
            sequence_output,  # batch_size, ensemble_num, max_len, emb_dim
            pooling_mask,
            question_bert_len=None,
            text_bert_len=None,
            attention_mask=None,
            if_activation_fun=True,
            if_res=True,
    ):
        batch_size, ensemble_num, max_len, emb_dim = sequence_output.shape
        assert ensemble_num == self.ensemble_num

        text = sequence_output  # batch_size, ensemble_num, max_len, emb_dim
        question = sequence_output.mean(1)  # 问题向量是5个ensemble的均值 batch_size, max_len, emb_dim

        Q = self.LinearQ(question)  # bs, max_len, emb_dim
        K = self.LinearK(text)  # bs, ensemble_num, max_len, emb_dim
        V = self.LinearV(text)  # bs, ensemble_num, max_len, emb_dim

        A = torch.matmul(K.transpose(0, 1) / (emb_dim ** 0.5), Q.transpose(1, 2))
        A = A * pooling_mask
        A = A.max(-1).values.sum(-1) / text_bert_len

        # 加个LN和非线性tanh？
        if if_activation_fun:
            A = torch.tanh(A)

        A = A.transpose(0, 1)  # batch_size, ensemble_num
        if if_res:  # 残差
            A = A + 1

        A_sum = A.sum(-1).unsqueeze(-1).unsqueeze(-1)  # 加权平均的分母
        sequence_output_new = torch.bmm(A.unsqueeze(1), V.view(batch_size, ensemble_num, max_len * emb_dim)).view(
            batch_size, max_len, emb_dim) / A_sum

        # sequence_output = self.dropout(sequence_output)
        qa_logits = self.qa_outputs(sequence_output_new)

        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1) * attention_mask
        end_logits = end_logits.squeeze(-1) * attention_mask

        return start_logits, end_logits


class Model(nn.Module):
    def __init__(self, modelname_or_path, config):
        super(Model, self).__init__()
        self.config = config
        self.xlm_roberta = AutoModel.from_pretrained(modelname_or_path, config=config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self._init_weights(self.qa_outputs)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
            self,
            input_ids,
            attention_mask=None,
    ):
        outputs = self.xlm_roberta(
            input_ids,
            attention_mask=attention_mask,
        )

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        # sequence_output = self.dropout(sequence_output)
        qa_logits = self.qa_outputs(sequence_output)

        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits, sequence_output


class DatasetRetriever(Dataset):
    def __init__(self, features, mode='train'):
        super(DatasetRetriever, self).__init__()
        self.features = features
        self.mode = mode

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        feature = self.features[item]
        if self.mode == 'train':
            return {
                'input_ids': torch.tensor(feature['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(feature['attention_mask'], dtype=torch.long),
                # 'offset_mapping': torch.tensor(feature['offset_mapping'], dtype=torch.long),
                # 'start_position': torch.tensor(feature['start_position'], dtype=torch.long),
                # 'end_position': torch.tensor(feature['end_position'], dtype=torch.long),
                # 'sequence_ids': torch.tensor(feature['sequence_ids'], dtype=torch.long),
                # 'id': feature['example_id'],
            }
        else:
            raise Exception("暂不支持别的mode！")
            # return {
            #     'input_ids': torch.tensor(feature['input_ids'], dtype=torch.long),
            #     'attention_mask': torch.tensor(feature['attention_mask'], dtype=torch.long),
            #     'offset_mapping': feature['offset_mapping'],
            #     'sequence_ids': feature['sequence_ids'],
            #     'id': feature['example_id'],
            #     'context': feature['context'],
            #     'question': feature['question']
            # }