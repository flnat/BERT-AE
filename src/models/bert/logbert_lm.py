from dataclasses import dataclass

import lightning.pytorch as pl
import torch
from torch import nn

from .bert import BERT


@dataclass
class LogBERTTuple:
    key_prediction: torch.Tensor
    cls_representation: torch.Tensor


class LogBERT(pl.LightningModule):
    """
    LogBERT Implementation
    10.1109/IJCNN52387.2021.9534113
    """

    def __init__(self, bert: BERT):
        """

        :param bert: BERT model
        :param vocab_size: total vocab size
        """

        super().__init__()
        self.bert = bert
        self.masked_lm = MaskedLogModel(self.bert.embedding_size, self.bert.vocab_size)

    def forward(self, x, time_info=None) -> LogBERTTuple:
        x = self.bert(x, time_info=time_info)

        # logkey_out is the result of MLM Task, i.e. the logits of the vocabulary
        # self.result["logkey_out"] = self.masked_lm(x)
        # cls_out is the learned representation of the <CLS> Token, which is used for the VHM Task
        # self.result["cls_out"] = x[:, 0]

        return LogBERTTuple(key_prediction=self.masked_lm(x), cls_representation=x[:, 0])


class MaskedLogModel(pl.LightningModule):
    """
    Project Transformer Encoder Embeddings back into original vocabulary space
    used for masked language modelling in BERT
    """

    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, vocab_size)
        # CrossEntropyLoss is a bit more efficient compared to NLLLoss, therefore logits are enough
        # self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.linear(x)


#################
"""
Modules below are not used in LogBERT Implementation, but were present in the original repo
"""


#################

class TimeLogModel(nn.Module):
    def __init__(self, hidde_size: int, time_size=1):
        super().__init__()
        self.linear = nn.Linear(hidde_size, time_size)

    def forward(self, x):
        return self.linear(x)


class LogClassifier(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, cls):
        return self.linear(cls)


class LinearCLS(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        return self.linear(x)
