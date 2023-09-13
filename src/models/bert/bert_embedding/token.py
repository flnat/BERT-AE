"""
Token Embedding Layer
"""

import lightning.pytorch as pl
from torch import nn


class TokenEmbedding(pl.LightningModule):
    """
    Simple Wrapper around torch Embedding Layer, mainly for coherent api
    """

    def __init__(self, vocab_size: int, embedding_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)

    def forward(self, x):
        return self.embedding(x)
