import math

import lightning.pytorch as pl
import torch
from torch import nn


class FixedPositionalEmbedding(pl.LightningModule):
    """
    Positional Embedding layer which applies a sinusoidal function on the token position
    pe(pos, 2i) = sin(pos / 10000^(2i / embedding_size)
    Reference BERT paper, page 6
    https://arxiv.org/pdf/1706.03762.pdf

    source: http://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding
    """

    def __init__(self, embedding_size: int, max_length: int = 512):
        super().__init__()

        # Initialize an empty matrix of
        positional_embedding = torch.empty((1, max_length, embedding_size), dtype=torch.float, device=self.device)
        # Disable gradient computation for positional embedding --> nothing to learn here
        positional_embedding.requires_grad = False

        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            (torch.arange(0, embedding_size, 2, dtype=torch.float) * - (math.log(10000.0) / embedding_size)))

        positional_embedding[:, :, 0::2] = torch.sin(position * div_term)
        positional_embedding[:, :, 1::2] = torch.cos(position * div_term)

        # register embedding layer as buffer
        # --> gets saved in statedict, without being trainable --> allows serialization of layer
        positional_embedding.unsqueeze(0)
        self.register_buffer("positional_embedding", positional_embedding)

    def forward(self, x):
        return self.positional_embedding[:, :x.size(1)]


class LearnedPositionalEmbedding(pl.LightningModule):
    """
    Learned PositionalEmbeddings
    """

    def __init__(self, embedding_size: int, max_length: int):
        super().__init__()
        self.embedding = nn.Embedding(max_length, embedding_size)

    def forward(self, x):
        # Programmatically get the device of the embedding layer to move the position idx to the same device as the
        # embedding table

        position_idx = torch.arange(0, x.size(1), dtype=torch.long, device=self.device).repeat((x.size(0), 1))

        return self.embedding(position_idx)
