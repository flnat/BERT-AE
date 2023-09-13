from __future__ import annotations

import torch.nn as nn

from .bert_embedding import BERTEmbedding


class BERT(nn.Module):
    def __init__(self, vocab_size: int, max_length: int, hidden_size: int, embedding_size: int, n_layers: int,
                 attention_heads: int,
                 dropout: float, uses_temporal: bool, learned_positional: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.attention_heads = attention_heads
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embedding_size=embedding_size,
                                       max_length=max_length, uses_temporal=uses_temporal,
                                       dropout=dropout, learned_positional=learned_positional)

        transformer_encoder = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=self.attention_heads,
                                                         dim_feedforward=self.hidden_size, dropout=self.dropout,
                                                         activation="gelu", batch_first=True, norm_first=True)

        self.transformer = nn.TransformerEncoder(transformer_encoder, num_layers=self.n_layers)

    def forward(self, x, segment_label=None, time_info=None):
        # Attention masking for padded tokens

        # The original paper use a self-rolled Transformer Implementation, whereby the padding mask
        # is expected to be 4-D Tensor of Shape [batch, 1, seq_len, seq_len]
        # not really sure on the reason for this, but as we use the torch implementation of a transformer, we will use
        # a simple mask
        # mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        mask = x == 0

        # Embedding indexed sequence
        x = self.embedding(x, segment_label, time_info)

        # Because we do not want to attend over <PAD> Tokens we will use also pass mask over all
        x = self.transformer(x, src_key_padding_mask=mask)

        return x

    @classmethod
    def from_config(cls, config: dict) -> BERT:
        return BERT(vocab_size=config["vocab_size"], max_length=config["max_length"], hidden_size=config["hidden_size"],
                    embedding_size=config["embedding_size"], n_layers=config["n_layers"],
                    attention_heads=config["attention_heads"],
                    dropout=config["dropout"], uses_temporal=config["uses_temporal"],
                    learned_positional=config["learned_positional"])
