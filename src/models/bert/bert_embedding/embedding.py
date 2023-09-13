import lightning.pytorch as pl
from torch import nn

from .position import FixedPositionalEmbedding, LearnedPositionalEmbedding
from .segment import SegmentEmbedding
from .time import TimeEmbedding
from .token import TokenEmbedding


class BERTEmbedding(pl.LightningModule):
    """
    BERT Embeddings adjusted for use in LogBERT Model, consisting of the sum of:\n
    1. TokenEmbedding: classic embedding for sequence tokens\n
    2. PositionalEmbedding: positional information using sinusoidal & cosinusoidal functions\n
    3. SegmentEmbedding: sequence segment information\n
    4. TimeEmbedding: capture temporal distance between tokens (optional)
    """

    def __init__(self, vocab_size: int, embedding_size: int, max_length: int, dropout: float,
                 uses_temporal: bool = False, learned_positional: bool = True):
        """
        :param vocab_size: total size of training vocabulary 
        :param embedding_size: size of token embeddings
        :param max_length: maximal length of input sequence
        :param dropout: dropout rate
        :param log_key: 
        :param uses_temporal: whether to apply temporal embedding layer
        """
        super().__init__()
        # Layer definition
        self.token = TokenEmbedding(vocab_size=vocab_size, embedding_size=embedding_size)

        if learned_positional:
            self.position = LearnedPositionalEmbedding(embedding_size=embedding_size, max_length=max_length)
        else:
            self.position = FixedPositionalEmbedding(embedding_size=embedding_size, max_length=max_length)

        self.segment = SegmentEmbedding(embedding_size=embedding_size)
        self.time = TimeEmbedding(embedding_size=embedding_size)
        self.dropout = nn.Dropout(p=dropout)
        # hyperparams
        self.embedding_size = embedding_size
        self.uses_temporal = uses_temporal

    def forward(self, sequence, segment_label=None, time_info=None):

        x = self.position(sequence)

        x = x + self.token(sequence)

        if segment_label is not None:
            x += self.segment(segment_label)
        if self.uses_temporal:
            x += self.time(time_info)

        return self.dropout(x)
