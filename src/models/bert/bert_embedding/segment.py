from torch import nn


class SegmentEmbedding(nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(3, embedding_size, padding_idx=0)

    def forward(self, x):
        return self.embedding(x)
