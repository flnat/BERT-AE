from torch import nn


class TimeEmbedding(nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()
        self.time_embedding = nn.Linear(1, embedding_size)

    def forward(self, time_interval):
        return self.time_embedding(time_interval)
