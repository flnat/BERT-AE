from typing import Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class UnmaskedInput(Dataset):
    def __init__(self, logs: list[list[str]], vocab, labels: Optional[list[int]] = None, max_seq_length: int = 512):
        self.logs = logs
        self.labels = labels
        # Assumption --> Transformer with batch first upstream
        self.length = len(logs)
        self.vocab = vocab
        self.max_seq_length = max_seq_length

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        tokens = []
        for event in self.logs[item]:
            tokens.append(self.vocab[event])

        if self.labels is not None:
            return tokens, self.labels[item]
        else:
            return tokens

    def collate(self, batch):

        if self.labels is not None:
            labels = [x[1] for x in batch]
            features = [x[0] for x in batch]

            features = [torch.LongTensor(seq) for seq in features]
            labels = torch.LongTensor([int(label) for label in labels])

            padded_sequences = pad_sequence(features, batch_first=True)
            if padded_sequences.size(1) > self.max_seq_length:
                padded_sequences = padded_sequences[:, :self.max_seq_length]

            return padded_sequences, labels
        else:
            batch = [torch.LongTensor(seq) for seq in batch]
            padded_sequences = pad_sequence(batch, batch_first=True)

            if padded_sequences.size(1) > self.max_seq_length:
                padded_sequences = padded_sequences[:, :self.max_seq_length]

            return padded_sequences
