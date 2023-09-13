from __future__ import annotations

import gc
import logging
import multiprocessing
import os
import random
import tempfile
from collections import Counter, OrderedDict
from copy import copy
from dataclasses import dataclass
from typing import Optional, Any, Iterable, List

import joblib
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import Vocab, vocab

from src.utils.model_selection import random_split
from src.utils.py_utils import non_blank_lines
from .data_config import BERTDataModuleConfig


class FileIterator:
    """
    Helper Class for building torchtext vocabs
    """

    def __init__(self, file_path, encoding):
        self.file_path = file_path
        self.encoding = encoding

    def __iter__(self):
        self.file_handle = open(self.file_path, "rt", encoding=self.encoding)
        return self

    def __next__(self):
        try:
            line = self.file_handle.readline()
            if line == "" or line == "\n":
                # Close file handle if EOF
                self.file_handle.close()
                # Raise StopIteration Exception for __iter__ protocol
                raise StopIteration
            return line.strip().split()
        except:
            # Close the file handle on any exception to prevent accidental file corruption. Corruption wouldn't be
            # all too bad because the file is created at runtime and is not persisted after cluster shutdown but
            # still better safe than sorry
            self.file_handle.close()
            raise StopIteration


class BERTVocab:
    """
    Wrapper around TorchText Vocabulary with some additional helper methods
    """

    def __init__(self, corpus_path: str, min_frequency: int, max_tokens: int,
                 corpus_encoding: str = "utf-8"):
        file_iterator = FileIterator(corpus_path, corpus_encoding)
        self.special_tokens = ["<PAD>", "<UNK>", "<EOS>", "<SOS>", "<MASK>"]
        self.torch_vocab, self._token_counts = self._build_vocab_from_iterator(file_iterator, min_freq=min_frequency,
                                                                               max_tokens=max_tokens,
                                                                               specials=self.special_tokens,
                                                                               special_first=True)
        self.torch_vocab.set_default_index(self.torch_vocab["<UNK>"])

    def __len__(self) -> int:
        return len(self.torch_vocab)

    def __call__(self, x):
        return self.torch_vocab(x)

    def __getitem__(self, item):
        return self.torch_vocab[item]

    def itos(self) -> list[str]:
        """
        Index to Token Mapping
        :return: List of Tokens (List Index is mapping index)
        """
        return self.torch_vocab.get_itos()

    def stoi(self) -> dict[str, int]:
        """
        Token to Index Mapping
        :return: Dict (Token -> Index)
        """
        return self.torch_vocab.get_stoi()

    def save(self, path) -> None:
        joblib.dump(self, os.path.join(path, "bert_vocab.pkl"))

    def vocab_size(self):
        return self.torch_vocab.__len__

    def token_counts(self):
        return self._token_counts

    @classmethod
    def from_dump(cls, path) -> BERTVocab:
        return joblib.load(path)

    @staticmethod
    def _build_vocab_from_iterator(
            iterator: Iterable,
            min_freq: int = 1,
            specials: Optional[List[str]] = None,
            special_first: bool = True,
            max_tokens: Optional[int] = None,
    ) -> tuple[Vocab, Iterable]:

        counter = Counter()
        for tokens in iterator:
            counter.update(tokens)

        specials = specials or []

        # First sort by descending frequency, then lexicographically
        sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

        if max_tokens is None:
            ordered_dict = OrderedDict(sorted_by_freq_tuples)
        else:
            assert len(
                specials) < max_tokens, "len(specials) >= max_tokens, so the vocab will be entirely special tokens."
            ordered_dict = OrderedDict(sorted_by_freq_tuples[: max_tokens - len(specials)])

        word_vocab = vocab(ordered_dict, min_freq=min_freq, specials=specials, special_first=special_first)
        token_counts = list(ordered_dict.values())
        return word_vocab, token_counts


@dataclass
class BERTInput:
    bert_tokens: torch.Tensor
    bert_labels: torch.Tensor
    sequence_labels: Optional[torch.Tensor]


class MaskedLogDataset(Dataset):
    def __init__(self, logs: list, vocab: BERTVocab, max_seq_len: int, semi_supervised: bool = True,
                 labels: Optional[list] = None, predict_mode: bool = True, mask_ratio: float = 0.15):

        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.predict_mode = predict_mode
        self.mask_ratio = mask_ratio
        self.semi_supervised = semi_supervised
        self.logs = logs
        self.labels = labels
        self.corpus_length = len(logs)

    def __len__(self):
        return self.corpus_length

    def __getitem__(self, item):
        masked_seq, labeled_seq = self.mask_input(self.logs[item])

        # Add Start-of-Sequence Token to Sequence, paper specifies <DIST> Token, but in implementation <SOS> is used,
        # doesn't really matter what kind of token is used, as long it is only used here
        masked_seq = [self.vocab["<SOS>"]] + masked_seq

        # ADD <PAD> at start of masked_labels to account for additional <DIST> Token
        labeled_seq = [self.vocab["<PAD>"]] + labeled_seq

        if self.semi_supervised:
            # If running on semi_supervised mode for validation/testing also return labels
            labels = self.labels[item]
            return masked_seq, labeled_seq, labels

        return masked_seq, labeled_seq

    def mask_input(self, sequence):

        MASK_INDEX = self.vocab["<MASK>"]

        tokens = copy(sequence)
        masked_labels = []

        for idx, token in enumerate(sequence):
            mask_probability = random.random()

            # Replace [mask_ratio] Tokens with the [MASK] Token
            if mask_probability <= self.mask_ratio:
                # 80 % of elected tokens are replaced with [MASK]
                # 10 % are replaced with random token from the vocabulary
                # 10 % remain unchanged
                if self.predict_mode:
                    # For inference default behaviour is sufficient
                    tokens[idx] = MASK_INDEX
                    masked_labels.append(self.vocab[token])
                    continue
                    # masked_labels.append(INDICES.get(token, self.vocab["<UNK>"]))
                mask_probability /= self.mask_ratio
                if mask_probability < 0.8:
                    tokens[idx] = MASK_INDEX
                elif mask_probability < 0.9:
                    tokens[idx] = random.randrange(len(self.vocab))
                else:
                    tokens[idx] = self.vocab[token]
                masked_labels.append(self.vocab[token])
            else:
                tokens[idx] = self.vocab[token]
                masked_labels.append(0)

        return tokens, masked_labels

    def collate_fn(self, batch, percentile=100, dynamic_pad=True) -> BERTInput:
        input_lengths = [len(seq[0]) for seq in batch]

        if dynamic_pad:
            seq_len = int(np.percentile(input_lengths, percentile))
            if seq_len is not None:
                seq_len = min(seq_len, self.max_seq_len)
        else:
            seq_len = self.max_seq_len

        bert_inputs = []
        bert_labels = []

        if self.semi_supervised:
            sequence_labels = []

        else:
            sequence_labels = None

        for seq in batch:
            bert_input = seq[0][:seq_len]
            bert_label = seq[1][:seq_len]

            padding = [self.vocab["<PAD>"] for _ in range(seq_len - len(bert_input))]
            bert_input.extend(padding)
            bert_label.extend(padding)

            bert_inputs.append(bert_input)
            bert_labels.append(bert_label)

            if self.semi_supervised:
                sequence_labels.append(int(seq[2]))

        if self.semi_supervised:
            sequence_labels = torch.LongTensor(sequence_labels)

        return BERTInput(bert_tokens=torch.LongTensor(bert_inputs), bert_labels=torch.LongTensor(bert_labels),
                         sequence_labels=sequence_labels)

    @classmethod
    def from_file(cls, log_corpus: str, vocab: BERTVocab, max_seq_len: int,
                  semi_supervised: bool = True, label_file: Optional[str] = None,
                  predict_mode: bool = False, mask_ratio: float = 0.15):

        with open(log_corpus, "rt") as f:
            logs = [line.strip().split() for line in non_blank_lines(f)]

        if semi_supervised:
            with open(label_file, "rt") as f:
                labels = [line.strip() for line in non_blank_lines(f)]
        else:
            labels = None

        return cls(logs, vocab=vocab, max_seq_len=max_seq_len, semi_supervised=semi_supervised,
                   labels=labels, predict_mode=predict_mode, mask_ratio=mask_ratio)


class BERTDataModule(pl.LightningDataModule):
    def __init__(self, parsed_logs: str, feature_column: str,
                 labels: str | None, label_column: str | None, join_key: str | None,
                 batch_size: int, splits: list[float], vocab_min_freq: int,
                 vocab_max_size: int, max_seq_length: int, mask_ratio: float,
                 rng_seed: int, num_workers: Optional[int] = None, semi_supervised: bool = True,
                 split_mode: str = "uncontaminated"):
        super().__init__()
        if num_workers is None:
            self.num_workers = multiprocessing.cpu_count()
        else:
            self.num_workers = num_workers

        # Data Fetching & Preparation Params

        self.features = pd.read_csv(parsed_logs)

        if semi_supervised:
            self.labels = pd.read_csv(labels)

        self.cache_dir = tempfile.mkdtemp()
        self.feature_column = feature_column

        self.split_mode = split_mode
        self.label_column = label_column
        self.join_key = join_key
        self.batch_size = batch_size
        self.splits = splits
        self.rng_seed = rng_seed
        self.semi_supervised = semi_supervised
        # Vocabulary Params
        self.vocab_min_freq = vocab_min_freq
        self.vocab_max_size = vocab_max_size

        # Preprocessing & Masking Params
        self.max_seq_length = max_seq_length
        self.mask_ratio = mask_ratio

        # Data objects
        self.vocab: Optional[BERTVocab] = None
        self.train_ds: Optional[MaskedLogDataset] = None
        self.val_ds: Optional[MaskedLogDataset] = None
        self.test_ds: Optional[MaskedLogDataset] = None
        self.predict_ds: Optional[MaskedLogDataset] = None

        # Helper attributes
        self.already_prepared = False
        self.vocab_size = None

    def prepare_data(self) -> None:
        if self.already_prepared:
            # Due to the fact that we need the vocab size to build the model, we will need to manually call prepare_data
            # before we are able to build the model
            # Because we will already have prepared our data, we will then no longer need to call prepare_data as part
            # lightning automatic setup
            return

        if self.semi_supervised:
            training, validation, testing = random_split(parsed_data=self.features, labels=self.labels,
                                                         join_key=self.join_key, label_encoding=(0, 1),
                                                         sizes=self.splits, rng_seed=self.rng_seed,
                                                         mode=self.split_mode)

        else:
            training, validation, testing = random_split(parsed_data=self.features, sizes=self.splits,
                                                         rng_seed=self.rng_seed)
        # Collect Dataset from spark, write it to file and then garbage collect it
        training = [row[self.feature_column] for _, row in training.iterrows()]
        with open(os.path.join(self.cache_dir, "training"), "wt") as f:
            f.writelines("\n".join(training))
        del training  # Free Memory
        gc.collect()

        # Serialize training data
        with open(os.path.join(self.cache_dir, "validation"), "wt") as f:
            f.writelines("\n".join([row[self.feature_column] for _, row in validation.iterrows()]))
        # If there are labels supplied, also serialize them
        if self.semi_supervised:
            with open(os.path.join(self.cache_dir, "validation_labels"), "wt") as f:
                f.writelines("\n".join([str(row[self.label_column]) for _, row in validation.iterrows()]))
        del validation
        gc.collect()

        # Serialize testing data
        with open(os.path.join(self.cache_dir, "testing"), "wt") as f:
            f.writelines("\n".join([row[self.feature_column] for _, row in testing.iterrows()]))

        if self.semi_supervised:
            with open(os.path.join(self.cache_dir, "testing_labels"), "wt") as f:
                f.writelines("\n".join([str(row[self.label_column]) for _, row in testing.iterrows()]))
        del testing
        gc.collect()

        # Creation of Vocab
        vocab = BERTVocab(corpus_path=os.path.join(self.cache_dir, "training"),
                          min_frequency=self.vocab_min_freq, max_tokens=self.vocab_max_size,
                          )

        joblib.dump(vocab, os.path.join(self.cache_dir, "vocab"))

        self.already_prepared = True
        self.vocab_size = len(vocab)

    def setup(self, stage: str) -> None:
        # TrainSet Creation

        train_path = os.path.join(self.cache_dir, "training")
        val_path = os.path.join(self.cache_dir, "validation")
        test_path = os.path.join(self.cache_dir, "testing")
        vocab_path = os.path.join(self.cache_dir, "vocab")

        val_labels_path = os.path.join(self.cache_dir, "validation_labels")
        test_labels_path = os.path.join(self.cache_dir, "testing_labels")
        if self.vocab is None:
            self.vocab = joblib.load(vocab_path)

        if stage == "fit" or stage is None:
            self.train_ds = MaskedLogDataset.from_file(log_corpus=train_path, vocab=self.vocab,
                                                       max_seq_len=self.max_seq_length, predict_mode=True,
                                                       semi_supervised=False, mask_ratio=self.mask_ratio)

            self.val_ds = MaskedLogDataset.from_file(log_corpus=val_path, vocab=self.vocab,
                                                     max_seq_len=self.max_seq_length, label_file=val_labels_path,
                                                     semi_supervised=self.semi_supervised, mask_ratio=self.mask_ratio,
                                                     predict_mode=True)

        if stage == "validate" or stage is None:
            self.val_ds = MaskedLogDataset.from_file(log_corpus=val_path, vocab=self.vocab,
                                                     max_seq_len=self.max_seq_length,
                                                     semi_supervised=self.semi_supervised, label_file=val_labels_path,
                                                     predict_mode=True, mask_ratio=self.mask_ratio)

        if stage == "test" or stage is None:
            self.test_ds = MaskedLogDataset.from_file(log_corpus=test_path, vocab=self.vocab,
                                                      max_seq_len=self.max_seq_length,
                                                      semi_supervised=self.semi_supervised, label_file=test_labels_path,
                                                      predict_mode=True, mask_ratio=self.mask_ratio)
        if stage == "predict" or stage is None:
            self.predict_ds = MaskedLogDataset.from_file(log_corpus=test_path, vocab=self.vocab,
                                                         max_seq_len=self.max_seq_length,
                                                         semi_supervised=False, predict_mode=True,
                                                         mask_ratio=self.mask_ratio)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                          collate_fn=self.train_ds.collate_fn)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          collate_fn=self.val_ds.collate_fn)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          collate_fn=self.test_ds.collate_fn)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.predict_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                          collate_fn=self.predict_ds.collate_fn)

    def save_vocab(self, save_path: str) -> None:
        if self.vocab is not None:
            joblib.dump(self.vocab, save_path)
        else:
            logging.error("Vocabulary has not yet been built! Call prepare_data and setup first.")

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        if isinstance(batch, BERTInput):
            batch.bert_tokens = batch.bert_tokens.to(device)
            batch.bert_labels = batch.bert_labels.to(device)
            if batch.sequence_labels is not None:
                batch.sequence_labels = batch.sequence_labels.to(device)
        else:
            batch = super().transfer_batch_to_device(batch, device, dataloader_idx)

        return batch

