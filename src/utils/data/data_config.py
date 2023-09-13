import copy
import json
import multiprocessing
from typing import Optional


class BERTDataModuleConfig:
    def __init__(self, feature_table: str, feature_column: str, db_schema: str, semi_supervised: bool = False,
                 label_table: Optional[str] = None, label_column: Optional[str] = None, join_key: Optional[str] = None,
                 batch_size: int = 1, splits: list[float] = (.01, .2, .8), vocab_min_freq: int = 1,
                 vocab_max_size: int = 10_000, max_seq_length: int = 512, mask_ratio: float = .15,
                 rng_seed: int = 42, num_workers: Optional[int] = None, split_mode: Optional[str] = None
                 ):

        self.semi_supervised = semi_supervised
        self.feature_table = feature_table
        self.label_table = label_table
        self.db_schema = db_schema
        self.feature_column = feature_column
        self.label_column = label_column
        self.join_key = join_key
        self.batch_size = batch_size
        self.splits = splits
        self.vocab_min_freq = vocab_min_freq
        self.vocab_max_size = vocab_max_size
        self.max_seq_length = max_seq_length
        self.mask_ratio = mask_ratio
        self.rng_seed = rng_seed
        self.split_mode = split_mode

        if num_workers is None:
            self.num_workers = multiprocessing.cpu_count()
        else:
            self.num_workers = num_workers

    @classmethod
    def from_dict(cls, json_object: dict):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""

        config = BERTDataModuleConfig(**json_object)
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "rt") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
