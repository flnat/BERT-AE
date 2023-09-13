import copy
import json


class AutoEncoderConfig:
    """
    Config Class for LogBERT Model initialization, inspired by https://github.com/google-research/bert/blob/master/modeling.py
    """

    def __init__(self,
                 input_size: int = 256,
                 hidden_size: int = 128,
                 code_size: int = 64,
                 dropout: float = 0.1,
                 contamination: float = 1e-2,
                 cell: str = "lstm",
                 recon_norm: str = "l2",
                 lr: float = 1e-3,
                 betas: list[float, float] = [0.9, 0.999],
                 weight_decay: float = 1e-3,
                 semi_supervised: bool = True,
                 plot_histogramm: bool = True):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.code_size = code_size
        self.dropout = dropout
        self.contamination = contamination
        self.cell = cell
        self.recon_norm = recon_norm
        self.lr = lr
        self.weight_decay = weight_decay
        self.semi_supervised = semi_supervised
        self.plot_histogramm = plot_histogramm
        self.betas = betas
    @classmethod
    def from_dict(cls, json_object: dict):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = AutoEncoderConfig()
        for key, value in json_object.items():
            config.__dict__[key] = value
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
