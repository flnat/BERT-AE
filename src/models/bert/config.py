import copy
import json


class LogBERTConfig:
    """
    Config Class for LogBERT Model initialization, inspired by https://github.com/google-research/bert/blob/master/modeling.py
    """

    def __init__(self,
                 max_length: int = None,
                 hidden_size: int = None,
                 n_layers: int = None,
                 attention_heads: int = None,
                 learned_positional: bool = True,
                 dropout: float = None,
                 uses_temporal: bool = None,
                 num_candidates: int = None,
                 lr: float = 1e-4,
                 betas: tuple[float, float] = (0.9, 0.999),
                 weight_decay: float = 1e-5,
                 use_hypersphere_loss: bool = True,
                 alpha: float = 0.1,
                 hs_nu: float = 0.25,
                 warmup_steps: int = 10_000,
                 prediction_method: str = "mlm",
                 beta_balanced_loss: float | None = None):
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.attention_heads = attention_heads
        self.learned_positional = learned_positional
        self.dropout = dropout
        self.uses_temporal = uses_temporal
        self.num_candidates = num_candidates
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.use_hypersphere_loss = use_hypersphere_loss
        self.alpha = alpha
        self.hs_nu = hs_nu
        self.warmup_steps = warmup_steps
        self.prediction_method = prediction_method
        self.beta_balanced_loss = beta_balanced_loss

    @classmethod
    def from_dict(cls, json_object: dict):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = LogBERTConfig()
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


class BERTConfig:
    """
    Config Class for BERT Model initialization, inspired by https://github.com/google-research/bert/blob/master/modeling.py
    """

    def __init__(self,
                 max_length: int = None,
                 hidden_size: int = None,
                 n_layers: int = None,
                 attention_heads: int = None,
                 learned_positional: bool = True,
                 dropout: float = None,
                 lr: float = 1e-4,
                 betas: tuple[float, float] = (0.9, 0.999),
                 weight_decay: float = 1e-5,
                 warmup_steps: int = 10_000):
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.attention_heads = attention_heads
        self.learned_positional = learned_positional
        self.dropout = dropout
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps

    @classmethod
    def from_dict(cls, json_object: dict):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = LogBERTConfig()
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
