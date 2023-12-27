from abc import ABC, abstractmethod

import numpy as np
import torch


from .nn import Unsqueezer, CPCLoss


class Morpher(ABC):
    @abstractmethod
    def __call__(self):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_data(self):
        raise NotImplementedError

    @abstractmethod
    def make_embedding(self):
        raise NotImplementedError

    @abstractmethod
    def make_predictor_head(self):
        raise NotImplementedError

    @abstractmethod
    def make_criterion(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def required_dtype(self):
        raise NotImplementedError

    @abstractmethod
    def save_state_dict(self):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_state_dict(cls, state_dict):
        raise NotImplementedError


class Normalizer(Morpher):
    def __init__(self, mean, std):
        if isinstance(mean, (np.ndarray, np.generic, torch.Tensor)):
            self.mean = mean.item()
            self.std = std.item()

        else:
            self.mean = mean
            self.std = std

    @property
    def required_dtype(self):
        return torch.float32

    @property
    def missing_value(self):
        return self.mean

    def normalize(self, x):
        x = (x - self.mean) / self.std
        return x.fill_nan(0).fill_null(0)

    def denormalize(self, x):
        # reverse operation
        return x * self.std + self.mean

    def __call__(self, x):
        return self.normalize(x)

    @classmethod
    def from_data(cls, x):
        mean = x.mean()
        std = x.std()

        return cls(mean, std)

    def save_state_dict(self):
        return {
            "mean": self.mean,
            "std": self.std,
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        return cls(**state_dict)

    def __repr__(self):
        return f"Normalizer(mean={self.mean}, std={self.std})"

    def make_embedding(self, x, /):
        return torch.nn.Sequential(
            Unsqueezer(dim=-1),
            torch.nn.Linear(in_features=1, out_features=x),
        )

    def make_predictor_head(self, x, /):
        return torch.nn.Linear(in_features=x, out_features=1)

    def make_criterion(self):
        return torch.nn.MSELoss(reduction="none")


class Integerizer(Morpher):
    MISSING_VALUE = "<MISSING>"

    def __init__(self, vocab):
        self.vocab = vocab

    @property
    def required_dtype(self):
        return torch.int64

    @property
    def missing_value(self):
        return self.MISSING_VALUE

    def __call__(self, x):
        return x.map_dict(self.vocab, default=len(self.vocab))

    @classmethod
    def from_data(cls, x):
        vocab = {t: i for i, t in enumerate(x.filter(x.is_not_null()).unique())}

        return cls(vocab)

    def save_state_dict(self):
        return {"vocab": self.vocab}

    @classmethod
    def from_state_dict(cls, state_dict):
        return cls(**state_dict)

    def __repr__(self):
        return f"Integerizer(<{len(self.vocab)} items>)"

    def make_embedding(self, x, /):
        return torch.nn.Embedding(len(self.vocab) + 1, x)

    def make_predictor_head(self, x, /):
        return torch.nn.Linear(in_features=x, out_features=len(self.vocab) + 1)

    def make_criterion(self):
        def fixed_ce_loss(input, target):
            return torch.nn.functional.cross_entropy(
                input.permute(0, 2, 1), target, reduction="none"
            )

        return fixed_ce_loss


class BigIntegerizer(Integerizer):
    N_NEGATIVE_SAMPLES = 15

    def make_embedding(self, x, /):
        self.embedding_layer = torch.nn.Embedding(len(self.vocab) + 1, x)
        return self.embedding_layer

    def make_predictor_head(self, x, /):
        return torch.nn.Linear(in_features=x, out_features=x)

    def make_criterion(self):
        if not hasattr(self, "embedding_layer"):
            raise RuntimeError("make_embedding must be called before make_criterion.")
        return CPCLoss(self.embedding_layer, self.N_NEGATIVE_SAMPLES)
