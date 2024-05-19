from abc import abstractmethod

import torch

from .base import Morpher
from .helpers import choose_options
from ..nn import CPCLoss


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

    @abstractmethod
    def __call__(self, x):
        return x.map_dict(self.vocab, default=len(self.vocab))

    @abstractmethod
    def from_data(cls, x):
        raise NotImplementedError

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
        """The idea here is that we'll always expect logits in the last dimension,
        but torch cross entropy wants them in the second dimension, so we'll permute
        if there are more than 2 dimensions. If it's two dimensions then nothing gets
        moved."""

        def fixed_ce_loss(input, target):
            input = torch.transpose(input, 1, -1)
            return torch.nn.functional.cross_entropy(input, target, reduction="none")

        return fixed_ce_loss

    def generate(self, x, temperature=1.0, **_):
        options = choose_options(x, temperature=temperature)
        return options


# I didn't do this one, sorry.
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
