from typing import List
from abc import abstractmethod

import numpy as np
import torch

from ..base.base import Morpher
from ..base.helpers import choose_options
from ..nn import Unsqueezer, RankScaleTransform, Time2Vec
from ..backends.polars import PolarsNormalizerBackend, PolarsQuantilerBackend


class Normalizer(Morpher):

    MISSING_VALUE = 0.0

    BACKEND_LOOKUP = {
        "polars": PolarsNormalizerBackend,
    }

    def __init__(self, mean, std, backend="polars"):
        self.mean = mean
        self.std = std
        self.backend = self.get_backend(backend)

    def __call__(self, x):
        return self.backend(x, self.mean, self.std)

    def fill_missing(self, x):
        return self.backend.fill_missing(x, self.MISSING_VALUE)

    @property
    def required_dtype(self):
        return torch.float32

    def denormalize(self, x):
        # TKTK: This should be a generic unmorph method
        # reverse operation
        return x * self.std + self.mean

    def save_state_dict(self):
        return {
            "mean": self.mean,
            "std": self.std,
        }

    @classmethod
    def from_state_dict(cls, state_dict, backend="polars"):
        return cls(**state_dict, backend=backend)

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


class Quantiler(Morpher):

    MISSING_VALUE = 0.5

    BACKEND_LOOKUP = {
        "polars": PolarsQuantilerBackend,
    }

    def __init__(self, quantiles, backend="polars"):
        self.quantiles = quantiles
        self.n_quantiles = len(quantiles)
        self.backend = self.get_backend(backend)

    def __call__(self, x):
        return self.backend(x, self.quantiles)

    def fill_missing(self, x):
        return self.backend.fill_missing(x, self.MISSING_VALUE)

    @property
    def required_dtype(self):
        return torch.float32

    def save_state_dict(self):
        return {"quantiles": self.quantiles}

    @classmethod
    def from_state_dict(cls, state_dict, backend="polars"):
        return cls(**state_dict, backend=backend)

    def __repr__(self):
        return f"Quantiler(<{self.n_quantiles} quantiles>)"

    def make_embedding(self, x, /):
        return torch.nn.Sequential(
            Unsqueezer(dim=-1),
            torch.nn.Linear(in_features=1, out_features=x),
        )

    def make_predictor_head(self, x, /):
        return torch.nn.Linear(in_features=x, out_features=self.n_quantiles)

    def make_criterion(self):
        # Each bucket means exactly the quantile value, so there's some
        # quantization error.
        def quantile_bce(input, target):
            input = torch.transpose(input, 1, -1)
            target = torch.round(target * self.n_quantiles).long()
            return torch.nn.functional.cross_entropy(input, target, reduction="none")

        return quantile_bce

    def generate(self, x, temperature=1.0, **_):
        options = choose_options(x, temperature=temperature)
        return options / self.n_quantiles


class NullNormalizer(Normalizer):

    @classmethod
    def from_data(cls, x):
        mean = x.drop_nulls().mean()
        std = x.drop_nulls().std()

        return cls(mean, std)


class RankScaler(Morpher):
    """I don't know what to call this one. It's from here:
    https://www.amazon.science/publications/an-inductive-bias-for-tabular-deep-learning
    """

    def __init__(self, mean, std, quantiles):
        self.mean = mean
        self.std = std
        self.quantiles = quantiles
        self.n_quantiles = len(quantiles)
        self.q_array = np.array(self.quantiles)

    @abstractmethod
    def __call__(self, x):
        raise NotImplementedError

    @property
    def required_dtype(self):
        return torch.float32

    @property
    def missing_value(self):
        return 0.0

    @classmethod
    def from_data(cls, x, n_quantiles):
        mean = x.mean()
        std = x.std()
        quantiles = np.nanquantile(
            (x - mean) / std, np.linspace(0, 1, n_quantiles)
        ).tolist()

        return cls(mean, std, quantiles)

    @classmethod
    def from_state_dict(cls, state_dict):
        return cls(**state_dict)

    def save_state_dict(self):
        return {"mean": self.mean, "std": self.std, "quantiles": self.quantiles}

    def make_embedding(self, x, /):
        return torch.nn.Sequential(
            RankScaleTransform(),
            Unsqueezer(dim=-1),
            torch.nn.Linear(in_features=1, out_features=x),
        )

    def make_predictor_head(self, x, /):
        return torch.nn.Linear(in_features=x, out_features=1)

    def make_criterion(self):
        return torch.nn.MSELoss(reduction="none")


class TimeNormalizer(Normalizer):
    def make_embedding(self, x, /):
        return torch.nn.Sequential(
            Unsqueezer(dim=-1),
            Time2Vec(x)
        )
    
    def __repr__(self):
        return f"TimeNormalizer(mean={self.mean}, std={self.std})"
