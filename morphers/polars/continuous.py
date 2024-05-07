from typing import List

import numpy as np
import polars as pl
import torch

from ..base.continuous import Normalizer, RankScaler, Quantiler


class PolarsNormalizer(Normalizer):
    def __call__(self, x):
        x = (x - self.mean) / self.std
        return x.fill_nan(0).fill_null(0)


class PolarsRankScaler(RankScaler):

    def __init__(self, mean, std, quantiles):
        self.mean = mean
        self.std = std
        self.quantiles = quantiles
        self.n_quantiles = len(quantiles)
        self.q_array = np.array(self.quantiles)

    def __call__(self, x):
        x = (x - self.mean) / self.std
        q = pl.Series(self.q_array[1:])
        # Ultra-defensive
        x = x.fill_nan(self.missing_value).fill_null(self.missing_value)
        return pl.concat_list(
            x,
            x.cut(q, labels=np.arange(self.n_quantiles).astype("str")).cast(pl.Float32)
            / self.n_quantiles,
        )


class PolarsQuantiler(Quantiler):

    def __call__(self, x):
        q = pl.Series(self.quantiles[1:])
        # k means between the (k-1)th quantile and the kth quantile
        return (
            x.cut(q, labels=np.arange(self.n_quantiles).astype("str")).cast(pl.Float32)
            / self.n_quantiles
        )
