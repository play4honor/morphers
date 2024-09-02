from typing import List

import numpy as np
import polars as pl

from ...base.base import MorpherBackend


class PolarsNormalizerBackend(MorpherBackend):

    def __call__(self, x, mean, std):
        x = (x - mean) / std
        return x.fill_nan(0).fill_null(0)

    def fill_missing(self, x, missing):
        return x.fill_nan(missing).fill_null(missing)

    @staticmethod
    def from_data(x) -> dict:
        mean = x.mean()
        std = x.std()

        return {"mean": mean, "std": std}


class PolarsQuantilerBackend(MorpherBackend):

    def __call__(self, x, quantiles):
        q = pl.Series(quantiles[1:])
        return x.cut(q, labels=np.arange(len(quantiles)).astype("str")).cast(
            pl.Float32
        ) / len(quantiles)

    def fill_missing(self, x, missing):
        return x.fill_nan(missing).fill_null(missing)

    @staticmethod
    def from_data(x, n_quantiles: int) -> dict:
        q = np.linspace(0.005, 0.995, n_quantiles)
        quantiles = np.nanquantile(x.to_numpy(), q).tolist()

        return {"quantiles": quantiles}


# class PolarsRankScaler(RankScaler):

#     def __init__(self, mean, std, quantiles):
#         self.mean = mean
#         self.std = std
#         self.quantiles = quantiles
#         self.n_quantiles = len(quantiles)
#         self.q_array = np.array(self.quantiles)

#     def __call__(self, x):
#         x = (x - self.mean) / self.std
#         q = pl.Series(self.q_array[1:])
#         # Ultra-defensive
#         x = x.fill_nan(self.missing_value).fill_null(self.missing_value)
#         return pl.concat_list(
#             x,
#             x.cut(q, labels=np.arange(self.n_quantiles).astype("str")).cast(pl.Float32)
#             / self.n_quantiles,
#         )
