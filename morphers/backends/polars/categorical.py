import torch

# Sorry
from ...base.base import MorpherBackend


class PolarsIntegerizerBackend(MorpherBackend):

    def __call__(self, x, vocab):
        return x.replace(vocab, default=len(vocab))

    def fill_missing(self, x, missing_value):
        return x.fill_null(missing_value)

    @staticmethod
    def from_data(x):
        vocab = {t: i for i, t in enumerate(x.filter(x.is_not_null()).unique())}
        return {"vocab": vocab}
