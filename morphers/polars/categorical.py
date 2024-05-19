import torch

from ..base.categorical import Integerizer


class PolarsIntegerizer(Integerizer):

    def __call__(self, x):
        return x.map_dict(self.vocab, default=len(self.vocab))

    def fill_missing(self, x):
        return x.fill_null(self.MISSING_VALUE)

    @classmethod
    def from_data(cls, x):
        vocab = {t: i for i, t in enumerate(x.filter(x.is_not_null()).unique())}

        return cls(vocab)
