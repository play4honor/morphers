import torch

from ..base.categorical import Integerizer


class PolarsIntegerizer(Integerizer):

    def __call__(self, x):
        return x.map_dict(self.vocab, default=len(self.vocab))

    @classmethod
    def from_data(cls, x):
        vocab = {t: i for i, t in enumerate(x.filter(x.is_not_null()).unique())}

        return cls(vocab)
