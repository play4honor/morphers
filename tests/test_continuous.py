import polars as pl
import torch
from morphers import Normalizer
from morphers.backends.polars import PolarsNormalizerBackend


def test_normalizer():
    testo = pl.DataFrame({"a": torch.randn([100]).numpy()})
    test_morpher = Normalizer.from_data(testo["a"])
    assert isinstance(test_morpher, Normalizer)
    assert isinstance(test_morpher.backend, PolarsNormalizerBackend)
    assert isinstance(test_morpher(testo["a"]), pl.Series)
