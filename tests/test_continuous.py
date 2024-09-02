import polars as pl
import torch
from morphers import Normalizer, Quantiler
from morphers.backends.polars import PolarsNormalizerBackend, PolarsQuantilerBackend


def test_normalizer():
    testo = pl.DataFrame({"a": torch.randn([100]).numpy()})
    test_morpher = Normalizer.from_data(testo["a"])
    assert isinstance(test_morpher, Normalizer)
    assert isinstance(test_morpher.backend, PolarsNormalizerBackend)
    assert isinstance(test_morpher(testo["a"]), pl.Series)


def test_quantiler():
    testo = pl.DataFrame({"a": torch.randn([100]).numpy()})
    test_morpher = Quantiler.from_data(testo["a"], n_quantiles=50)
    assert isinstance(test_morpher, Quantiler)
    assert isinstance(test_morpher.backend, PolarsQuantilerBackend)
    assert isinstance(test_morpher(testo["a"]), pl.Series)
