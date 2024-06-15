import polars as pl
import torch
from morphers import Integerizer
from morphers.backends.polars.categorical import PolarsIntegerizerBackend


def test_integerizer():
    testo = pl.DataFrame({"a": torch.randint(0, 10, [100]).numpy()})
    test_morpher = Integerizer.from_data(testo["a"])
    assert isinstance(test_morpher, Integerizer)
    assert isinstance(test_morpher.backend, PolarsIntegerizerBackend)
    assert isinstance(test_morpher(testo["a"]), pl.Series)
