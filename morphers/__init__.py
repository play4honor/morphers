from .morphers.categorical import Integerizer, BigIntegerizer
from .morphers.continuous import (
    Normalizer,
    RankScaler,
    Quantiler,
    MissingIndicatorQuantiler,
)

__version__ = "0.1.2"
