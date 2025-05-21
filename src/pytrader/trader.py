from abc import ABC
from typing import TypeVar, Generic

from pytrader import Exchange, Algorithm

T = TypeVar('T', bound=Exchange)


class Trader(Generic[T], ABC):

    def __init__(self, exchange: T, algorithm: Algorithm):

        self._exchange: T = exchange
        self._algorithm: Algorithm = algorithm
