from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Any, TypeVar, Generic

from pytrader.util import normalize_time_unit_label, time_unit_label_to_timedelta


class MarketData(ABC):

    def __init__(self, symbol: str):
        self._set_symbol(symbol)

    @property
    def symbol(self) -> str:
        return self._symbol

    def _set_symbol(self, value: str):

        if value is None:
            raise ValueError("'symbol' must not be None")

        if not isinstance(value, str):
            raise TypeError(f"'symbol' must be a str type, but found: {type(value)}")

        if not value:
            raise ValueError("'symbol' cannot be empty")

        self._symbol = value


class PriceData(MarketData):

    def __init__(self, symbol: str, last_price: float):
        super().__init__(symbol)

        self._set_last_price(last_price)

    @property
    def last_price(self) -> float:
        return self._last_price

    @last_price.setter
    def last_price(self, value: float):
        self._set_last_price(value)

    def _set_last_price(self, value: float):

        if value is None:
            raise ValueError("'last_price' must not be None")

        if not isinstance(value, float):
            raise TypeError(f"'last_price' must be a float type, but found: {type(value)}")

        if not (value > 0):
            raise ValueError(
                f"[{self.symbol}] 'last_price' must be > 0, but found {value}"
            )

        self._last_price = value


class PerpetualFuturesMarketData(PriceData):

    def __init__(self, symbol: str, last_price: float, mark_price: float, index_price: float, funding_rate: float):
        super().__init__(symbol, last_price)

        self._set_mark_price(mark_price)
        self._set_index_price(index_price)
        self._set_funding_rate(funding_rate)

    @property
    def mark_price(self) -> float:
        return self._mark_price

    @mark_price.setter
    def mark_price(self, value: float):
        self._set_mark_price(value)

    def _set_mark_price(self, value: float):

        if value is None:
            raise ValueError("'mark_price' must not be None")

        if not isinstance(value, float):
            raise TypeError(f"'mark_price' must be a float type, but found: {type(value)}")

        if not (value > 0):
            raise ValueError(
                f"[{self.symbol}] 'mark_price' must be > 0, but found {value}"
            )

        self._mark_price = value

    @property
    def index_price(self) -> float:
        return self._index_price

    @index_price.setter
    def index_price(self, value: float):
        self._set_index_price(value)

    def _set_index_price(self, value: float):

        if value is None:
            raise ValueError("'index_price' must not be None")

        if not isinstance(value, float):
            raise TypeError(f"'index_price' must be a float type, but found: {type(value)}")

        if not (value > 0):
            raise ValueError(
                f"[{self.symbol}] 'index_price' must be > 0, but found {value}"
            )

        self._index_price = value

    @property
    def funding_rate(self) -> float:
        return self._funding_rate

    @funding_rate.setter
    def funding_rate(self, value: float):
        self._set_funding_rate(value)

    def _set_funding_rate(self, value: float):

        if value is None:
            raise ValueError("'funding_rate' must not be None")

        if not isinstance(value, float):
            raise TypeError(f"'funding_rate' must be a float type, but found: {type(value)}")

        if not (value > 0):
            raise ValueError(
                f"[{self.symbol}] 'funding_rate' must be > 0, but found {value}"
            )

        self._funding_rate = value


T = TypeVar('T', bound=MarketData)


class MarketDataGenerator(Generic[T], ABC):

    def __init__(self, step_duration: timedelta, time_unit: float | str | timedelta = "1m"):

        self._set_step_duration(step_duration)

    @abstractmethod
    def step(self, *args: Any, **kwargs: Any) -> dict[str, T]:
        pass

    @property
    def step_duration(self) -> timedelta:
        return self._step_duration

    def _set_step_duration(self, value: timedelta) -> None:

        if not isinstance(value, timedelta):
            raise TypeError(
                f"'step_duration' must be provided as datetime.timedelta, but found: {type(value)}"
            )

        if value.total_seconds() <= 0:
            raise ValueError("'step_duration' must be positive")

        self._step_duration = value

    @property
    def time_unit(self) -> timedelta:
        return self._time_unit

    def _set_time_unit(self, value: str | timedelta) -> None:

        if value is None:
            raise ValueError("'time_unit' must not be None")

        if isinstance(value, timedelta):
            time_unit = value
        elif isinstance(value, str):
            label = normalize_time_unit_label(value)
            time_unit = time_unit_label_to_timedelta(label)
        else:
            raise TypeError(
                f"'time_unit' must be provided as a string like '1m' or a datetime.timedelta, but found: {type(value)}"
            )

        if time_unit.total_seconds() <= 0:
            raise ValueError("'time_unit' must be positive")

        self._time_unit = time_unit
