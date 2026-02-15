from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Generic, TypeVar

from pytrader.exchange import Exchange


class Indicator(ABC):
    def __init__(self, value: float | None):
        self._is_valid: bool = False
        self._value: float | None = value

    def update(self, exchange: Exchange):
        self._update(*self._transform(exchange))

    @abstractmethod
    def _transform(self, exchange: Exchange) -> tuple[float, ...]:
        pass

    @abstractmethod
    def _update(self, *args: float):
        pass

    @property
    def is_valid(self) -> bool:
        return self._is_valid

    @property
    def value(self) -> float | None:
        return self._value


class SimpleMovingAverage(Indicator, ABC):
    def __init__(self, period: int, value: float | None):
        if period < 1:
            raise ValueError("'period' must be >= 1")

        super().__init__(value)

        self._values: deque[float] = deque(maxlen=period)
        if value is not None:
            self._values.append(value)

    def _update(self, value: float):
        self._values.append(value)
        self._value = sum(self._values) / len(self._values)
        self._is_valid = len(self._values) == self._values.maxlen


class ExponentialMovingAverage(Indicator, ABC):
    def __init__(self, period: int, value: float | None):
        if period < 1:
            raise ValueError("'period' must be >= 1")

        super().__init__(value)
        self._alpha: float = 2.0 / (period + 1.0)

    def _update(self, value: float):
        if self._value is None:
            self._value = value
        else:
            self._value = self._alpha * value + (1.0 - self._alpha) * self._value

        self._is_valid = True


class PriceEMA(ExponentialMovingAverage):
    """
    Exponential moving average for a market's last price.
    """

    def __init__(self, symbol: str, period: int, value: float | None = None):
        super().__init__(period, value)

        if not isinstance(symbol, str):
            raise TypeError(f"'symbol' must be of str type, but found: {type(symbol)}")
        self._symbol: str = symbol.strip()
        if not self._symbol:
            raise ValueError("'symbol' cannot be empty")

    def _transform(self, exchange: Exchange) -> tuple[float, ...]:
        snapshot = exchange.get_market_snapshot(self._symbol)
        if snapshot is None:
            raise ValueError(f"Market snapshot for symbol '{self._symbol}' not found")
        return (float(snapshot.last_price),)


class Strategy(ABC):
    """
    Exchange-agnostic strategy contract.

    Lifecycle:
    1. construction calls _build() for indicator registration
    2. on_startup(exchange)
    3. repeating: on_update(exchange) then on_trade(exchange) when is_ready
    4. on_shutdown(exchange)
    """

    def __init__(
        self,
        currency: str,
        symbols: str | list[str],
        trade_interval_seconds: int,
        name: str | None = None,
    ):
        if not isinstance(currency, str):
            raise TypeError(f"'currency' must be of str type, but found: {type(currency)}")
        self._currency: str = currency.strip()
        if not self._currency:
            raise ValueError("'currency' cannot be empty")

        self._symbols: list[str] = self._normalize_symbols(symbols)

        if not isinstance(trade_interval_seconds, int):
            raise TypeError(
                f"'trade_interval_seconds' must be of int type, but found: {type(trade_interval_seconds)}"
            )
        if trade_interval_seconds < 1:
            raise ValueError("'trade_interval_seconds' must be >= 1")
        self._trade_interval_seconds: int = trade_interval_seconds

        if name is not None and not isinstance(name, str):
            raise TypeError(f"'name' must be of str type when provided, but found: {type(name)}")
        self._name: str = name.strip() if isinstance(name, str) and name.strip() else type(self).__name__

        # name -> (update_order, insertion_index, indicator)
        self._indicator_bindings: dict[str, tuple[int, int, Indicator]] = {}
        self._next_indicator_index: int = 0

        self._build()

    @staticmethod
    def _normalize_symbols(symbols: str | list[str]) -> list[str]:
        if isinstance(symbols, str):
            values = [symbols]
        elif isinstance(symbols, list):
            if len(symbols) == 0:
                raise ValueError("'symbols' cannot be empty")
            values = symbols
        else:
            raise TypeError(f"'symbols' must be a str or list[str], but found: {type(symbols)}")

        normalized: list[str] = []
        seen: set[str] = set()
        for symbol in values:
            if not isinstance(symbol, str):
                raise TypeError(f"Each symbol must be of str type, but found: {type(symbol)}")

            value = symbol.strip()
            if not value:
                raise ValueError("Symbols cannot contain empty values")
            if value in seen:
                raise ValueError(f"Duplicate symbol '{value}' is not allowed")

            seen.add(value)
            normalized.append(value)

        return normalized

    @abstractmethod
    def _build(self) -> None:
        pass

    def _add_indicator(self, name: str, indicator: Indicator, update_order: int) -> None:
        if not isinstance(name, str):
            raise TypeError(f"'name' must be of str type, but found: {type(name)}")
        indicator_name = name.strip()
        if not indicator_name:
            raise ValueError("'name' cannot be empty")
        if indicator_name in self._indicator_bindings:
            raise ValueError(f"Indicator with name '{indicator_name}' already exists")

        if not isinstance(indicator, Indicator):
            raise TypeError(f"'indicator' must be an Indicator instance, but found: {type(indicator)}")

        if not isinstance(update_order, int):
            raise TypeError(f"'update_order' must be of int type, but found: {type(update_order)}")
        if update_order < 0:
            raise ValueError("'update_order' must be >= 0")

        self._indicator_bindings[indicator_name] = (update_order, self._next_indicator_index, indicator)
        self._next_indicator_index += 1

    def _get_ordered_indicator_pairs(self) -> list[tuple[str, Indicator]]:
        ordered = sorted(
            self._indicator_bindings.items(),
            key=lambda item: (item[1][0], item[1][1]),
        )
        return [(name, binding[2]) for name, binding in ordered]

    @property
    def indicators(self) -> dict[str, Indicator]:
        return {name: indicator for name, indicator in self._get_ordered_indicator_pairs()}

    def get_indicator(self, name: str) -> Indicator:
        binding = self._indicator_bindings.get(name)
        if binding is None:
            raise KeyError(f"Unknown indicator '{name}'")
        return binding[2]

    def on_update(self, exchange: Exchange) -> None:
        for _, indicator in self._get_ordered_indicator_pairs():
            indicator.update(exchange)

    @property
    def currency(self) -> str:
        return self._currency

    @property
    def symbols(self) -> list[str]:
        return list(self._symbols)

    @property
    def trade_interval_seconds(self) -> int:
        return self._trade_interval_seconds

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_ready(self) -> bool:
        return self._get_is_ready()

    @abstractmethod
    def _get_is_ready(self) -> bool:
        pass

    @property
    def has_terminated(self) -> bool:
        return self._get_has_terminated()

    @abstractmethod
    def _get_has_terminated(self) -> bool:
        pass

    @property
    def state(self) -> dict[str, Any]:
        return self._get_state()

    @abstractmethod
    def _get_state(self) -> dict[str, Any]:
        pass

    @property
    def action(self) -> dict[str, Any]:
        return self._get_action()

    @abstractmethod
    def _get_action(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def on_startup(self, exchange: Exchange) -> None:
        pass

    @abstractmethod
    def on_shutdown(self, exchange: Exchange) -> None:
        pass

    @abstractmethod
    def on_trade(self, exchange: Exchange) -> None:
        pass


T = TypeVar("T", bound=Exchange)


class Trader(Generic[T]):
    def __init__(self, exchange: T, strategy: Strategy):
        if exchange is None:
            raise ValueError("'exchange' is required")
        if strategy is None:
            raise ValueError("'strategy' is required")

        self._exchange: T = exchange
        self._strategy: Strategy = strategy

    @property
    def exchange(self) -> T:
        return self._exchange

    @property
    def strategy(self) -> Strategy:
        return self._strategy

    @property
    def is_ready(self) -> bool:
        return self._strategy.is_ready

    @property
    def has_terminated(self) -> bool:
        return self._strategy.has_terminated

    @property
    def state(self) -> dict[str, Any]:
        return self._strategy.state

    @property
    def action(self) -> dict[str, Any]:
        return self._strategy.action

    def startup(self) -> None:
        self._strategy.on_startup(self._exchange)

    def update(self) -> None:
        self._strategy.on_update(self._exchange)

    def trade(self) -> None:
        self._strategy.on_trade(self._exchange)

    def shutdown(self) -> None:
        self._strategy.on_shutdown(self._exchange)
