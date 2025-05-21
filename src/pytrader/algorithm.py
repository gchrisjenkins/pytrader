from abc import abstractmethod, ABC
from collections import deque
from typing import Any, Callable, TypeAlias

from pytrader import Exchange


class Indicator(ABC):

    def __init__(self):

        self._value: float | None = None

    @abstractmethod
    def is_valid(self) -> bool:
        pass

    @abstractmethod
    def update(self, *args: Any):
        pass

    @property
    def value(self) -> float | None:
        return self._value


class SimpleMovingAverage(Indicator):

    def __init__(self, period: int):
        super().__init__()

        self._values: deque[float] = deque(maxlen=period)

    def is_valid(self) -> bool:
        return len(self._values) == self._values.maxlen

    def update(self, value: float):

        self._values.append(value)
        self._value = sum(self._values) / len(self._values)


class ExponentialMovingAverage(Indicator):

    def __init__(self, period: int):
        super().__init__()

        self._alpha: float = 2. / (period + 1.)

    def is_valid(self) -> bool:
        return self._value is not None

    def update(self, value: float):

        if self._value is None:
            self._value = value
        else:
            self._value = self._alpha * value + (1. - self._alpha) * self._value


IndicatorDefType: TypeAlias = tuple[Indicator, Callable[[Exchange], tuple[float, ...]], int]


class Algorithm(ABC):

    def __init__(self):

        self._indicator_defs: dict[str, IndicatorDefType] = self._build_indicator_defs()

    def _get_indicator(self, name: str) -> Indicator:
        return self._indicator_defs[name][0]

    def on_update(self, exchange: Exchange):

        # Sort the indicators by the update order
        sorted_indicators = sorted(self._indicator_defs.items(), key=lambda item: item[1][2])

        # Iterate and update the indicators in the specified order
        for key, (indicator, transform_fn, _) in sorted_indicators:
            values = transform_fn(exchange)  # Get the values from the callable
            indicator.update(*values)  # Update the indicator with the computed values

    @abstractmethod
    def _build_indicator_defs(self) -> dict[str, IndicatorDefType]:
        pass

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
    def on_startup(self, exchange: Exchange):
        pass

    @abstractmethod
    def on_shutdown(self, exchange: Exchange):
        pass

    @abstractmethod
    def on_trade(self, exchange: Exchange):
        pass
