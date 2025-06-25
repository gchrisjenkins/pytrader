from abc import ABC, abstractmethod
from typing import Any, cast

from pytrader import Algorithm, Market, Exchange, Trader, Settings


class SimulatedMarket(Market, ABC):

    @abstractmethod
    def step(self, *args, **kwargs) -> dict[str, Any]:
        pass


class SimulatedExchange(Exchange, ABC):

    @abstractmethod
    def step(self, *args, **kwargs) -> int:
        pass

    @property
    def current_step(self):
        return self._get_current_step()

    @abstractmethod
    def _get_current_step(self) -> int:
        pass


class SimulatedTrader(Trader[SimulatedExchange]):

    def __init__(self, exchange: SimulatedExchange, algorithm: Algorithm):
        super().__init__(exchange, algorithm)

        step_duration_seconds = cast(Settings, exchange.settings).step_duration_seconds
        self._steps_per_trade = int(algorithm.trade_interval_seconds / step_duration_seconds)

    def step(self) -> bool:

        current_step = self._exchange.step()

        if current_step % self._steps_per_trade == 0:
            self._algorithm.on_update(self._exchange)
            if self._algorithm.is_ready:
                self._algorithm.on_trade(self._exchange)

        if self._algorithm.has_terminated:
            return True

        return False

    def steps_per_trade(self) -> int:
        return self._steps_per_trade

    @property
    def current_step(self) -> int:
        return self._exchange.current_step
