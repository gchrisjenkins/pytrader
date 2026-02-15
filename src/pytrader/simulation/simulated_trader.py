import dataclasses
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any, cast

from pytrader.exchange import OrderSide, Order, Market, Exchange
from pytrader import Algorithm, Trader


class SimulatedMarket(Market, ABC):

    @abstractmethod
    def step(self, *args, **kwargs) -> dict[str, Any]:
        pass

    @abstractmethod
    def create_market_order(self, side: OrderSide, quantity: Decimal) -> Order:
        pass

    @abstractmethod
    def create_limit_order(self, side: OrderSide, price: Decimal, quantity: Decimal) -> Order:
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> Order | None:
        pass


class SimulatedExchange(Exchange, ABC):

    def __init__(self, settings: "SimulatedExchange.Settings"):

        self._settings: SimulatedExchange.Settings = settings

    @property
    def settings(self) -> "SimulatedExchange.Settings":
        return self._settings

    @abstractmethod
    def step(self, *args, **kwargs) -> int:
        pass

    @abstractmethod
    def _get_current_step(self) -> int:
        pass

    @property
    def current_step(self):
        return self._get_current_step()

    @dataclasses.dataclass
    class Settings:
        maker_fee: Decimal
        taker_fee: Decimal
        funding_interval: int
        underlying_volatility: float
        step_duration_seconds: float
        seed: int = 42


class SimulatedTrader(Trader[SimulatedExchange]):

    def __init__(self, exchange: SimulatedExchange, algorithm: Algorithm):
        super().__init__(exchange, algorithm)

        step_duration_seconds = exchange.settings.step_duration_seconds
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
