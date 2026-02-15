import logging
from datetime import timedelta
from typing import Any

from pytrader.simulation.data import MarketDataGenerator, PerpetualFuturesMarketData, PriceData


class SimplePerpetualFuturesMarketDataGenerator(MarketDataGenerator[PerpetualFuturesMarketData]):

    def __init__(self,
        initial_margin_rate: float, maintenance_margin_rate: float, interest_rate: float, premium_volatility: float,
        funding_basis_ema_alpha: float, premium_rate_min: float, premium_rate_max: float,
        price_data_generator: MarketDataGenerator[PriceData], step_duration: timedelta, seed: int = 42
    ):
        super().__init__(step_duration)

        cls = type(self)
        self._logger: logging.Logger = logging.getLogger(f"{cls.__module__}.{cls.__name__}")

    def step(self, *args: Any, **kwargs: Any) -> dict[str, PerpetualFuturesMarketData]:
        pass

    class _Market(PerpetualFuturesMarketData):

        def __init__(self,
            symbol: str, last_price: float, interest_rate: float, premium_volatility: float, funding_basis: float,
            funding_basis_ema_alpha: float, premium_rate_min: float, premium_rate_max: float
        ):
            super().__init__(symbol, last_price, last_price, last_price, interest_rate)

            self._set_interest_rate(interest_rate)
            self._set_premium_volatility(premium_volatility)
            self._set_funding_basis(funding_basis)
            self._set_funding_basis_ema_alpha(funding_basis_ema_alpha)
            self._set_premium_rate_min(premium_rate_min)
            self._set_premium_rate_max(premium_rate_max)

            # self._set_last_price(initial_price)
            # self._set_index_price(self.last_price)  # Start index price at last price
            # self._set_funding_basis(Decimal("0.0"))
            # self._set_mark_price(self.last_price)
            # self._set_funding_rate(self.interest_rate)  # Initial funding rate
            #
            #
            # self._set_volatility(volatility)
            # self._set_correlation(correlation)

            # self._data = PerpetualFuturesMarketData(symbol, last_price, last_price, last_price, interest_rate)

        @property
        def interest_rate(self) -> float:
            return self._interest_rate

        def _set_interest_rate(self, value: float) -> None:

            if value is None:
                raise ValueError(f"[{self.symbol}] 'interest_rate' must not be None")

            if not isinstance(value, float):
                raise TypeError(f"[{self.symbol}] 'interest_rate' must be a float type, but found: {type(value)}")

            if value < 0:
                raise ValueError(f"[{self.symbol}] 'interest_rate' must be non-negative, but found: {value}")

            self._interest_rate = value

        @property
        def premium_volatility(self) -> float:
            return self._premium_volatility

        def _set_premium_volatility(self, value: float) -> None:

            if value is None:
                raise ValueError(f"[{self.symbol}] 'premium_volatility' must not be None")

            if not isinstance(value, float):
                raise TypeError(f"[{self.symbol}] 'premium_volatility' must be a float type, but found: {type(value)}")

            if value < 0:
                raise ValueError(f"[{self.symbol}] 'premium_volatility' must be non-negative, but found: {value}")

            self._premium_volatility = value

        @property
        def funding_basis(self) -> float:
            return self._funding_basis

        def _set_funding_basis(self, value: float) -> None:

            if value is None:
                raise ValueError(f"[{self.symbol}] 'funding_basis' must not be None")

            if not isinstance(value, float):
                raise TypeError(f"[{self.symbol}] 'funding_basis' must be a float type, but found: {type(value)}")

            if value < 0:
                raise ValueError(f"[{self.symbol}] 'funding_basis' must be non-negative, but found: {value}")

            self._funding_basis = value

        @property
        def funding_basis_ema_alpha(self) -> float:
            return self._funding_basis_ema_alpha

        def _set_funding_basis_ema_alpha(self, value: float) -> None:

            if value is None:
                raise ValueError(f"[{self.symbol}] 'funding_basis_ema_alpha' must not be None")

            if not isinstance(value, float):
                raise TypeError(f"[{self.symbol}] 'funding_basis_ema_alpha' must be a float type, but found: {type(value)}")

            if not (0 < value <= 1):
                raise ValueError(
                    f"[{self.symbol}] 'funding_basis_ema_alpha' must be between 0 and 1 (inclusive), but found: {value}"
                )

            self._funding_basis_ema_alpha = value

        @property
        def premium_rate_min(self) -> float:
            return self._premium_rate_min

        def _set_premium_rate_min(self, value: float) -> None:

            if value is None:
                raise ValueError(f"[{self.symbol}] 'premium_rate_min' must not be None")

            if not isinstance(value, float):
                raise TypeError(f"[{self.symbol}] 'premium_rate_min' must be a float type, but found: {type(value)}")

            self._premium_rate_min = value

        @property
        def premium_rate_max(self) -> float:
            return self._premium_rate_max

        def _set_premium_rate_max(self, value: float) -> None:

            if value is None:
                raise ValueError(f"[{self.symbol}] 'premium_rate_max' must not be None")

            if not isinstance(value, float):
                raise TypeError(f"[{self.symbol}] 'premium_rate_max' must be a float type, but found: {type(value)}")

            self._premium_rate_max = value
