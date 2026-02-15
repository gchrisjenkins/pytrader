import logging
import math
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from pytrader.data import load_and_assemble
from pytrader.simulation.data import MarketDataGenerator
from pytrader.simulation.data import PriceData
from pytrader.util import normalize_time_unit_label, time_unit_label_to_timedelta

__all__ = ["SimplePriceDataGenerator"]


class SimplePriceDataGenerator(MarketDataGenerator[PriceData]):

    def __init__(self, config: dict[str, Any], step_duration: timedelta, seed: int = 42) -> None:
        super().__init__(step_duration)

        cls = type(self)
        self._logger: logging.Logger = logging.getLogger(f"{cls.__module__}.{cls.__name__}")

        self._set_time_unit(config.get("time_unit"))
        self._set_base_volatility(config.get("base_volatility"))

        self._markets: dict[str, "SimplePriceDataGenerator._Market"] = {}
        for market in config["markets"]:
            symbol = market["symbol"]
            initial_price = market["initial_price"]
            volatility = market["volatility"]
            correlation = market["correlation"]
            self._markets[symbol] = self._Market(symbol, initial_price, volatility, correlation)

        self._rng = np.random.default_rng(seed)

    def step(self, *args: Any, **kwargs: Any) -> dict[str, Any]:

        data: dict[str, Any] = {}

        scale = math.sqrt(self.step_duration.total_seconds() / self.time_unit.total_seconds())

        base_sigma = self.base_volatility * scale
        underlying_noise = self._rng.normal(0, base_sigma)
        for market in self._markets.values():

            # Rescale the calibrated asset volatility to the step interval
            asset_sigma = float(market.volatility) * scale
            corr = market.correlation

            shared_component = 0.0
            if base_sigma > 0 and asset_sigma > 0 and corr != 0.0:
                shared_component = corr * (asset_sigma / base_sigma) * underlying_noise

            idio_sigma = asset_sigma * math.sqrt(max(1.0 - corr ** 2, 0.0))
            market_specific_noise = self._rng.normal(0.0, idio_sigma)

            # Sum shared and idiosyncratic noise to obtain the per-asset return
            noise = shared_component + market_specific_noise

            # Update price using the computed return
            market.last_price = float(market.last_price) * (1.0 + noise)

            data[market.symbol] = market.last_price

        return data

    @property
    def base_volatility(self) -> float:
        return self._base_volatility

    def _set_base_volatility(self, value: Any) -> None:

        if value is None:
            raise ValueError("'base_volatility' must not be None")

        if not isinstance(value, float):
            raise TypeError(f"'base_volatility' must be a float type, but found: {type(value)}")

        if value < 0:
            raise ValueError(f"'base_volatility' cannot be negative, but found {value}")

        self._base_volatility = value

    # @property
    # def time_unit(self) -> timedelta:
    #     return self._time_unit
    #
    # def _set_time_unit(self, value: str | timedelta | None) -> None:
    #
    #     if value is None:
    #         raise ValueError("'time_unit' must not be None")
    #
    #     if isinstance(value, timedelta):
    #         time_unit = value
    #     elif isinstance(value, str):
    #         label = normalize_time_unit_label(value)
    #         time_unit = time_unit_label_to_timedelta(label)
    #     else:
    #         raise TypeError(
    #             f"'time_unit' must be provided as a string like '1m' or a datetime.timedelta, but found: {type(value)}"
    #         )
    #
    #     if time_unit.total_seconds() <= 0:
    #         raise ValueError("'time_unit' must be positive")
    #
    #     self._time_unit = time_unit

    @classmethod
    def calibrate(
        cls,
        *,
        data_dir: str,
        symbols: list[str],
        candle_duration: str,
        start_date: str | None = None,
        end_date: str | None = None,
        max_candles: int | None = None,
        min_segment_length: int | None = None,
        allow_multi_segments: bool = True,
    ) -> dict[str, Any]:

        if not symbols:
            raise ValueError("'symbols' must contain at least one entry")

        logger = logging.getLogger(f"{cls.__module__}.{cls.__name__}")

        assembled = load_and_assemble(
            data_dir=data_dir,
            symbols=list(symbols),
            candle_duration=candle_duration,
            start_date=start_date,
            end_date=end_date,
            max_candles=max_candles,
            min_segment_length=min_segment_length,
            allow_multi_segments=allow_multi_segments,
        )

        time_unit_label = normalize_time_unit_label(candle_duration)
        expected_delta = time_unit_label_to_timedelta(time_unit_label)
        assembled_delta = assembled.time_interval

        if not math.isclose(
            expected_delta.total_seconds(),
            assembled_delta.total_seconds(),
            rel_tol=1e-9,
            abs_tol=1e-6,
        ):
            logger.warning(
                "Calibrated data interval %s does not match expected interval %s derived from '%s'",
                assembled_delta,
                expected_delta,
                candle_duration,
            )

        close_cols: list[str] = []
        for sym in symbols:
            col = f"close_{sym}"
            if col not in assembled.df.columns:
                raise KeyError(f"Close price column '{col}' not found in assembled data")
            close_cols.append(col)

        close_prices = assembled.df[close_cols].dropna()
        if len(close_prices) < 2:
            raise ValueError("Not enough data to compute returns for calibration")

        log_prices = np.log(close_prices)
        returns = log_prices.diff().dropna()
        if returns.empty:
            raise ValueError("Return series is empty after differencing\u2014cannot calibrate")

        returns.columns = list(symbols)

        returns_matrix = returns.to_numpy()
        cov = np.cov(returns_matrix, rowvar=False)

        eig_vals, eig_vecs = np.linalg.eigh(cov)
        idx_max = int(np.argmax(eig_vals))
        factor_weights = eig_vecs[:, idx_max]

        factor_scores = returns_matrix @ factor_weights
        reference_series = returns_matrix[:, 0]
        ref_correlation = np.corrcoef(reference_series, factor_scores)[0, 1]
        if np.isnan(ref_correlation):
            ref_correlation = 0.0
        if ref_correlation < 0:
            factor_weights = -factor_weights
            factor_scores = -factor_scores

        factor_series = pd.Series(factor_scores, index=returns.index, name="factor")
        base_volatility = float(factor_series.std(ddof=1))
        if base_volatility <= 0:
            raise ValueError("Calibrated base volatility is non-positive")

        markets: list[dict[str, float | str]] = []
        for sym in symbols:
            asset_returns = returns[sym]
            sigma = float(asset_returns.std(ddof=1))
            if sigma <= 0 or math.isnan(sigma):
                logger.warning("Computed non-positive variance for %s; treating volatility as 0", sym)
                sigma = 0.0

            if sigma > 0 and base_volatility > 0:
                corr = float(asset_returns.corr(factor_series))
            else:
                corr = 0.0
            if math.isnan(corr):
                corr = 0.0
            corr = max(min(corr, 0.999999), -0.999999)

            initial_price = float(close_prices[f"close_{sym}"].iloc[-1])

            markets.append(
                {
                    "symbol": sym,
                    "volatility": sigma,
                    "correlation": corr,
                    "initial_price": initial_price,
                }
            )

        config: dict[str, Any] = {
            "time_unit": time_unit_label,
            "base_volatility": float(base_volatility),
            "markets": markets,
        }

        logger.info(
            "Calibrated generator for symbols %s with base_volatility %.6f",
            ", ".join(symbols),
            config["base_volatility"],
        )

        return config

    class _Market(PriceData):

        def __init__(self, symbol: str, last_price: float, volatility: float, correlation: float):
            super().__init__(symbol, last_price)

            self._set_volatility(volatility)
            self._set_correlation(correlation)

        @property
        def volatility(self) -> float:
            return self._volatility

        def _set_volatility(self, value: float) -> None:

            if value is None:
                raise ValueError("'volatility' must not be None")

            if not isinstance(value, float):
                raise TypeError(f"'volatility' must be a float type, but found: {type(value)}")

            if value < 0:
                raise ValueError(f"[{self.symbol}] 'volatility' cannot be negative, but found {value}")

            self._volatility = value

        @property
        def correlation(self) -> float:
            return self._correlation

        def _set_correlation(self, value: float) -> None:

            if value is None:
                raise ValueError("'correlation' must not be None")

            if not isinstance(value, float):
                raise TypeError(f"'correlation' must be a float type, but found: {type(value)}")

            if not (-1.0 <= value <= 1.0):
                raise ValueError(
                    f"[{self.symbol}] 'correlation' must be in range -1.0 and 1.0, but found {value}"
                )

            self._correlation = value


if __name__ == "__main__":

    from datetime import timedelta
    import json
    import matplotlib.pyplot as plt

    data_dir = "/home/cjenkins/Documents/Git/apex-omni-trader/market_data/binance"
    candle_duration = "1m"
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
    start_date = "2025-06-01 08:00:00"
    end_date = "2025-09-01"
    max_candles = 30_000
    min_segment_length = 10_000
    allow_multi_segments = True
    seed = 44

    steps = 60 * 60 * 24

    config = SimplePriceDataGenerator.calibrate(
        data_dir=data_dir,
        symbols=symbols,
        candle_duration=candle_duration,
        start_date=start_date,
        end_date=end_date,
        max_candles=max_candles,
        min_segment_length=min_segment_length,
        allow_multi_segments=allow_multi_segments,
    )

    print("Calibrated generator configuration:")
    print(json.dumps(config, indent=2))

    # config = {
    #   "time_unit": "1m",
    #   "base_volatility": 0.0017110226592335904,
    #   "markets": [
    #     {
    #       "symbol": "BTCUSDT",
    #       "volatility": 0.0004912481309027645,
    #       "correlation": 0.8122669879015095,
    #       "initial_price": 108246.35
    #     },
    #     {
    #       "symbol": "ETHUSDT",
    #       "volatility": 0.0010101149838506029,
    #       "correlation": 0.9119822789273377,
    #       "initial_price": 4391.83
    #     },
    #     {
    #       "symbol": "SOLUSDT",
    #       "volatility": 0.0011841398845172386,
    #       "correlation": 0.9439240304456611,
    #       "initial_price": 200.62
    #     },
    #     {
    #       "symbol": "XRPUSDT",
    #       "volatility": 0.0009070097279220328,
    #       "correlation": 0.9027356978933576,
    #       "initial_price": 2.7757
    #     }
    #   ]
    # }
    #
    # step_duration = timedelta(seconds=1)
    # generator = SimplePriceDataGenerator(config, step_duration=step_duration, seed=seed)
    #
    # initial_prices = {
    #     market["symbol"]: float(market["initial_price"]) for market in config["markets"]
    # }
    # normalized_prices: dict[str, list[float]] = {
    #     symbol: [1.0] for symbol in initial_prices.keys()
    # }
    #
    # for _ in range(steps):
    #     snapshot = generator.step()
    #     for symbol, price in snapshot.items():
    #         normalized_prices[symbol].append(price / initial_prices[symbol])
    #
    # for symbol in normalized_prices:
    #     plt.plot(normalized_prices[symbol], label=symbol)
    #
    # plt.title("Simulated Market Prices (Normalized)")
    # plt.xlabel("Step")
    # plt.ylabel("Price / Initial Price")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
