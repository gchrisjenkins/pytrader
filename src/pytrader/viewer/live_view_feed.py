from __future__ import annotations

import copy
import dataclasses
import time
from datetime import timedelta
from decimal import Decimal
from typing import Any

from pytrader.exchange import Account, MarketDataMessage, NormalizedMarketDataMessage, TimeUnit, convert_to_decimal
from pytrader.trader import Trader
from pytrader.viewer.models import (
    AccountViewState,
    MarketViewState,
    Ohlc,
    OrderView,
    PositionView,
    StrategyIndicator,
    StrategyViewState,
    TraderViewDataUpdate,
    TraderViewFrame,
    WalletView,
)


def _enum_to_str(value: Any) -> str | None:
    if value is None:
        return None
    enum_value = getattr(value, "value", None)
    if enum_value is not None:
        return str(enum_value)
    return str(value)


_CANDLE_INTERVAL_TO_SECONDS: dict[str, int] = {
    "1m": 60,
    "5m": 5 * 60,
    "15m": 15 * 60,
    "1h": 60 * 60,
}


def _normalize_candle_interval(value: str | None) -> str:
    if not isinstance(value, str):
        raise ValueError("Candle interval must be a string")
    normalized = value.strip().lower()
    if normalized not in _CANDLE_INTERVAL_TO_SECONDS:
        supported = ", ".join(sorted(_CANDLE_INTERVAL_TO_SECONDS.keys()))
        raise ValueError(f"Unsupported candle interval '{value}'. Supported intervals: {supported}")
    return normalized


def _candle_interval_from_time_unit(unit: TimeUnit | None) -> str | None:
    if unit is None:
        return None
    if unit == TimeUnit.MINUTE:
        return "1m"
    if unit == TimeUnit.HOUR:
        return "1h"
    return None


@dataclasses.dataclass(frozen=True)
class LiveTraderViewFeedConfiguration:
    # Preferred candle interval selector for live viewer candles.
    default_candle_interval: str = "1m"
    # Backward-compatible fallback for older callers that still pass candle_time_unit.
    candle_time_unit: TimeUnit | None = None
    max_wallet_rows: int = 8
    max_position_rows: int = 12
    max_order_rows: int = 25


class LiveTraderViewFeed:
    def __init__(
        self,
        trader: Trader,
        *,
        config: LiveTraderViewFeedConfiguration | None = None,
    ) -> None:
        if trader is None:
            raise ValueError("'trader' is required")

        self._trader = trader
        self._config = config or LiveTraderViewFeedConfiguration()
        self._candle_by_symbol: dict[str, Ohlc] = {}
        self._candle_interval = self._resolve_default_candle_interval(self._config)

        if self._config.max_wallet_rows < 1:
            raise ValueError("'max_wallet_rows' must be >= 1")
        if self._config.max_position_rows < 1:
            raise ValueError("'max_position_rows' must be >= 1")
        if self._config.max_order_rows < 1:
            raise ValueError("'max_order_rows' must be >= 1")

    @property
    def candle_interval(self) -> str:
        return self._candle_interval

    def set_candle_interval(self, interval: str) -> bool:
        normalized = _normalize_candle_interval(interval)
        if normalized == self._candle_interval:
            return False
        self._candle_interval = normalized
        self._candle_by_symbol.clear()
        return True

    def build_update(
        self,
        message: MarketDataMessage | NormalizedMarketDataMessage,
    ) -> TraderViewDataUpdate | None:
        if message is None:
            return None

        symbol = message.symbol.strip().upper()
        market_state_payload = self._trader.exchange.get_market_state(symbol)
        if market_state_payload is None:
            return None

        market = self._build_market_view_state(symbol=symbol, payload=market_state_payload)
        event_ts_ms = market.event_ts_ms if market.event_ts_ms is not None else message.event_ts_ms
        ohlc = self._build_or_update_candle(
            symbol=symbol,
            event_ts_ms=event_ts_ms,
            last_price=market.last_price,
        )

        frame = self._build_frame(
            symbol=symbol,
            event_ts_ms=event_ts_ms,
            sequence=market.sequence if market.sequence is not None else message.sequence,
            ohlc=ohlc,
            market=market,
        )
        return TraderViewDataUpdate.from_frame(frame)

    def build_snapshot_update(
        self,
        *,
        symbol: str,
        event_ts_ms: int | None = None,
    ) -> TraderViewDataUpdate | None:
        normalized_symbol = symbol.strip().upper()
        if not normalized_symbol:
            return None

        market_state_payload = self._trader.exchange.get_market_state(normalized_symbol)
        if not isinstance(market_state_payload, dict):
            return None

        market = self._build_market_view_state(symbol=normalized_symbol, payload=market_state_payload)
        resolved_event_ts_ms = (
            market.event_ts_ms
            if market.event_ts_ms is not None
            else (event_ts_ms if event_ts_ms is not None else int(time.time() * 1000))
        )
        ohlc = self._build_or_update_candle(
            symbol=normalized_symbol,
            event_ts_ms=resolved_event_ts_ms,
            last_price=market.last_price,
        )
        frame = self._build_frame(
            symbol=normalized_symbol,
            event_ts_ms=resolved_event_ts_ms,
            sequence=market.sequence,
            ohlc=ohlc,
            market=market,
        )
        return TraderViewDataUpdate.from_frame(frame)

    def build_seed_updates(
        self,
        *,
        symbol: str,
    candles: list[dict[str, Any]],
    ) -> list[TraderViewDataUpdate]:
        normalized_symbol = symbol.strip().upper()
        if not normalized_symbol:
            return []

        bucket_size_ms = int(_CANDLE_INTERVAL_TO_SECONDS[self._candle_interval] * 1000)
        candle_rows: list[tuple[int, Ohlc]] = []
        for candle in candles:
            if not isinstance(candle, dict):
                continue
            open_time_ms = self._safe_int(candle.get("open_time_ms"))
            close_time_ms = self._safe_int(candle.get("close_time_ms"))
            if open_time_ms is None:
                continue
            if close_time_ms is None or close_time_ms < open_time_ms:
                close_time_ms = open_time_ms

            open_price = candle.get("open")
            high_price = candle.get("high")
            low_price = candle.get("low")
            close_price = candle.get("close")
            if any(value is None for value in (open_price, high_price, low_price, close_price)):
                continue
            try:
                open_decimal = convert_to_decimal(open_price)
                high_decimal = convert_to_decimal(high_price)
                low_decimal = convert_to_decimal(low_price)
                close_decimal = convert_to_decimal(close_price)
            except (TypeError, ValueError):
                continue

            bucket = int(open_time_ms // bucket_size_ms)
            candle_rows.append(
                (
                    close_time_ms,
                    Ohlc(
                        time=bucket,
                        open=open_decimal,
                        high=high_decimal,
                        low=low_decimal,
                        close=close_decimal,
                    ),
                )
            )

        if not candle_rows:
            return []

        candle_rows.sort(key=lambda item: (item[0], item[1].time))
        market_payload = self._trader.exchange.get_market_state(normalized_symbol)
        market = (
            self._build_market_view_state(symbol=normalized_symbol, payload=market_payload)
            if isinstance(market_payload, dict)
            else None
        )
        sequence = market.sequence if market is not None else None

        latest_candle = self._copy_ohlc(candle_rows[-1][1])
        if latest_candle is not None:
            self._candle_by_symbol[normalized_symbol] = latest_candle

        latest_event_ts_ms = candle_rows[-1][0]
        latest_ohlc = self._copy_ohlc(candle_rows[-1][1])
        frame = self._build_frame(
            symbol=normalized_symbol,
            event_ts_ms=latest_event_ts_ms,
            sequence=sequence,
            ohlc=latest_ohlc,
            market=market,
        )
        return [
            TraderViewDataUpdate(
                symbol=normalized_symbol,
                time=frame.time,
                frame=frame,
                reset_history=True,
                seed_points=[(event_ts_ms, self._copy_ohlc(ohlc) or ohlc) for event_ts_ms, ohlc in candle_rows],
            )
        ]

    def _build_frame(
        self,
        *,
        symbol: str,
        event_ts_ms: int,
        sequence: int | None,
        ohlc: Ohlc | None,
        market: MarketViewState | None,
    ) -> TraderViewFrame:
        return TraderViewFrame(
            symbol=symbol,
            time=timedelta(milliseconds=event_ts_ms),
            ohlc=ohlc,
            indicators=self._build_strategy_indicators(),
            market=market,
            account=self._build_account_view_state(),
            strategy=StrategyViewState(
                state=copy.deepcopy(self._trader.state),
                action=copy.deepcopy(self._trader.action),
            ),
            event_ts_ms=event_ts_ms,
            sequence=sequence,
        )

    def _build_market_view_state(self, *, symbol: str, payload: dict[str, Any]) -> MarketViewState:
        return MarketViewState(
            symbol=symbol,
            status=_enum_to_str(payload.get("status")),
            last_price=payload.get("last_price"),
            last_quantity=payload.get("last_quantity"),
            best_bid=payload.get("best_bid"),
            best_bid_quantity=payload.get("best_bid_quantity"),
            best_ask=payload.get("best_ask"),
            best_ask_quantity=payload.get("best_ask_quantity"),
            mark_price=payload.get("mark_price"),
            index_price=payload.get("index_price"),
            funding_rate=payload.get("funding_rate"),
            next_funding_time=payload.get("next_funding_time"),
            open_interest=payload.get("open_interest"),
            open_interest_value=payload.get("open_interest_value"),
            last_liquidation_side=_enum_to_str(payload.get("last_liquidation_side")),
            last_liquidation_price=payload.get("last_liquidation_price"),
            last_liquidation_quantity=payload.get("last_liquidation_quantity"),
            event_ts_ms=payload.get("last_market_data_event_ts_ms"),
            recv_ts_ms=payload.get("last_market_data_recv_ts_ms"),
            sequence=payload.get("last_market_data_sequence"),
        )

    def _build_or_update_candle(
        self,
        *,
        symbol: str,
        event_ts_ms: int,
        last_price: Decimal | None,
    ) -> Ohlc | None:
        if last_price is None:
            return self._copy_ohlc(self._candle_by_symbol.get(symbol))

        bucket_size_ms = int(_CANDLE_INTERVAL_TO_SECONDS[self._candle_interval] * 1000.0)
        bucket = int(event_ts_ms // bucket_size_ms)

        candle = self._candle_by_symbol.get(symbol)
        if candle is None or candle.time != bucket:
            candle = Ohlc(time=bucket, open=last_price)
            self._candle_by_symbol[symbol] = candle

        candle.high = max(candle.high, last_price) if candle.high is not None else last_price
        candle.low = min(candle.low, last_price) if candle.low is not None else last_price
        candle.close = last_price
        return self._copy_ohlc(candle)

    @staticmethod
    def _safe_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _resolve_default_candle_interval(config: LiveTraderViewFeedConfiguration) -> str:
        if config.candle_time_unit is not None:
            interval_from_unit = _candle_interval_from_time_unit(config.candle_time_unit)
            if interval_from_unit is None:
                raise ValueError(
                    f"'candle_time_unit' value '{config.candle_time_unit}' does not map to a supported interval"
                )
            return interval_from_unit
        return _normalize_candle_interval(config.default_candle_interval)

    @staticmethod
    def _copy_ohlc(candle: Ohlc | None) -> Ohlc | None:
        if candle is None:
            return None
        return Ohlc(
            time=candle.time,
            open=candle.open,
            high=candle.high,
            low=candle.low,
            close=candle.close,
        )

    def _build_strategy_indicators(self) -> list[StrategyIndicator]:
        indicators: list[StrategyIndicator] = []
        for name, indicator in self._trader.strategy.indicators.items():
            raw_value = indicator.value
            value: float | None = None
            if raw_value is not None:
                value = float(raw_value)
            indicators.append(
                StrategyIndicator(
                    name=name,
                    value=value,
                    is_price_indicator=True,
                )
            )
        return indicators

    def _build_account_view_state(self) -> AccountViewState | None:
        try:
            account = self._trader.exchange.get_account(self._trader.strategy.currency)
        except Exception:
            return None

        if account is None:
            return None

        if not isinstance(account, Account):
            return None

        wallets = self._build_wallet_rows(account=account)
        positions = self._build_position_rows(account=account)
        orders = self._build_order_rows(account=account)

        return AccountViewState(
            quote_asset=account.quote_asset,
            margin_asset=account.margin_asset,
            balance=account.balance,
            equity=account.equity,
            unrealized_pnl=account.unrealized_pnl,
            available_margin_balance=account.available_margin_balance,
            initial_margin_requirement=account.initial_margin_requirement,
            maintenance_margin_requirement=account.maintenance_margin_requirement,
            wallets=wallets,
            positions=positions,
            orders=orders,
        )

    def _build_wallet_rows(self, *, account: Account) -> list[WalletView]:
        rows: list[WalletView] = []
        for _, wallet in sorted(account.wallets.items(), key=lambda item: item[0]):
            rows.append(
                WalletView(
                    asset=wallet.asset,
                    balance=wallet.balance,
                    equity=wallet.equity,
                    available_margin_balance=wallet.available_margin_balance,
                    unrealized_pnl=wallet.unrealized_pnl,
                )
            )
        return rows[: self._config.max_wallet_rows]

    def _build_position_rows(self, *, account: Account) -> list[PositionView]:
        rows: list[PositionView] = []
        for symbol, positions in sorted(account.positions.items(), key=lambda item: item[0]):
            for position in positions:
                if position.quantity == 0:
                    continue
                rows.append(
                    PositionView(
                        symbol=symbol,
                        mode=position.mode.value,
                        side=position.side.value if position.side is not None else None,
                        quantity=position.quantity,
                        entry_price=position.entry_price,
                    )
                )

        rows.sort(key=lambda row: (row.symbol, row.side or ""))
        return rows[: self._config.max_position_rows]

    def _build_order_rows(self, *, account: Account) -> list[OrderView]:
        rows: list[OrderView] = []
        for symbol, orders in sorted(account.orders.items(), key=lambda item: item[0]):
            for order in orders:
                rows.append(
                    OrderView(
                        id=str(order.id),
                        client_id=order.client_id,
                        symbol=symbol,
                        type=order.type.value,
                        side=order.side.value,
                        status=order.status.value,
                        time_in_force=order.time_in_force.value,
                        price=order.price,
                        quantity=order.quantity,
                        filled_quantity=order.filled_quantity,
                        average_fill_price=order.average_fill_price,
                        timestamp=order.timestamp,
                    )
                )

        rows.sort(key=lambda row: row.timestamp, reverse=True)
        return rows[: self._config.max_order_rows]
