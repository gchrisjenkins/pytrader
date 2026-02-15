from __future__ import annotations

import dataclasses
from datetime import timedelta
from decimal import Decimal
from typing import Any, Mapping


def _to_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _enum_to_str(value: Any) -> str | None:
    if value is None:
        return None
    enum_value = getattr(value, "value", None)
    if enum_value is not None:
        return str(enum_value)
    return str(value)


@dataclasses.dataclass
class Ohlc:
    time: int
    open: Decimal
    high: Decimal | None = None
    low: Decimal | None = None
    close: Decimal | None = None

    def __post_init__(self) -> None:
        if self.high is None:
            self.high = self.open
        if self.low is None:
            self.low = self.open
        if self.close is None:
            self.close = self.open


@dataclasses.dataclass
class StrategyIndicator:
    name: str
    value: float | None
    is_price_indicator: bool = False


@dataclasses.dataclass
class MarketViewState:
    symbol: str
    status: str | None = None
    last_price: Decimal | None = None
    last_quantity: Decimal | None = None
    best_bid: Decimal | None = None
    best_bid_quantity: Decimal | None = None
    best_ask: Decimal | None = None
    best_ask_quantity: Decimal | None = None
    mark_price: Decimal | None = None
    index_price: Decimal | None = None
    funding_rate: Decimal | None = None
    next_funding_time: int | None = None
    open_interest: Decimal | None = None
    open_interest_value: Decimal | None = None
    last_liquidation_side: str | None = None
    last_liquidation_price: Decimal | None = None
    last_liquidation_quantity: Decimal | None = None
    event_ts_ms: int | None = None
    recv_ts_ms: int | None = None
    sequence: int | None = None


@dataclasses.dataclass
class WalletView:
    asset: str
    balance: Decimal
    equity: Decimal
    available_margin_balance: Decimal
    unrealized_pnl: Decimal


@dataclasses.dataclass
class PositionView:
    symbol: str
    mode: str
    side: str | None
    quantity: Decimal
    entry_price: Decimal


@dataclasses.dataclass
class OrderView:
    id: str
    client_id: str | None
    symbol: str
    type: str
    side: str
    status: str
    time_in_force: str
    price: Decimal
    quantity: Decimal
    filled_quantity: Decimal
    average_fill_price: Decimal | None
    timestamp: int


@dataclasses.dataclass
class AccountViewState:
    quote_asset: str
    margin_asset: str
    balance: Decimal
    equity: Decimal
    unrealized_pnl: Decimal
    available_margin_balance: Decimal
    initial_margin_requirement: Decimal
    maintenance_margin_requirement: Decimal
    wallets: list[WalletView] = dataclasses.field(default_factory=list)
    positions: list[PositionView] = dataclasses.field(default_factory=list)
    orders: list[OrderView] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class StrategyViewState:
    state: dict[str, Any] = dataclasses.field(default_factory=dict)
    action: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class TraderViewFrame:
    symbol: str
    time: timedelta
    ohlc: Ohlc | None = None
    indicators: list[StrategyIndicator] = dataclasses.field(default_factory=list)
    market: MarketViewState | None = None
    account: AccountViewState | None = None
    strategy: StrategyViewState | None = None
    event_ts_ms: int | None = None
    sequence: int | None = None


@dataclasses.dataclass
class TraderViewDataUpdate:
    symbol: str
    time: timedelta
    frame: TraderViewFrame | None = None
    reset_history: bool = False
    seed_points: list[tuple[int, Ohlc]] = dataclasses.field(default_factory=list)
    data: dict[str, Any] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_frame(cls, frame: TraderViewFrame) -> "TraderViewDataUpdate":
        return cls(symbol=frame.symbol, time=frame.time, frame=frame)

    def resolve_frame(self) -> TraderViewFrame:
        if self.frame is not None:
            return self.frame
        return _build_frame_from_legacy_payload(self)


def _build_frame_from_legacy_payload(update: TraderViewDataUpdate) -> TraderViewFrame:
    payload = update.data if isinstance(update.data, dict) else {}

    ohlc = _parse_ohlc(payload.get("ohlc"))
    indicators = _parse_indicators(payload.get("indicators"))
    market = _parse_market(payload.get("market"))
    strategy = _parse_strategy(payload)
    event_ts_ms = _to_int(payload.get("event_ts_ms"))
    sequence = _to_int(payload.get("sequence"))

    return TraderViewFrame(
        symbol=update.symbol,
        time=update.time,
        ohlc=ohlc,
        indicators=indicators,
        market=market,
        strategy=strategy,
        event_ts_ms=event_ts_ms,
        sequence=sequence,
    )


def _parse_ohlc(value: Any) -> Ohlc | None:
    if not isinstance(value, Mapping):
        return None

    if "time" not in value or "open" not in value:
        return None

    open_price = _to_decimal(value["open"])
    if open_price is None:
        return None

    return Ohlc(
        time=int(value["time"]),
        open=open_price,
        high=_to_decimal(value.get("high")),
        low=_to_decimal(value.get("low")),
        close=_to_decimal(value.get("close")),
    )


def _parse_indicators(value: Any) -> list[StrategyIndicator]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("'indicators' payload must be a list")

    indicators: list[StrategyIndicator] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue

        name = str(item.get("name", "")).strip()
        if not name:
            continue

        indicator_value = item.get("value")
        indicators.append(
            StrategyIndicator(
                name=name,
                value=None if indicator_value is None else float(indicator_value),
                is_price_indicator=bool(item.get("is_price_indicator", False)),
            )
        )
    return indicators


def _parse_market(value: Any) -> MarketViewState | None:
    if not isinstance(value, Mapping):
        return None

    symbol_raw = value.get("symbol")
    if not isinstance(symbol_raw, str) or not symbol_raw.strip():
        return None

    return MarketViewState(
        symbol=symbol_raw.strip().upper(),
        status=_enum_to_str(value.get("status")),
        last_price=_to_decimal(value.get("last_price")),
        last_quantity=_to_decimal(value.get("last_quantity")),
        best_bid=_to_decimal(value.get("best_bid")),
        best_bid_quantity=_to_decimal(value.get("best_bid_quantity")),
        best_ask=_to_decimal(value.get("best_ask")),
        best_ask_quantity=_to_decimal(value.get("best_ask_quantity")),
        mark_price=_to_decimal(value.get("mark_price")),
        index_price=_to_decimal(value.get("index_price")),
        funding_rate=_to_decimal(value.get("funding_rate")),
        next_funding_time=_to_int(value.get("next_funding_time")),
        open_interest=_to_decimal(value.get("open_interest")),
        open_interest_value=_to_decimal(value.get("open_interest_value")),
        last_liquidation_side=_enum_to_str(value.get("last_liquidation_side")),
        last_liquidation_price=_to_decimal(value.get("last_liquidation_price")),
        last_liquidation_quantity=_to_decimal(value.get("last_liquidation_quantity")),
        event_ts_ms=_to_int(value.get("last_market_data_event_ts_ms")),
        recv_ts_ms=_to_int(value.get("last_market_data_recv_ts_ms")),
        sequence=_to_int(value.get("last_market_data_sequence")),
    )


def _parse_strategy(value: Any) -> StrategyViewState | None:
    if not isinstance(value, Mapping):
        return None

    state = value.get("state")
    action = value.get("action")

    if not isinstance(state, dict) and not isinstance(action, dict):
        return None

    return StrategyViewState(
        state=state if isinstance(state, dict) else {},
        action=action if isinstance(action, dict) else {},
    )
