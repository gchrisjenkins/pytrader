import copy
import queue
import threading
from abc import abstractmethod, ABC
from datetime import timedelta, datetime
from decimal import Decimal, InvalidOperation
from decimal import ROUND_HALF_UP
from enum import Enum
from enum import unique
from typing import TypeVar, Generic, Any, Self, Literal, Iterable

import logging

import pydantic.dataclasses as dataclasses
from pydantic import BaseModel, field_validator, ConfigDict, Field, ValidationError, \
    PrivateAttr
from pydantic_core.core_schema import ValidationInfo
from pydantic_settings import BaseSettings, SettingsConfigDict

from pytrader.util import PydanticModel


def convert_to_decimal(value: str | int | float | Decimal) -> Decimal:

    try:
        if value is None:
            raise ValueError("'value' is required")
        if isinstance(value, Decimal):
            return value
        if isinstance(value, float):
            return Decimal(str(value))
        if isinstance(value, (int, str)):
            return Decimal(value)
    except (InvalidOperation, ValueError) as e:
        raise ValueError(f"Invalid value {value} for conversion to decimal.Decimal") from e

    raise TypeError(f"Unsupported type {type(value)} for conversion to decimal.Decimal (value={value})")


class TimeUnit(Enum):

    MICROSECOND = timedelta(microseconds=1)
    MILLISECOND = timedelta(milliseconds=1)
    SECOND = timedelta(seconds=1)
    MINUTE = timedelta(minutes=1)
    HOUR = timedelta(hours=1)
    DAY = timedelta(days=1)
    WEEK = timedelta(weeks=1)

    @property
    def seconds(self) -> float:
        return self.value.total_seconds()

    @property
    def full_name(self) -> str:
        """Return the full unit name."""
        return {
            TimeUnit.MICROSECOND: 'microsecond',
            TimeUnit.MILLISECOND: 'millisecond',
            TimeUnit.SECOND:      'second',
            TimeUnit.MINUTE:      'minute',
            TimeUnit.HOUR:        'hour',
            TimeUnit.DAY:         'day',
            TimeUnit.WEEK:        'week',
        }[self]

    @property
    def abbreviation(self) -> str:
        """Return the abbreviated unit name."""
        return {
            TimeUnit.MICROSECOND: '\u03BCs',
            TimeUnit.MILLISECOND: 'ms',
            TimeUnit.SECOND:      's',
            TimeUnit.MINUTE:      'min',
            TimeUnit.HOUR:        'h',
            TimeUnit.DAY:         'd',
            TimeUnit.WEEK:        'wk',
        }[self]

    def __str__(self) -> str:
        """Default string representation uses the full name."""
        return self.full_name


@dataclasses.dataclass(frozen=True, slots=True)
class Duration:

    value: float
    time_unit: TimeUnit

    @property
    def seconds(self) -> float:
        return self.value * self.time_unit.seconds


@dataclasses.dataclass
class Order:

    @unique
    class Type(Enum):
        MARKET = "market"
        LIMIT = "limit"
        STOP = "stop"
        STOP_MARKET = "stop_market"
        TAKE_PROFIT = "take_profit"
        TAKE_PROFIT_MARKET = "take_profit_market"
        TRAILING_STOP_MARKET = "trailing_stop_market"

    @unique
    class Side(Enum):
        BUY = "buy"
        SELL = "sell"

    @unique
    class Status(Enum):
        NEW = "new"
        FILLED = "filled"
        PARTIALLY_FILLED = "partially_filled"
        CANCELED = "canceled"
        REJECTED = "rejected"
        EXPIRED = "expired"

    @unique
    class TimeInForce(Enum):
        GTC = "good_till_canceled"
        IOC = "immediate_or_cancel"
        FOK = "fill_or_kill"
        GTX = "good_till_crossing"
        RPI = "retail_price_improvement"
        HIDDEN = "hidden"

    id: str
    client_id: str | None
    symbol: str
    type: Type
    time_in_force: TimeInForce
    side: Side
    price: Decimal
    quantity: Decimal
    # Order creation time (epoch milliseconds)
    timestamp: int
    status: Status = Status.NEW
    filled_quantity: Decimal = Decimal("0")
    average_fill_price: Decimal | None = None


@dataclasses.dataclass
class Position:

    @unique
    class Mode(Enum):
        NET = "net"
        HEDGE = "hedge"

    @unique
    class Side(Enum):
        LONG = "long"
        SHORT = "short"

    symbol: str
    mode: Mode = Mode.NET
    quantity: Decimal = Decimal("0.0")
    entry_price: Decimal = Decimal("0.0")
    side: Side | None = None


# @dataclasses.dataclass(frozen=True, slots=True)
# class MarketSnapshot:
#
#     symbol: str
#     timestamp: int
#
#     last_price: Decimal
#     last_quantity: Decimal
#
#     best_bid: Decimal
#     best_bid_quantity: Decimal
#     best_ask: Decimal
#     best_ask_quantity: Decimal
#
#     mark_price: Decimal
#     index_price: Decimal
#     funding_rate: Decimal
#     next_funding_time: int
#
#     @property
#     def mid_price(self) -> Decimal:
#         return (self.best_bid + self.best_ask) / Decimal("2")
#
#     @property
#     def spread(self) -> Decimal:
#         return self.best_ask - self.best_bid


class PriceTicker(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    timestamp: int

    best_bid: Decimal
    best_bid_quantity: Decimal
    best_ask: Decimal
    best_ask_quantity: Decimal

    @property
    def mid_price(self) -> Decimal:
        return (self.best_bid + self.best_ask) / Decimal("2")

    @property
    def spread(self) -> Decimal:
        return self.best_ask - self.best_bid


class TopOfBookTicker(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    timestamp: int

    last_price: Decimal
    last_quantity: Decimal


class MarketDataType(str, Enum):
    INSTRUMENT = "instrument"
    BOOK_TOP = "book_top"
    TRADE = "trade"
    MARK_FUNDING = "mark_funding"
    MARKET_STATUS = "market_status"
    BOOK_SNAPSHOT = "book_snapshot"
    BOOK_DELTA = "book_delta"
    OPEN_INTEREST = "open_interest"
    LIQUIDATION = "liquidation"
    KLINE = "kline"


class MarketDataSource(str, Enum):
    WEBSOCKET = "websocket"
    REST = "rest"
    REPLAY = "replay"
    SIMULATION = "simulation"
    OTHER = "other"


class ContractType(str, Enum):
    PERPETUAL = "perpetual"
    FUTURE = "future"
    UNKNOWN = "unknown"


class MarketTradingStatus(str, Enum):
    TRADING = "trading"
    HALTED = "halted"
    CLOSE_ONLY = "close_only"
    POST_ONLY = "post_only"
    SETTLEMENT = "settlement"
    DELISTED = "delisted"


class InstrumentPayload(PydanticModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str
    base_asset: str
    quote_asset: str
    margin_asset: str
    contract_type: ContractType
    tick_size: Decimal
    quantity_step: Decimal
    min_quantity: Decimal | None = None
    max_quantity: Decimal | None = None
    min_notional: Decimal | None = None
    status: MarketTradingStatus = MarketTradingStatus.TRADING


class BookTopPayload(PydanticModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str
    best_bid: Decimal
    best_bid_quantity: Decimal
    best_ask: Decimal
    best_ask_quantity: Decimal


class TradePayload(PydanticModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str
    price: Decimal
    quantity: Decimal
    aggressor_side: Order.Side | None = None


class MarkFundingPayload(PydanticModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str
    mark_price: Decimal
    index_price: Decimal | None = None
    funding_rate: Decimal | None = None
    next_funding_time: int | None = None


class MarketStatusPayload(PydanticModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str
    status: MarketTradingStatus
    reason: str | None = None


class BookLevelPayload(PydanticModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    price: Decimal
    quantity: Decimal


class BookSnapshotPayload(PydanticModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str
    bids: list[BookLevelPayload]
    asks: list[BookLevelPayload]


class BookDeltaPayload(PydanticModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str
    bids: list[BookLevelPayload] = Field(default_factory=list)
    asks: list[BookLevelPayload] = Field(default_factory=list)


class OpenInterestPayload(PydanticModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str
    open_interest: Decimal
    open_interest_value: Decimal | None = None


class LiquidationPayload(PydanticModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str
    side: Order.Side
    price: Decimal
    quantity: Decimal


class MarketDataMessage(PydanticModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=1, ge=1)
    provider: str
    type: MarketDataType
    symbol: str
    event_ts_ms: int = Field(ge=0)
    recv_ts_ms: int = Field(ge=0)
    sequence: int | None = Field(default=None, ge=0)
    source: MarketDataSource = MarketDataSource.WEBSOCKET
    payload: dict[str, Any]
    raw: dict[str, Any] | None = None

    @field_validator("provider", "symbol", mode="before")
    @classmethod
    def _transform_required_non_empty_string(cls, value: Any, info: ValidationInfo) -> str:
        field_name = info.field_name

        if value is None:
            raise ValueError(f"'{field_name}' is required")

        if not isinstance(value, str):
            raise TypeError(f"'{field_name}' must be of str type, but found: {type(value)}")

        transformed = value.strip()
        if not transformed:
            raise ValueError(f"'{field_name}' cannot be empty")

        return transformed.upper() if field_name == "symbol" else transformed


class InstrumentMarketDataMessage(MarketDataMessage):
    type: Literal[MarketDataType.INSTRUMENT] = MarketDataType.INSTRUMENT
    payload: InstrumentPayload


class BookTopMarketDataMessage(MarketDataMessage):
    type: Literal[MarketDataType.BOOK_TOP] = MarketDataType.BOOK_TOP
    payload: BookTopPayload


class TradeMarketDataMessage(MarketDataMessage):
    type: Literal[MarketDataType.TRADE] = MarketDataType.TRADE
    payload: TradePayload


class MarkFundingMarketDataMessage(MarketDataMessage):
    type: Literal[MarketDataType.MARK_FUNDING] = MarketDataType.MARK_FUNDING
    payload: MarkFundingPayload


class MarketStatusMarketDataMessage(MarketDataMessage):
    type: Literal[MarketDataType.MARKET_STATUS] = MarketDataType.MARKET_STATUS
    payload: MarketStatusPayload


class BookSnapshotMarketDataMessage(MarketDataMessage):
    type: Literal[MarketDataType.BOOK_SNAPSHOT] = MarketDataType.BOOK_SNAPSHOT
    payload: BookSnapshotPayload


class BookDeltaMarketDataMessage(MarketDataMessage):
    type: Literal[MarketDataType.BOOK_DELTA] = MarketDataType.BOOK_DELTA
    payload: BookDeltaPayload


class OpenInterestMarketDataMessage(MarketDataMessage):
    type: Literal[MarketDataType.OPEN_INTEREST] = MarketDataType.OPEN_INTEREST
    payload: OpenInterestPayload


class LiquidationMarketDataMessage(MarketDataMessage):
    type: Literal[MarketDataType.LIQUIDATION] = MarketDataType.LIQUIDATION
    payload: LiquidationPayload


NormalizedMarketDataMessage = (
    InstrumentMarketDataMessage
    | BookTopMarketDataMessage
    | TradeMarketDataMessage
    | MarkFundingMarketDataMessage
    | MarketStatusMarketDataMessage
    | BookSnapshotMarketDataMessage
    | BookDeltaMarketDataMessage
    | OpenInterestMarketDataMessage
    | LiquidationMarketDataMessage
)

REQUIRED_FUTURES_MARKET_DATA_TYPES: set[MarketDataType] = {
    MarketDataType.INSTRUMENT,
    MarketDataType.BOOK_TOP,
    MarketDataType.TRADE,
    MarketDataType.MARK_FUNDING,
    MarketDataType.MARKET_STATUS,
}

EXTENDED_FUTURES_MARKET_DATA_TYPES: set[MarketDataType] = {
    MarketDataType.BOOK_SNAPSHOT,
    MarketDataType.BOOK_DELTA,
    MarketDataType.OPEN_INTEREST,
    MarketDataType.LIQUIDATION,
}

SUPPORTED_FUTURES_MARKET_DATA_TYPES: set[MarketDataType] = (
    REQUIRED_FUTURES_MARKET_DATA_TYPES | EXTENDED_FUTURES_MARKET_DATA_TYPES
)


MarketSettingsType = TypeVar("MarketSettingsType", bound="Market.Settings")


class Market(Generic[MarketSettingsType]):

    def __init__(self, settings: MarketSettingsType):

        cls = type(self)
        self._logger = logging.getLogger(f"{cls.__module__}.{cls.__name__}")

        self._set_settings(settings)

        self._last_price: Decimal | None = None
        self._last_quantity: Decimal | None = None

        self._best_bid: Decimal | None = None
        self._best_bid_quantity: Decimal | None = None

        self._best_ask: Decimal | None = None
        self._best_ask_quantity: Decimal | None = None

        self._mark_price: Decimal | None = None
        self._index_price: Decimal | None = None
        self._funding_rate: Decimal | None = None
        self._next_funding_time: int | None = None
        self._market_status: MarketTradingStatus | None = None
        self._open_interest: Decimal | None = None
        self._open_interest_value: Decimal | None = None
        self._last_liquidation_side: Order.Side | None = None
        self._last_liquidation_price: Decimal | None = None
        self._last_liquidation_quantity: Decimal | None = None
        self._book_bids: dict[Decimal, Decimal] = {}
        self._book_asks: dict[Decimal, Decimal] = {}
        self._last_market_data_event_ts_ms: int | None = None
        self._last_market_data_recv_ts_ms: int | None = None
        self._last_market_data_sequence: int | None = None
        self._market_data_input_queue: queue.Queue[MarketDataMessage | NormalizedMarketDataMessage] = queue.Queue()
        self._market_data_worker_stop_event = threading.Event()
        self._market_data_worker_thread: threading.Thread | None = None
        self._market_data_worker_poll_timeout_sec: float = 0.25
        self._market_data_drop_oldest_on_full: bool = True
        self._critical_market_data_types: set[MarketDataType] = {
            MarketDataType.INSTRUMENT,
            MarketDataType.MARKET_STATUS,
            MarketDataType.LIQUIDATION,
            MarketDataType.MARK_FUNDING,
        }
        self._dropped_market_data_messages: int = 0

    def round_price(self, price: str | int | float | Decimal) -> Decimal:
        return (convert_to_decimal(price) / self.settings.tick_size).quantize(
            Decimal('1'), rounding=ROUND_HALF_UP
        ) * self.settings.tick_size

    def round_quantity(self, quantity: str | int | float | Decimal) -> Decimal:
        return (convert_to_decimal(quantity) / self.settings.order_increment).quantize(
            Decimal('1'), rounding=ROUND_HALF_UP
        ) * self.settings.order_increment

    @property
    def settings(self) -> MarketSettingsType:
        return self._settings

    def _set_settings(self, value: MarketSettingsType):

        if value is None:
            raise ValueError("'settings' is required")

        if not isinstance(value, Market.Settings):
            raise TypeError(f"'settings' must be of Market.Settings type, but found: {type(value)}")

        self._settings = value

    @property
    def last_price(self) -> Decimal | None:
        return self._last_price

    def _set_last_price(self, value: str | int | float | Decimal):

        if value is None:
            raise ValueError("'last_price' is required")

        if not isinstance(value, (str, int, float, Decimal)):
            raise TypeError(f"'last_price' must be of str, int, float, or Decimal type, but found: {type(value)}")

        rounded_price = self.round_price(value)
        if rounded_price <= 0:
            raise ValueError(
                f"[{self._settings.symbol}] 'last_price' ({value}) rounded to zero or less with 'tick_size' "
                f"({self.settings.tick_size})"
            )

        self._last_price = rounded_price

    @property
    def last_quantity(self) -> Decimal | None:
        return self._last_quantity

    def _set_last_quantity(self, value: str | int | float | Decimal):

        if value is None:
            raise ValueError("'last_quantity' is required")

        if not isinstance(value, (str, int, float, Decimal)):
            raise TypeError(f"'last_quantity' must be of str, int, float, or Decimal type, but found: {type(value)}")

        rounded_quantity = self.round_quantity(value)
        if rounded_quantity < 0:
            raise ValueError(
                f"[{self._settings.symbol}] 'last_quantity' ({value}) rounded to less than zero with 'order_increment' "
                f"({self.settings.order_increment})"
            )

        self._last_quantity = rounded_quantity

    @property
    def best_bid(self) -> Decimal | None:
        return self._best_bid

    def _set_best_bid(self, value: str | int | float | Decimal):

        if value is None:
            raise ValueError("'best_bid' is required")

        if not isinstance(value, (str, int, float, Decimal)):
            raise TypeError(f"'best_bid' must be of str, int, float, or Decimal type, but found: {type(value)}")

        rounded_price = self.round_price(value)
        if rounded_price <= 0:
            raise ValueError(
                f"[{self._settings.symbol}] 'best_bid' ({value}) rounded to zero or less with 'tick_size' "
                f"({self.settings.tick_size})"
            )

        self._best_bid = rounded_price

    @property
    def best_bid_quantity(self) -> Decimal | None:
        return self._best_bid_quantity

    def _set_best_bid_quantity(self, value: str | int | float | Decimal):

        if value is None:
            raise ValueError("'best_bid_quantity' is required")

        if not isinstance(value, (str, int, float, Decimal)):
            raise TypeError(f"'best_bid_quantity' must be of str, int, float, or Decimal type, but found: {type(value)}")

        rounded_quantity = self.round_quantity(value)
        if rounded_quantity < 0:
            raise ValueError(
                f"[{self._settings.symbol}] 'best_bid_quantity' ({value}) rounded to less than zero with "
                f"'order_increment' ({self.settings.order_increment})"
            )

        self._best_bid_quantity = rounded_quantity

    @property
    def best_ask(self) -> Decimal | None:
        return self._best_ask

    def _set_best_ask(self, value: str | int | float | Decimal):

        if value is None:
            raise ValueError("'best_ask' is required")

        if not isinstance(value, (str, int, float, Decimal)):
            raise TypeError(f"'best_ask' must be of str, int, float, or Decimal type, but found: {type(value)}")

        rounded_price = self.round_price(value)
        if rounded_price <= 0:
            raise ValueError(
                f"[{self._settings.symbol}] 'best_ask' ({value}) rounded to zero or less with 'tick_size' "
                f"({self.settings.tick_size})"
            )

        self._best_ask = rounded_price

    @property
    def best_ask_quantity(self) -> Decimal | None:
        return self._best_ask_quantity

    def _set_best_ask_quantity(self, value: str | int | float | Decimal):

        if value is None:
            raise ValueError("'best_ask_quantity' is required")

        if not isinstance(value, (str, int, float, Decimal)):
            raise TypeError(f"'best_ask_quantity' must be of str, int, float, or Decimal type, but found: {type(value)}")

        rounded_quantity = self.round_quantity(value)
        if rounded_quantity < 0:
            raise ValueError(
                f"[{self._settings.symbol}] 'best_ask_quantity' ({value}) rounded to less than zero with "
                f"'order_increment' ({self.settings.order_increment})"
            )

        self._best_ask_quantity = rounded_quantity

    @property
    def mark_price(self) -> Decimal | None:
        return self._mark_price

    def _set_mark_price(self, value: str | int | float | Decimal):

        if value is None:
            raise ValueError("'mark_price' is required")

        if not isinstance(value, (str, int, float, Decimal)):
            raise TypeError(f"'mark_price' must be of str, int, float, or Decimal type, but found: {type(value)}")

        rounded_price = self.round_price(value)
        if rounded_price <= 0:
            raise ValueError(
                f"[{self.settings.symbol}] 'mark_price' ({value}) rounded to zero or less with 'tick_size' "
                f"({self.settings.tick_size})"
            )

        self._mark_price = rounded_price

    @property
    def index_price(self) -> Decimal | None:
        return self._index_price

    def _set_index_price(self, value: str | int | float | Decimal):

        if value is None:
            raise ValueError("'index_price' is required")

        if not isinstance(value, (str, int, float, Decimal)):
            raise TypeError(f"'index_price' must be of str, int, float, or Decimal type, but found: {type(value)}")

        rounded_price = self.round_price(value)
        if rounded_price <= 0:
            raise ValueError(
                f"[{self.settings.symbol}] 'index_price' ({value}) rounded to zero or less with 'tick_size' "
                f"({self.settings.tick_size})"
            )

        self._index_price = rounded_price

    @property
    def funding_rate(self) -> Decimal | None:
        return self._funding_rate

    def _set_funding_rate(self, value: str | int | float | Decimal):

        if value is None:
            raise ValueError("'funding_rate' is required")

        if not isinstance(value, (str, int, float, Decimal)):
            raise TypeError(f"'funding_rate' must be of str, int, float, or Decimal type, but found: {type(value)}")

        self._funding_rate = convert_to_decimal(value)

    @property
    def next_funding_time(self) -> int | None:
        return self._next_funding_time

    def _set_next_funding_time(self, value: int):

        if value is None:
            raise ValueError("'next_funding_time' is required")

        if not isinstance(value, int):
            raise TypeError(f"'next_funding_time' must be of int type, but found: {type(value)}")

        self._next_funding_time = value

    @property
    def market_status(self) -> MarketTradingStatus | None:
        return self._market_status

    @property
    def open_interest(self) -> Decimal | None:
        return self._open_interest

    @property
    def open_interest_value(self) -> Decimal | None:
        return self._open_interest_value

    @property
    def last_liquidation_side(self) -> Order.Side | None:
        return self._last_liquidation_side

    @property
    def last_liquidation_price(self) -> Decimal | None:
        return self._last_liquidation_price

    @property
    def last_liquidation_quantity(self) -> Decimal | None:
        return self._last_liquidation_quantity

    @property
    def last_market_data_event_ts_ms(self) -> int | None:
        return self._last_market_data_event_ts_ms

    @property
    def last_market_data_recv_ts_ms(self) -> int | None:
        return self._last_market_data_recv_ts_ms

    @property
    def last_market_data_sequence(self) -> int | None:
        return self._last_market_data_sequence

    def get_state_snapshot(self) -> dict[str, Any]:
        """
        Return a point-in-time view of normalized market state.
        """
        return {
            "symbol": self.settings.symbol,
            "status": self._market_status,
            "last_price": self._last_price,
            "last_quantity": self._last_quantity,
            "best_bid": self._best_bid,
            "best_bid_quantity": self._best_bid_quantity,
            "best_ask": self._best_ask,
            "best_ask_quantity": self._best_ask_quantity,
            "mark_price": self._mark_price,
            "index_price": self._index_price,
            "funding_rate": self._funding_rate,
            "next_funding_time": self._next_funding_time,
            "open_interest": self._open_interest,
            "open_interest_value": self._open_interest_value,
            "last_liquidation_side": self._last_liquidation_side,
            "last_liquidation_price": self._last_liquidation_price,
            "last_liquidation_quantity": self._last_liquidation_quantity,
            "last_market_data_event_ts_ms": self._last_market_data_event_ts_ms,
            "last_market_data_recv_ts_ms": self._last_market_data_recv_ts_ms,
            "last_market_data_sequence": self._last_market_data_sequence,
        }

    def is_market_data_worker_running(self) -> bool:
        return self._market_data_worker_thread is not None and self._market_data_worker_thread.is_alive()

    def get_pending_market_data_count(self) -> int:
        return self._market_data_input_queue.qsize()

    def get_dropped_market_data_count(self) -> int:
        return self._dropped_market_data_messages

    def configure_market_data_queue(
        self,
        *,
        maxsize: int = 0,
        drop_oldest_on_full: bool = True,
        critical_types: set[MarketDataType] | None = None,
    ) -> None:
        if maxsize < 0:
            raise ValueError("'maxsize' must be >= 0")

        if self._market_data_input_queue.qsize() > 0 and maxsize != self._market_data_input_queue.maxsize:
            raise RuntimeError("Cannot resize market queue while it is non-empty")

        if maxsize != self._market_data_input_queue.maxsize:
            self._market_data_input_queue = queue.Queue(maxsize=maxsize)

        self._market_data_drop_oldest_on_full = bool(drop_oldest_on_full)
        self._critical_market_data_types = (
            set(critical_types)
            if critical_types is not None
            else {
                MarketDataType.INSTRUMENT,
                MarketDataType.MARKET_STATUS,
                MarketDataType.LIQUIDATION,
                MarketDataType.MARK_FUNDING,
            }
        )

    def start_market_data_worker(
        self,
        *,
        poll_timeout_sec: float = 0.25,
        thread_name: str | None = None,
    ) -> None:
        if poll_timeout_sec <= 0:
            raise ValueError("'poll_timeout_sec' must be positive")

        if self.is_market_data_worker_running():
            return

        self._market_data_worker_poll_timeout_sec = poll_timeout_sec
        self._market_data_worker_stop_event.clear()
        self._market_data_worker_thread = threading.Thread(
            target=self._market_data_worker_loop,
            name=thread_name or f"{type(self).__name__}:{self.settings.symbol}",
            daemon=True,
        )
        self._market_data_worker_thread.start()

    def stop_market_data_worker(self, *, timeout_sec: float = 5.0) -> None:
        if timeout_sec <= 0:
            raise ValueError("'timeout_sec' must be positive")

        self._market_data_worker_stop_event.set()
        if self._market_data_worker_thread and self._market_data_worker_thread.is_alive():
            self._market_data_worker_thread.join(timeout=timeout_sec)
            if self._market_data_worker_thread.is_alive():
                self._logger.warning(
                    "[%s] market-data worker did not stop within %.2f sec",
                    self.settings.symbol,
                    timeout_sec,
                )
        self._market_data_worker_thread = None

    def enqueue_market_data_message(
        self,
        message: MarketDataMessage | NormalizedMarketDataMessage,
        *,
        block: bool = True,
        timeout_sec: float | None = None,
    ) -> None:
        self.validate_market_data_message(message)

        if timeout_sec is not None and timeout_sec < 0:
            raise ValueError("'timeout_sec' must be non-negative")
        if not self._market_data_drop_oldest_on_full:
            self._market_data_input_queue.put(message, block=block, timeout=timeout_sec)
            return

        try:
            self._market_data_input_queue.put(message, block=False)
            return
        except queue.Full:
            pass

        incoming_is_critical = message.type in self._critical_market_data_types
        oldest = self._peek_oldest_market_data_message()
        oldest_is_critical = oldest is not None and oldest.type in self._critical_market_data_types

        if not incoming_is_critical and oldest_is_critical:
            self._dropped_market_data_messages += 1
            return

        while True:
            try:
                _dropped = self._market_data_input_queue.get_nowait()
                self._market_data_input_queue.task_done()
                self._dropped_market_data_messages += 1
            except queue.Empty:
                pass

            try:
                self._market_data_input_queue.put(message, block=False)
                return
            except queue.Full:
                continue

    def get_supported_market_data_types(self) -> set[MarketDataType]:
        """
        Contract method for providers to advertise the normalized market-data types
        this market can consume.
        """
        return set(SUPPORTED_FUTURES_MARKET_DATA_TYPES)

    def validate_market_data_message(self, message: MarketDataMessage | NormalizedMarketDataMessage):
        if message is None:
            raise ValueError("'message' is required")

        if message.symbol.upper() != self.settings.symbol:
            raise ValueError(
                f"Market-data symbol mismatch. message={message.symbol} market={self.settings.symbol}"
            )

        if message.type not in self.get_supported_market_data_types():
            raise ValueError(
                f"Unsupported market-data type '{message.type.value}' for market '{self.settings.symbol}'"
            )

    def apply_market_data_message(self, message: MarketDataMessage | NormalizedMarketDataMessage):
        """
        Provider-agnostic default router from normalized messages to internal market state.
        Providers can override for custom behavior while keeping this signature stable.
        """
        self.validate_market_data_message(message)

        # Keep metadata monotonic so older packets (e.g., REST polled OI) do not
        # move the observable market clock backwards.
        if (
            self._last_market_data_event_ts_ms is None
            or message.event_ts_ms >= self._last_market_data_event_ts_ms
        ):
            self._last_market_data_event_ts_ms = message.event_ts_ms

        if (
            self._last_market_data_recv_ts_ms is None
            or message.recv_ts_ms >= self._last_market_data_recv_ts_ms
        ):
            self._last_market_data_recv_ts_ms = message.recv_ts_ms

        if message.sequence is not None:
            if (
                self._last_market_data_sequence is None
                or message.sequence >= self._last_market_data_sequence
            ):
                self._last_market_data_sequence = message.sequence

        if message.type == MarketDataType.BOOK_TOP:
            payload = BookTopPayload.model_validate(message.payload)
            self._set_best_bid(payload.best_bid)
            self._set_best_bid_quantity(payload.best_bid_quantity)
            self._set_best_ask(payload.best_ask)
            self._set_best_ask_quantity(payload.best_ask_quantity)
            return

        if message.type == MarketDataType.BOOK_SNAPSHOT:
            payload = BookSnapshotPayload.model_validate(message.payload)
            self._book_bids = {level.price: level.quantity for level in payload.bids if level.quantity > 0}
            self._book_asks = {level.price: level.quantity for level in payload.asks if level.quantity > 0}
            self._refresh_top_of_book_from_depth()
            return

        if message.type == MarketDataType.BOOK_DELTA:
            payload = BookDeltaPayload.model_validate(message.payload)
            self._apply_depth_delta(self._book_bids, payload.bids)
            self._apply_depth_delta(self._book_asks, payload.asks)
            self._refresh_top_of_book_from_depth()
            return

        if message.type == MarketDataType.TRADE:
            payload = TradePayload.model_validate(message.payload)
            self._set_last_price(payload.price)
            self._set_last_quantity(payload.quantity)
            return

        if message.type == MarketDataType.MARK_FUNDING:
            payload = MarkFundingPayload.model_validate(message.payload)
            self._set_mark_price(payload.mark_price)
            if payload.index_price is not None:
                self._set_index_price(payload.index_price)
            if payload.funding_rate is not None:
                self._set_funding_rate(payload.funding_rate)
            if payload.next_funding_time is not None:
                self._set_next_funding_time(payload.next_funding_time)
            return

        if message.type == MarketDataType.MARKET_STATUS:
            payload = MarketStatusPayload.model_validate(message.payload)
            self._market_status = payload.status
            return

        if message.type == MarketDataType.OPEN_INTEREST:
            payload = OpenInterestPayload.model_validate(message.payload)
            self._open_interest = payload.open_interest
            self._open_interest_value = payload.open_interest_value
            return

        if message.type == MarketDataType.LIQUIDATION:
            payload = LiquidationPayload.model_validate(message.payload)
            self._last_liquidation_side = payload.side
            self._last_liquidation_price = payload.price
            self._last_liquidation_quantity = payload.quantity
            return

        if message.type == MarketDataType.INSTRUMENT:
            # Instrument updates are part of the normalized contract but are not
            # automatically applied to mutable market settings in the base class.
            return

    def _market_data_worker_loop(self):
        while True:
            if self._market_data_worker_stop_event.is_set() and self._market_data_input_queue.empty():
                return

            try:
                message = self._market_data_input_queue.get(timeout=self._market_data_worker_poll_timeout_sec)
            except queue.Empty:
                continue

            try:
                self.apply_market_data_message(message)
            except Exception:
                self._logger.exception(
                    "[%s] failed to apply queued market-data message type '%s'",
                    self.settings.symbol,
                    message.type.value,
                )
            finally:
                self._market_data_input_queue.task_done()

    def _peek_oldest_market_data_message(self) -> MarketDataMessage | NormalizedMarketDataMessage | None:
        with self._market_data_input_queue.mutex:
            if not self._market_data_input_queue.queue:
                return None
            return self._market_data_input_queue.queue[0]

    def _apply_depth_delta(self, book: dict[Decimal, Decimal], deltas: list[BookLevelPayload]):
        for level in deltas:
            if level.quantity <= 0:
                book.pop(level.price, None)
            else:
                book[level.price] = level.quantity

    def _refresh_top_of_book_from_depth(self):
        if self._book_bids:
            best_bid = max(self._book_bids)
            self._set_best_bid(best_bid)
            self._set_best_bid_quantity(self._book_bids[best_bid])

        if self._book_asks:
            best_ask = min(self._book_asks)
            self._set_best_ask(best_ask)
            self._set_best_ask_quantity(self._book_asks[best_ask])

    class Settings(BaseModel):
        model_config = ConfigDict(validate_assignment=True, extra="forbid")

        symbol: str
        base_asset: str
        quote_asset: str
        margin_asset: str
        tick_size: Decimal
        order_increment: Decimal
        maker_fee: Decimal
        taker_fee: Decimal

        @field_validator("symbol", "base_asset", "quote_asset", "margin_asset", mode="before")
        @classmethod
        def _transform_str_not_empty(cls, value: Any, info: ValidationInfo) -> str:
            field_name = info.field_name

            if value is None:
                raise ValueError(f"'{field_name}' is required")

            if not isinstance(value, str):
                raise TypeError(f"'{field_name}' must be of str type, but found: {type(value)}")

            value = value.strip().upper()
            if not value:
                raise ValueError(f"'{field_name}' cannot be empty")

            return value

        @field_validator("tick_size", "order_increment", "taker_fee", "taker_fee", mode="before")
        @classmethod
        def _transform_required_decimal(cls, value: Any, info: ValidationInfo) -> Decimal:
            field_name = info.field_name

            if value is None:
                raise ValueError(f"'{field_name}' is required")

            if not isinstance(value, (str, int, float, Decimal)):
                raise TypeError(
                    f"'{field_name}' must be one of str, int, float, or Decimal type, but found: {type(value)}"
                )

            return convert_to_decimal(value)

        @field_validator("tick_size", "order_increment", mode="after")
        @classmethod
        def _validate_positive_decimal(cls, value: Decimal, info: ValidationInfo) -> Decimal:
            field_name = info.field_name

            if value <= 0:
                raise ValueError(f"'{field_name}' must be positive, but found {value}")

            return value


WalletType = TypeVar("WalletType", bound="Account.Wallet")


class Account(PydanticModel, Generic[WalletType]):

    margin_asset: str
    quote_asset: str
    balance: Decimal
    unrealized_pnl: Decimal
    initial_margin_requirement: Decimal
    maintenance_margin_requirement: Decimal
    position_initial_margin_requirement: Decimal
    open_order_initial_margin_requirement: Decimal
    cross_margin_balance: Decimal
    cross_margin_unrealized_pnl: Decimal
    available_margin_balance: Decimal

    _wallets: dict[str, WalletType] = PrivateAttr(default_factory=dict)
    _positions: dict[str, list[Position]] = PrivateAttr(default_factory=dict)
    _orders: dict[str, list[Order]] = PrivateAttr(default_factory=dict)

    @property
    def equity(self) -> Decimal:
        return self.balance + self.unrealized_pnl

    def get_wallet(self, asset: str) -> WalletType | None:
        return self._wallets.get(asset, None)

    def add_wallet(self, wallet: WalletType):
        self._wallets[wallet.asset] = wallet

    @property
    def wallets(self) -> dict[str, WalletType]:
        return copy.deepcopy(self._wallets)

    def get_position(self, symbol: str, type_: Position.Mode) -> Position | None:
        return next((position for position in self._get_positions(symbol) if position.mode == type_), None)

    def add_position(self, position: Position):
        self._get_positions(position.symbol).append(position)

    def delete_position(self, position: Position):
        self._get_positions(position.symbol).remove(position)

    def _get_positions(self, symbol: str) -> list[Position]:
        return self._positions.setdefault(symbol, [])

    @property
    def positions(self) -> dict[str, list[Position]]:
        return copy.deepcopy(self._positions)

    def get_orders(self, symbol: str, type_: Order.Type | None, status: Order.Status | None) -> list[Order]:
        return [
            order for order in self._get_orders(symbol)
            if (type_ is None or order.type == type_) and (status is None or order.status == status)
        ]

    def add_order(self, order: Order):
        self._get_orders(order.symbol).append(order)

    def _get_orders(self, symbol: str) -> list[Order]:
        return self._orders.setdefault(symbol, [])

    @property
    def orders(self) -> dict[str, list[Order]]:
        return copy.deepcopy(self._orders)

    class Wallet(BaseModel):

        asset: str
        balance: Decimal
        unrealized_pnl: Decimal
        equity: Decimal
        initial_margin_requirement: Decimal
        maintenance_margin_requirement: Decimal
        position_initial_margin_requirement: Decimal
        open_order_initial_margin_requirement: Decimal
        cross_margin_balance: Decimal
        cross_margin_unrealized_pnl: Decimal
        available_margin_balance: Decimal
        is_marginable: bool


ExchangeConfigurationType = TypeVar("ExchangeConfigurationType", bound="Exchange.Configuration")
ExchangeCredentialsType = TypeVar("ExchangeCredentialsType", bound="Exchange.Credentials")


class Exchange(Generic[ExchangeConfigurationType, ExchangeCredentialsType], ABC):

    _registry: dict[str, type["Exchange"]] = {}

    def __init__(self, config: ExchangeConfigurationType, credentials: ExchangeCredentialsType) -> None:

        self._logger = logging.getLogger(type(self).__name__)

        self._config: ExchangeConfigurationType = config
        self._credentials: ExchangeCredentialsType = credentials

        self._account: Account | None = None
        self._markets: dict[str, Market] = {}
        self._market_data_queue: queue.Queue[MarketDataMessage | NormalizedMarketDataMessage] = queue.Queue()
        self._market_data_drop_oldest_on_full: bool = True
        self._critical_market_data_types: set[MarketDataType] = {
            MarketDataType.INSTRUMENT,
            MarketDataType.MARKET_STATUS,
            MarketDataType.LIQUIDATION,
            MarketDataType.MARK_FUNDING,
        }
        self._dropped_market_data_messages: int = 0

        self._is_running = threading.Event()
        self._is_ready = threading.Event()
        self._is_stopping = threading.Event()
        self._main_thread = None

    def start(self):

        if self._main_thread is None or not self._main_thread.is_alive():
            self._is_running.set()
            self._main_thread = threading.Thread(
                target=self._main_loop,
                name=f"{type(self).__name__}Thread",
                daemon=False
            )
            self._main_thread.start()
            self._logger.debug(f"{type(self).__name__} has started")
        else:
            self._logger.debug(f"{type(self).__name__} is already running")

    def stop(self):

        if self._main_thread and self._main_thread.is_alive():
            self._is_running.clear()
            self._is_stopping.set()
            self._main_thread.join()
            self._is_stopping.clear()
            self._logger.debug(f"{type(self).__name__} has stopped")
        else:
            self._logger.debug(f"{type(self).__name__} is not running")

    def is_running(self) -> bool:
        return self._is_running.is_set()

    def is_ready(self) -> bool:
        return self._is_ready.is_set()

    def wait_until_ready(self, timeout_sec: float | None = None) -> bool:
        if timeout_sec is not None and timeout_sec < 0:
            raise ValueError("'timeout_sec' must be non-negative")
        return self._is_ready.wait(timeout=timeout_sec)

    def is_alive(self) -> bool:
        return self._main_thread is not None and self._main_thread.is_alive()

    def get_market_symbols(self) -> list[str]:
        return list(self._markets.keys())

    def get_market_state(self, symbol: str) -> dict[str, Any] | None:
        if symbol is None:
            raise ValueError("'symbol' is required")

        normalized = symbol.strip().upper()
        if not normalized:
            raise ValueError("'symbol' cannot be empty")

        market = self._markets.get(normalized)
        if market is None:
            return None
        return market.get_state_snapshot()

    @abstractmethod
    def _startup(self):
        pass

    @abstractmethod
    def _shutdown(self):
        pass

    def _main_loop(self):
        startup_succeeded = False
        try:
            self._startup()
            startup_succeeded = True

            while self._is_running.is_set():
                if self._is_ready.is_set():

                    now = datetime.now()
                    next_snapshot_time = (now + timedelta(seconds=5)).replace(second=0, microsecond=0)
                    sleep_duration = (next_snapshot_time - now).total_seconds()
                    if sleep_duration > 0:
                        self._logger.debug(f"next snapshot in {sleep_duration} sec")
                        self._is_stopping.wait(timeout=sleep_duration)

                    if self._is_running.is_set() and not self._is_stopping.is_set():
                        try:
                            self._logger.debug("taking market snapshots...")
                        except Exception as e:
                            self._logger.error(e)
                    else:
                        self._logger.debug("exchange is stopping or no longer running")
                else:
                    self._logger.debug("exchange is not ready, continue waiting...")
                    self._is_ready.wait(30)
        except Exception:
            self._logger.exception("%s encountered a fatal runtime error", type(self).__name__)
        finally:
            try:
                self._shutdown()
            except Exception:
                if startup_succeeded:
                    self._logger.exception("%s failed during shutdown", type(self).__name__)
            self._is_ready.clear()
            self._is_running.clear()
            self._is_stopping.clear()

    def get_supported_market_data_types(self) -> set[MarketDataType]:
        """
        Contract method for exchanges to advertise the normalized market-data
        types expected by the base futures interface.
        """
        return set(SUPPORTED_FUTURES_MARKET_DATA_TYPES)

    def configure_market_data_queue(
        self,
        *,
        maxsize: int = 0,
        drop_oldest_on_full: bool = True,
        critical_types: set[MarketDataType] | None = None,
    ) -> None:
        if maxsize < 0:
            raise ValueError("'maxsize' must be >= 0")

        if self._market_data_queue.qsize() > 0 and maxsize != self._market_data_queue.maxsize:
            raise RuntimeError("Cannot resize exchange market-data queue while it is non-empty")

        if maxsize != self._market_data_queue.maxsize:
            self._market_data_queue = queue.Queue(maxsize=maxsize)

        self._market_data_drop_oldest_on_full = bool(drop_oldest_on_full)
        self._critical_market_data_types = (
            set(critical_types)
            if critical_types is not None
            else {
                MarketDataType.INSTRUMENT,
                MarketDataType.MARKET_STATUS,
                MarketDataType.LIQUIDATION,
                MarketDataType.MARK_FUNDING,
            }
        )

    def get_dropped_market_data_count(self) -> int:
        return self._dropped_market_data_messages

    def publish_market_data_message(self, message: MarketDataMessage | NormalizedMarketDataMessage):
        """
        Enqueue a normalized market-data message from any upstream source
        (websocket, REST poller, replay loader, simulator).
        """
        if message is None:
            raise ValueError("'message' is required")

        if message.type not in self.get_supported_market_data_types():
            raise ValueError(f"Unsupported market-data message type '{message.type.value}'")
        if not self._market_data_drop_oldest_on_full:
            self._market_data_queue.put(message)
            return

        try:
            self._market_data_queue.put(message, block=False)
            return
        except queue.Full:
            pass

        incoming_is_critical = message.type in self._critical_market_data_types
        oldest = self._peek_oldest_market_data_message()
        oldest_is_critical = oldest is not None and oldest.type in self._critical_market_data_types

        if not incoming_is_critical and oldest_is_critical:
            self._dropped_market_data_messages += 1
            return

        while True:
            try:
                _dropped = self._market_data_queue.get_nowait()
                self._market_data_queue.task_done()
                self._dropped_market_data_messages += 1
            except queue.Empty:
                pass

            try:
                self._market_data_queue.put(message, block=False)
                return
            except queue.Full:
                continue

    def poll_market_data_message(self, timeout_sec: float | None = None) -> MarketDataMessage | NormalizedMarketDataMessage | None:
        """
        Pull one normalized market-data message from the exchange ingress queue.
        """
        if timeout_sec is not None and timeout_sec < 0:
            raise ValueError("'timeout_sec' must be non-negative")

        try:
            return self._market_data_queue.get(timeout=timeout_sec)
        except queue.Empty:
            return None

    def mark_market_data_message_done(self) -> None:
        """
        Acknowledge completion of one message pulled via poll_market_data_message().
        """
        self._market_data_queue.task_done()

    def route_market_data_message(self, message: MarketDataMessage | NormalizedMarketDataMessage) -> bool:
        """
        Route a normalized market-data message to the correct market instance.
        Returns True when successfully routed, False if the symbol is unknown.
        """
        if message is None:
            raise ValueError("'message' is required")

        market = self._markets.get(message.symbol)
        if market is None:
            return False

        if market.is_market_data_worker_running():
            market.enqueue_market_data_message(message)
        else:
            market.apply_market_data_message(message)
        return True

    def drain_and_route_market_data_messages(
        self, max_messages: int | None = None, timeout_sec: float | None = 0.0
    ) -> int:
        """
        Consume and route queued normalized market-data messages.
        """
        if max_messages is not None and max_messages < 1:
            raise ValueError("'max_messages' must be >= 1 when provided")

        if timeout_sec is not None and timeout_sec < 0:
            raise ValueError("'timeout_sec' must be non-negative")

        routed = 0
        while max_messages is None or routed < max_messages:
            message = self.poll_market_data_message(timeout_sec=timeout_sec)
            if message is None:
                break

            self.route_market_data_message(message)
            routed += 1
            self._market_data_queue.task_done()

        return routed

    def replay_market_data_messages(
        self, messages: Iterable[MarketDataMessage | NormalizedMarketDataMessage], *, route: bool = True
    ) -> int:
        """
        Ingest normalized historical messages into the same path used for live data.
        """
        if messages is None:
            raise ValueError("'messages' is required")

        count = 0
        for message in messages:
            self.publish_market_data_message(message)
            count += 1

        if route:
            self.drain_and_route_market_data_messages(max_messages=count, timeout_sec=0.0)

        return count

    def _peek_oldest_market_data_message(self) -> MarketDataMessage | NormalizedMarketDataMessage | None:
        with self._market_data_queue.mutex:
            if not self._market_data_queue.queue:
                return None
            return self._market_data_queue.queue[0]

    @abstractmethod
    def get_account(self, quote_currency: str) -> Account:
        pass

    @abstractmethod
    def get_market_settings(self, symbol: str) -> Market.Settings | None:
        pass

    # @abstractmethod
    # def get_market_snapshot(self, symbol: str) -> MarketSnapshot | None:
    #     pass

    @abstractmethod
    def create_market_order(self,
        symbol: str, side: Order.Side, quantity: Decimal, position_side: Position.Side | None = None
    ) -> Order:
        pass

    @abstractmethod
    def create_limit_order(self,
        symbol: str, side: Order.Side, price: Decimal, quantity: Decimal, position_side: Position.Side | None = None
    ) -> Order:
        pass

    @abstractmethod
    def cancel_order(self, order: Order) -> Order:
        pass

    # @dataclasses.dataclass(config=ConfigDict(validate_assignment=True, extra="forbid"))
    class Settings(BaseModel):
        model_config = ConfigDict(validate_assignment=True, extra="forbid")

        maker_fee: Decimal
        taker_fee: Decimal

        @field_validator("maker_fee", mode="before")
        @classmethod
        def _transform_maker_fee(cls, value: Any) -> Decimal:

            if value is None:
                raise ValueError("'maker_fee' is required")

            if not isinstance(value, (str, int, float, Decimal)):
                raise TypeError(
                    f"'maker_fee' must be one of str, int, float, or Decimal type, but found: {type(value)}"
                )

            return convert_to_decimal(value)

        @field_validator("taker_fee", mode="before")
        @classmethod
        def _transform_taker_fee(cls, value: Any) -> Decimal:

            if value is None:
                raise ValueError("'taker_fee' is required")

            if not isinstance(value, (str, int, float, Decimal)):
                raise TypeError(
                    f"'taker_fee' must be one of str, int, float, or Decimal type, but found: {type(value)}"
                )

            return convert_to_decimal(value)

    class Configuration(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)

        provider: str
        markets: list[str] = Field(min_length=1, description="List of symbols to subscribe for market data")

    class Credentials(BaseSettings):
        model_config = SettingsConfigDict(extra="forbid")

    class EnvOverrides(BaseSettings):
        model_config = SettingsConfigDict()

    @classmethod
    def register_provider(cls, name: str):

        def decorator(concrete_cls: type[Self]) -> type[Self]:
            if not issubclass(concrete_cls, Exchange):
                raise TypeError(f"{concrete_cls.__name__} must subclass Exchange")

            # Enforce concrete Configuration and Credentials
            if concrete_cls.Configuration is Exchange.Configuration:
                raise TypeError(f"{concrete_cls.__name__} must define a concrete Configuration class")
            if concrete_cls.Credentials is Exchange.Credentials:
                raise TypeError(f"{concrete_cls.__name__} must define a concrete Credentials class")

            cls._registry[name] = concrete_cls
            return concrete_cls

        return decorator

    @classmethod
    def build(cls, raw_config: dict) -> "Exchange":
        """
        Factory method to create an instance from a raw configuration dictionary.
        """
        if not isinstance(raw_config, dict):
            raise TypeError(f"'raw_config' must be of dict type, but found: {type(raw_config)}")

        if "provider" not in raw_config:
            raise ValueError("Configuration must include a 'provider' field")

        provider = raw_config["provider"]

        if provider not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown Exchange provider '{provider}'. "
                f"Available providers: {available or 'none'}"
            )

        concrete_cls = cls._registry[provider]

        # Validate non-sensitive configuration (enforces Literal provider match)
        try:
            config = concrete_cls.Configuration.model_validate(raw_config)
        except ValidationError as e:
            raise ValueError(f"Invalid configuration for provider '{provider}': {e}") from e

        # Always apply env overrides (empty base does nothing; custom in concrete class does the work)
        try:
            overrides = concrete_cls.EnvOverrides()
        except ValidationError as e:
            raise ValueError(f"Invalid env overrides for provider '{provider}': {e}") from e

        override_dict = overrides.model_dump(exclude_unset=True, exclude_none=True)
        if override_dict:
            config = config.model_copy(update=override_dict)

        # Load and validate credentials from env
        try:
            credentials = concrete_cls.Credentials()
        except ValidationError as e:
            raise ValueError(f"Missing/invalid credentials for provider '{provider}': {e}") from e

        try:
            return concrete_cls(config, credentials)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize provider '{provider}': {e}") from e
