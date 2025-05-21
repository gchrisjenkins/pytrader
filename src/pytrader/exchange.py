import copy
import dataclasses
from abc import abstractmethod, ABC
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum, unique
from typing import Any


def convert_to_decimal(value: str | int | float | Decimal) -> Decimal:

    if isinstance(value, Decimal):
        value_as_decimal = value
    elif isinstance(value, float):
        value_as_decimal = Decimal(str(value))
    elif isinstance(value, int | str):
        value_as_decimal = Decimal(value)
    else:
        raise TypeError(f"Unsupported type '{type(value)}' for conversion to decimal.Decimal")

    return value_as_decimal


@dataclasses.dataclass
class Ticker:
    symbol: str
    last_price: Decimal
    mark_price: Decimal
    index_price: Decimal
    funding_rate: Decimal


@unique
class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


@unique
class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@unique
class OrderStatus(Enum):
    OPEN = "open"
    FILLED = "filled"
    CANCELED = "canceled"


@dataclasses.dataclass
class Order:
    id: str
    symbol: str
    type: OrderType
    side: OrderSide
    price: Decimal
    quantity: Decimal
    status: OrderStatus = OrderStatus.OPEN


@unique
class PositionType(Enum):
    LONG = "long"
    SHORT = "short"
    NET = "net"


@dataclasses.dataclass
class Position:
    symbol: str
    type: PositionType = PositionType.NET
    quantity: Decimal = Decimal("0.0")
    entry_price: Decimal = Decimal("0.0")


@dataclasses.dataclass
class Account:

    currency: str
    cash_balance: Decimal = Decimal("0.0")
    realized_pnl: Decimal = Decimal("0.0")
    unrealized_pnl: Decimal = Decimal("0.0")
    equity: Decimal = Decimal("0.0")
    available_margin: Decimal = Decimal("0.0")
    initial_margin_requirement: Decimal = Decimal("0.0")
    maintenance_margin_requirement: Decimal = Decimal("0.0")
    reserved_margin_requirement: Decimal = Decimal("0.0")

    _positions: dict[str, list[Position]] = dataclasses.field(default_factory=dict)
    _orders: dict[str, list[Order]] = dataclasses.field(default_factory=dict)

    def get_position(self, symbol: str, type_: PositionType) -> Position:
        return next((position for position in self._get_positions(symbol) if position.type == type_), None)

    def add_position(self, position: Position):
        self._get_positions(position.symbol).append(position)

    def delete_position(self, position: Position):
        self._get_positions(position.symbol).remove(position)

    def _get_positions(self, symbol: str) -> list[Position]:
        return self._positions.setdefault(symbol, [])

    @property
    def positions(self) -> dict[str, list[Position]]:
        return copy.deepcopy(self._positions)

    def get_orders(self, symbol: str, type_: OrderType | None, status: OrderStatus | None) -> list[Order]:
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


class Market(ABC):

    def __init__(self,
        symbol: str, quote_currency: str, tick_size: Decimal, order_increment: Decimal, initial_margin_rate: Decimal,
        maintenance_margin_rate: Decimal, interest_rate: Decimal
    ):

        self._set_symbol(symbol)
        self._set_quote_currency(quote_currency)
        self._set_tick_size(tick_size)
        self._set_order_increment(order_increment)
        self._set_initial_margin_rate(initial_margin_rate)
        self._set_maintenance_margin_rate(maintenance_margin_rate)
        self._interest_rate: Decimal = interest_rate

        self._last_price = Decimal("sNaN")
        self._mark_price = Decimal("sNaN")
        self._index_price = Decimal("sNaN")
        self._funding_rate = Decimal("sNaN")

    @abstractmethod
    def create_market_order(self, side: OrderSide, quantity: Decimal) -> Order:
        pass

    @abstractmethod
    def create_limit_order(self, side: OrderSide, price: Decimal, quantity: Decimal) -> Order:
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> Order | None:
        pass

    @property
    def symbol(self) -> str:
        return self._symbol

    def _set_symbol(self, value: str):
        if not value:
            raise ValueError("'symbol' cannot be empty")
        self._symbol = value

    @property
    def quote_currency(self) -> str:
        return self._quote_currency

    def _set_quote_currency(self, value: str):
        if not value:
            raise ValueError("'quote_currency' cannot be empty")
        self._quote_currency = value

    @property
    def tick_size(self) -> Decimal:
        return self._tick_size

    def _set_tick_size(self, value: Decimal):

        if not isinstance(value, Decimal):
            raise TypeError(f"'tick_size' must be a Decimal type, but found: {type(value)}")

        if value <= 0:
            raise ValueError(f"'tick_size' must be positive, but found {value}")

        self._tick_size = value

    @property
    def order_increment(self) -> Decimal:
        return self._order_increment

    def _set_order_increment(self, value: Decimal):
        if not isinstance(value, Decimal):
            raise TypeError(f"'order_increment' must be a Decimal type, but found: {type(value)}")
        if value <= 0:
            raise ValueError(f"'order_increment' must be positive, but found {value}")
        self._order_increment = value

    @property
    def initial_margin_rate(self) -> Decimal:
        return self._initial_margin_rate

    def _set_initial_margin_rate(self, value: Decimal):
        if not isinstance(value, Decimal):
            raise TypeError(f"'initial_margin_rate' must be a Decimal type, but found: {type(value)}")
        if not (0 < value < 1):
            raise ValueError(f"'initial_margin_rate' must be between 0 and 1, but found {value}")
        self._initial_margin_rate = value

    @property
    def maintenance_margin_rate(self) -> Decimal:
        return self._maintenance_margin_rate

    def _set_maintenance_margin_rate(self, value: Decimal):
        if not isinstance(value, Decimal):
            raise TypeError(f"'maintenance_margin_rate' must be a Decimal type, but found: {type(value)}")
        if not (0 < value < 1):
            raise ValueError(f"'maintenance_margin_rate' must be between 0 and 1, but found {value}")
        self._maintenance_margin_rate = value

    @property
    def interest_rate(self) -> Decimal:
        return self._interest_rate

    def _set_interest_rate(self, value: Decimal):

        if not isinstance(value, Decimal):
            raise TypeError(f"'interest_rate' must be a Decimal type, but found: {type(value)}")

        if value < 0:
            raise ValueError(f"'interest_rate' must be non-negative, but found: {value}")

        self._interest_rate = value

    @property
    def last_price(self) -> Decimal:
        return self._last_price

    def _set_last_price(self, value: int | float | Decimal):

        rounded_price = self.round_price(value)
        if rounded_price <= 0:
            raise ValueError(
                f"[{self.symbol}] 'last_price' ({value}) rounded to zero or less with 'tick_size' ({self.tick_size})"
            )

        self._last_price = rounded_price

    @property
    def mark_price(self) -> Decimal:
        return self._mark_price

    def _set_mark_price(self, value: int | float | Decimal):

        rounded_price = self.round_price(value)
        if rounded_price <= 0:
            raise ValueError(
                f"[{self.symbol}] 'mark_price' ({value}) rounded to zero or less with 'tick_size' ({self.tick_size})"
            )

        self._mark_price = self.round_price(value)

    @property
    def index_price(self) -> Decimal:
        return self._index_price

    def _set_index_price(self, value: int | float | Decimal):

        rounded_price = self.round_price(value)
        if rounded_price <= 0:
            raise ValueError(
                f"[{self.symbol}] 'index_price' ({value}) rounded to zero or less with 'tick_size' ({self.tick_size})"
            )

        self._index_price = rounded_price

    @property
    def funding_rate(self) -> Decimal:
        return self._funding_rate

    def _set_funding_rate(self, value: Decimal):

        if not isinstance(value, Decimal):
            raise TypeError(f"'funding_rate' must be a Decimal type, but found: {type(value)}")

        self._funding_rate = value

    def round_price(self, price: int | float | Decimal) -> Decimal:
        """
        Round a price to the nearest tick size. Handles int, float, or Decimal inputs.

        Args:
            price (int | float | Decimal): Price to be rounded.

        Returns:
            Decimal: Price rounded to the nearest multiple of tick_size.
        """
        return (convert_to_decimal(price) / self._tick_size).quantize(
            Decimal('1'), rounding=ROUND_HALF_UP
        ) * self._tick_size

    def round_quantity(self, quantity: int | float | Decimal) -> Decimal:
        """
        Round an order quantity to the nearest order increment. Handles int, float, or Decimal inputs.

        Args:
            quantity (int | float | Decimal): Order quantity to be rounded.

        Returns:
            Decimal: Order quantity rounded to the nearest multiple of order_increment.
        """
        return (convert_to_decimal(quantity) / self._order_increment).quantize(
            Decimal('1'), rounding=ROUND_HALF_UP
        ) * self._order_increment


class SimulatedMarket(Market, ABC):

    @abstractmethod
    def step(self, *args, **kwargs) -> dict[str, Any]:
        pass


class Exchange(ABC):

    @abstractmethod
    def get_ticker(self, symbol: str) -> Ticker | None:
        pass

    @abstractmethod
    def get_account(self, quote_currency: str) -> Account | None:
        pass

    @abstractmethod
    def create_market_order(self, symbol: str, side: OrderSide, quantity: Decimal) -> Order:
        pass

    @abstractmethod
    def create_limit_order(self, symbol: str, side: OrderSide, price: Decimal, quantity: Decimal) -> Order:
        pass

    @abstractmethod
    def cancel_order(self, order: Order) -> Order:
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
