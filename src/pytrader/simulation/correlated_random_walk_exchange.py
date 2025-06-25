import copy
import dataclasses
import logging
import math
import uuid
from decimal import Decimal
from typing import Any

import numpy as np

from pytrader import (
    Order, OrderType, OrderSide, OrderStatus, Position, PositionType, Account, Ticker, convert_to_decimal, Settings
)
from pytrader.simulation import SimulatedExchange, SimulatedMarket



class CorrelatedRandomWalkMarket(SimulatedMarket):
    """
    Simulates the price dynamics of a random walk market correlated with specified underlying statistics. Manages the
    creation, matching, and cancellation of limit orders.
    """

    def __init__(self,
        symbol: str, quote_currency: str, tick_size: Decimal, order_increment: Decimal, initial_price: Decimal,
        initial_margin_rate: Decimal, maintenance_margin_rate: Decimal, interest_rate: Decimal, volatility: float,
        premium_volatility: float, funding_basis_ema_alpha: float, correlation_factor: float, premium_rate_min: Decimal,
        premium_rate_max: Decimal, seed: int = 42
    ):
        super().__init__(
            symbol, quote_currency, tick_size, order_increment, initial_margin_rate, maintenance_margin_rate,
            interest_rate
        )

        cls = type(self)
        self._logger: logging.Logger = logging.getLogger(f"{cls.__module__}.{cls.__name__}")

        self._set_volatility(volatility)
        self._set_correlation_factor(correlation_factor)
        self._set_premium_volatility(premium_volatility)
        self._set_premium_rate_min(premium_rate_min)
        self._set_premium_rate_max(premium_rate_max)
        self._set_funding_basis_ema_alpha(funding_basis_ema_alpha)

        # --- Relationship validations ---
        if self.maintenance_margin_rate >= self.initial_margin_rate:
            raise ValueError(
                f"[{self.symbol}] 'maintenance_margin_rate' ({self.maintenance_margin_rate}) "
                f"must be strictly less than 'initial_margin_rate' ({self.initial_margin_rate})"
            )
        if self._premium_rate_min > self._premium_rate_max:
            raise ValueError(
                f"[{self.symbol}] 'premium_rate_min' ({self._premium_rate_min}) cannot be greater than "
                f"'premium_rate_max' ({self._premium_rate_max})"
            )

        # --- Initialize state after validation ---
        self.rng = np.random.default_rng(seed)

        self._set_last_price(initial_price)
        self._set_index_price(self.last_price)  # Start index price at last price
        self._set_funding_basis(Decimal("0.0"))
        self._set_mark_price(self.last_price)
        self._set_funding_rate(self.interest_rate)  # Initial funding rate

        self._limit_orders: dict[str, Order] = {}

    def step(self, underlying_noise: float) -> dict[str, Any]:
        """Take one step in the progression of the index price and market price dynamics.

        1. Index price moves based on correlated underlying noise + market specific volatility.
        2. Market price (last_price) performs a random walk around the *new* index price.
        3. Match and fill limit orders for the current step.

        Args:
            underlying_noise (float): Relative change driven purely by the underlying asset.

        Returns:
            list[Order]: A list of limit orders filled in the current step.
        """
        # --- 1. Update Index Price ---
        # Calculate noise specific to the index/market using its volatility relative to underlying
        index_noise = self.rng.normal(0., float(self.volatility) * math.sqrt(1. - self.correlation_factor ** 2))
        # Combine correlated underlying noise with index specific noise
        total_index_noise = self.correlation_factor * underlying_noise + index_noise
        # Update index price based on the previous value and the combined index noise
        self._set_index_price(max(
            self.tick_size,  # Ensure index price is positive and adheres to tick size
            self.round_price(float(self.index_price) * (1. + total_index_noise))
        ))

        # --- 2. Update Market Price (Last Price) ---
        # Generate noise for the market price's random walk around the index price
        premium_noise = self.rng.normal(0., float(self.premium_volatility))
        # Calculate the new market price based on the *updated* index price plus premium noise
        # The base for the random walk is the *new* index price for this step
        self._set_last_price(max(
            self.tick_size,  # Ensure market price is positive and adheres to tick size
            self.round_price(float(self.index_price) * (1. + premium_noise))
        ))

        # --- 3. Update EMA of the Basis ---
        # Basis = Contract Price - Index Price
        # Funding Basis = EMA(Basis)
        # Mark Price = Index Price + Funding Basis
        # EMA = alpha * current_price + (1 - alpha) * previous_EMA
        # Handle initialization on the very first step where the funding basis is zero.
        basis = self.last_price - self.index_price
        self._set_funding_basis(self.round_price(
            self.funding_basis_ema_alpha * float(basis)
            + (1. - self.funding_basis_ema_alpha) * float(self.funding_basis)
        ))
        self._set_mark_price(self.index_price + self.funding_basis)

        self._update_funding_rate()

        return {'filled_limit_orders': self._match_and_fill_limit_orders()}

    def create_market_order(self, side: OrderSide, quantity: Decimal) -> Order:
        """Create a market order to buy or sell at market price.

        Args:
            side (OrderSide): Direction of the order.
            quantity (Decimal): Amount to buy or sell.

        Returns:
            Order: The created market order.
        """
        if not side:
            raise ValueError("Order side is required")

        if quantity <= 0:
            raise ValueError("Order quantity must be positive")

        quantity = self.round_quantity(float(quantity))
        if quantity == Decimal("0"):
            raise ValueError("Order quantity rounds to zero based on order increment")

        return Order(
            id=str(uuid.uuid4()), symbol=self.symbol, type=OrderType.MARKET, side=side, price=self.last_price,
            quantity=quantity
        )

    def create_limit_order(self, side: OrderSide, price: Decimal, quantity: Decimal) -> Order:
        """Create a limit order to buy or sell at a specified price.

        Args:
            side (str): Direction of the order ('buy' or 'sell').
            price (Decimal): Price at which to place the order.
            quantity (Decimal): Amount to buy or sell.

        Returns:
            str: Unique identifier for the created limit order.
        """
        if not side:
            raise ValueError("Order side is required")

        if price <= 0:  # Assuming prices must be positive
            raise ValueError("Order price must be positive")

        price = self.round_price(float(price))  # Careful with float conversion if critical
        if price <= Decimal("0"):  # Re-check after rounding
            raise ValueError("Order price rounds to zero or less based on tick size")

        if quantity <= 0:
            raise ValueError("Order quantity must be positive")

        quantity = self.round_quantity(float(quantity))
        if quantity == Decimal("0"):
            raise ValueError("Order quantity rounds to zero based on order increment")

        order = Order(
            id=str(uuid.uuid4()), symbol=self.symbol, type=OrderType.LIMIT, side=side, price=price, quantity=quantity
        )
        self._limit_orders[order.id] = order

        return order

    def cancel_order(self, order_id: str) -> Order | None:
        """
        Cancel an order by its identifier (only open orders are limit orders).

        Args:
            order_id (str): The id of order to be cancelled.

        Returns:
            Order | None: The cancelled limit order.

        Raises:
            ValueError: If the order ID does not exist or the order is not active.
        """
        if order_id not in self._limit_orders:
            self._logger.debug(f"Order ID {order_id} not found in limit orders")
            return None

        return self._limit_orders.pop(order_id)

    def get_limit_orders(self):
        return copy.deepcopy(self._limit_orders)

    def _match_and_fill_limit_orders(self) -> list[Order]:
        """Match and fill all limit orders at the current price.

        Returns:
            list[Order]: The filled limit orders.
        """
        filled_orders: list[Order] = []
        filled_order_ids: list[str] = []

        if not self._limit_orders:
            return filled_orders

        for order in list(self._limit_orders.values()):
            is_filled = False

            if order.side == OrderSide.BUY and self.last_price <= order.price:
                is_filled = True
            elif order.side == OrderSide.SELL and self.last_price >= order.price:
                is_filled = True

            if is_filled:
                # Simple fill model: fill completely at the last trade price
                filled_order_ids.append(order.id)

        for order_id in filled_order_ids:
            filled_order = self._limit_orders.pop(order_id)
            filled_orders.append(filled_order)
            filled_order.status = OrderStatus.FILLED

        return filled_orders

    def _update_funding_rate(self):
        """Calculates and updates the funding rate based on index and mark price premium."""

        if self.index_price <= 0:  # Avoid division by zero
            # Default to interest rate if index is zero or negative (edge case)
            self._set_funding_rate(self.interest_rate)
            return

        premium = self.mark_price - self.index_price

        # Premium rate clamped to [premium rate min, premium rate max]
        premium_rate = max(min(premium / self.index_price, self.premium_rate_max), self.premium_rate_min)

        # Calculate final rate (Interest Rate + Premium Rate)
        self._set_funding_rate(self.interest_rate + premium_rate)

    @property
    def volatility(self) -> float:
        return self._volatility

    def _set_volatility(self, value: float):

        if not isinstance(value, float):
            raise TypeError(f"'volatility' must be a float type, but found: {type(value)}")

        if value < 0:
            raise ValueError(f"[{self.symbol}] 'volatility' cannot be negative, but found {value}")

        self._volatility = value

    @property
    def correlation_factor(self) -> float:
        return self._correlation_factor

    def _set_correlation_factor(self, value: float):

        if not isinstance(value, float):
            raise TypeError(f"'volatility' must be a float type, but found: {type(value)}")

        if not (-1.0 <= value <= 1.0):
            raise ValueError(
                f"[{self.symbol}] 'correlation_factor' must be in range -1.0 and 1.0, but found {value}"
            )

        self._correlation_factor = value

    @property
    def premium_volatility(self) -> float:
        return self._premium_volatility

    def _set_premium_volatility(self, value: float):

        if not isinstance(value, float):
            raise TypeError(f"'premium_volatility' must be a float type, but found: {type(value)}")

        if value < 0:
            raise ValueError(f"[{self.symbol}] 'premium_volatility' cannot be negative, but found {value}")

        self._premium_volatility = value

    @property
    def funding_basis(self) -> Decimal:
        return self._funding_basis

    def _set_funding_basis(self, value: int | float | Decimal):
        self._funding_basis = self.round_price(value)

    @property
    def funding_basis_ema_alpha(self) -> float:
        return self._funding_basis_ema_alpha

    def _set_funding_basis_ema_alpha(self, value: float):

        if not isinstance(value, float):
            raise TypeError(f"'funding_basis_ema_alpha' must be a float type, but found: {type(value)}")

        if not (0 < value <= 1):
            raise ValueError(
                f"[{self.symbol}] 'funding_basis_ema_alpha' must be between 0 and 1 (inclusive), but found {value}"
            )

        self._funding_basis_ema_alpha = value

    @property
    def premium_rate_min(self) -> Decimal:
        return self._premium_rate_min

    def _set_premium_rate_min(self, value: Decimal):

        if not isinstance(value, Decimal):
            raise TypeError(f"'premium_rate_min' must be a Decimal type, but found: {type(value)}")

        self._premium_rate_min = value

    @property
    def premium_rate_max(self) -> Decimal:
        return self._premium_rate_max

    def _set_premium_rate_max(self, value: Decimal):

        if not isinstance(value, Decimal):
            raise TypeError(f"'premium_rate_max' must be a Decimal type, but found: {type(value)}")

        self._premium_rate_max = value


class CorrelatedRandomWalkExchange(SimulatedExchange):
    """
    Manages multiple markets, user account, and order handling for leveraged perpetual futures. The price dynamics of
    all markets are correlated with an underlying random walk model.
    """

    def __init__(self, config: dict):

        cls = type(self)
        self._logger: logging.Logger = logging.getLogger(f"{cls.__module__}.{cls.__name__}")

        settings: Settings = type(self)._validate_and_create_settings(config)

        super().__init__(settings)

        # --- Validate and create account ---
        self._account: Account = self._validate_and_create_account(config)

        # --- Validate and create markets ---
        self._markets = self._validate_and_create_markets(config)

        # --- Initialize exchange state ---
        self._current_step = 0

        # Use the global seed for the exchange's underlying RNG
        self._underlying_rng = np.random.default_rng(self._settings.seed)

        self._update_account()

    def step(self, *args, **kwargs) -> int:
        """
        Advance the exchange by one step, update markets, and check for funding and liquidation.
        """
        self._current_step += 1

        underlying_noise = self._underlying_rng.normal(0, self._settings.underlying_volatility)
        for market in self._markets.values():
            filled_limit_orders: list[Order] = market.step(underlying_noise)['filled_limit_orders']
            for order in filled_limit_orders:
                self._execute_order(order)
        self._apply_funding()
        self._check_for_liquidation()
        self._update_account()

        return self._current_step

    def _get_current_step(self) -> int:
        return self._current_step

    def get_ticker(self, symbol: str) -> Ticker:
        market = self._get_market(symbol)
        return Ticker(
            symbol=symbol, last_price=market.last_price, mark_price=market.mark_price, index_price=market.index_price,
            funding_rate=market.funding_rate
        )

    def get_account(self, quote_currency: str) -> Account:
        if self._account.currency != quote_currency:
            raise ValueError(f"Unknown quote currency: {quote_currency}")
        return copy.deepcopy(self._account)

    def create_market_order(self, symbol: str, side: OrderSide, quantity: Decimal) -> Order:
        """
        Create and execute a market order immediately at the current price.
        """
        market = self._get_market(symbol)
        order = market.create_market_order(side, quantity)
        self._account.add_order(self._execute_order(order))

        return order

    def create_limit_order(self, symbol: str, side: OrderSide, price: Decimal, quantity: Decimal) -> Order:
        """
        Create a limit order and add it to the market's limit order book.
        """
        market = self._get_market(symbol)

        # Calculate potential IM for *this specific* order
        potential_order_value = quantity * price  # Use input price
        potential_initial_margin_for_this_order = potential_order_value * Decimal(str(market.initial_margin_rate))

        # Check if AVAILABLE margin (Equity - PositionIM - ReservedLimitIM) can cover THIS order's IM
        current_available_margin = self._calculate_available_margin()  # This now includes reservation
        if current_available_margin < potential_initial_margin_for_this_order:
            raise ValueError(
                f"Insufficient margin to place limit order. Available: {current_available_margin}, "
                f"Needed: {potential_initial_margin_for_this_order}"
            )

        # Available margin will now recalculate lower on the next call because of the new order requirements
        order = market.create_limit_order(side, price, quantity)
        self._account.add_order(order)
        return order

    def cancel_order(self, order: Order) -> Order:
        """
        Cancel the order (the only open orders on the exchange should be limit orders).
        """
        market = self._markets[order.symbol]
        market.cancel_order(order.id)
        order.status = OrderStatus.CANCELED
        return order

    def _get_market(self, symbol: str) -> SimulatedMarket:
        if symbol not in self._markets:
            raise ValueError(f"Unknown market symbol: {symbol}")
        return self._markets[symbol]

    def _get_position(self, symbol: str) -> Position:
        """
        Get the current position (create if not present) for symbol. The only supported position type is currently
        PositionType.NET.
        """
        _ = self._get_market(symbol)
        if self._account.get_position(symbol, PositionType.NET) is None:
            self._account.add_position(Position(symbol))
        return self._account.get_position(symbol, PositionType.NET)

    def _execute_order(self, order: Order) -> Order:
        """
        Execute an order, update position, and apply fees.
        """
        position = self._get_position(order.symbol)

        side_multiplier = 1 if order.side == OrderSide.BUY else -1
        new_quantity = position.quantity + side_multiplier * order.quantity

        order_value = order.quantity * order.price
        fee = order_value * Decimal(str(
            self._settings.maker_fee if order.type == OrderType.LIMIT else self._settings.taker_fee
        ))
        if self._account.cash_balance < fee:
            raise ValueError("Insufficient balance to pay trading fee")

        # Position reversal
        if position.quantity * new_quantity < 0:
            # Step 1: Close the existing position
            realized_pnl = (order.price - position.entry_price) * position.quantity
            self._account.realized_pnl += realized_pnl
            self._account.cash_balance += realized_pnl
            # Step 2: Open new position with remaining quantity
            if abs(new_quantity) > 0:
                position.quantity = new_quantity
                position.entry_price = order.price
            else:
                self._account.delete_position(position)
                # del self._account.positions[symbol]
        # Position fully closed
        elif new_quantity == 0:
            realized_pnl = (order.price - position.entry_price) * position.quantity
            self._account.realized_pnl += realized_pnl
            self._account.cash_balance += realized_pnl
            self._account.delete_position(position)
            # del self._account.positions[symbol]
        # Position size adjusted
        else:
            if position.quantity == 0:
                position.entry_price = order.price
            else:
                delta_quantity = new_quantity - position.quantity
                if abs(new_quantity) > abs(position.quantity):
                    position.entry_price = (
                        (position.entry_price * position.quantity + order.price * delta_quantity) / new_quantity
                    )
            position.quantity = new_quantity

        self._account.cash_balance -= fee
        order.status = OrderStatus.FILLED

        return order

    def _apply_funding(self):
        """
        Apply funding rates to positions at expiration of funding interval.
        """
        if self._current_step % self._settings.funding_interval == 0:
            # for symbol, position in self._account.get.positions.items():
            for symbol, positions in self._account.positions.items():
                for position in positions:
                    market = self._markets[position.symbol]
                    # Use the updated dynamic rate from the market
                    # mark_price = market.last_price  # Use last_price as mark price
                    funding_payment = position.quantity * market.mark_price * market.funding_rate

                    # Apply payment: Subtracting handles direction automatically based on signs
                    # print(
                    #     f"  {symbol}: PosQty={position.quantity}, MarkPx={mark_price:.2f}, "
                    #     f"Rate={market.funding_rate:.8f}, Payment={funding_payment:.4f}"
                    # )  # Debug
                    self._account.cash_balance -= funding_payment

    def _check_for_liquidation(self):
        """
        Check if the account equity is below the total maintenance margin requirement.
        """
        total_equity = self._calculate_equity()
        total_maintenance_margin_req = self._calculate_maintenance_margin_requirement()

        if total_equity < total_maintenance_margin_req:
            print(f"LIQUIDATION TRIGGERED: Equity ({total_equity}) < Total MM Req ({total_maintenance_margin_req})")
            # Implement liquidation strategy (e.g., close all positions via market orders)
            # positions_to_liquidate = list(self._account.positions.items())  # Copy items
            # for symbol, position in positions_to_liquidate:
            for symbol, positions in self._account.positions.items():
                market = self._markets[symbol]
                for position in positions:
                    # Check if position still exists (may have been closed by previous iteration if liquidating one by
                    # one)
                    # if symbol in self._account.positions and self._account.positions[symbol].quantity != Decimal("0"):
                    if position.quantity != Decimal("0"):
                        print(f"Liquidating position in {symbol} ({position.quantity})")
                        liq_order = Order(
                            id=str(uuid.uuid4()), symbol=symbol, type=OrderType.MARKET,
                            side=OrderSide.SELL if position.quantity > 0 else OrderSide.BUY,
                            price=market.last_price,  # Fill price determined by execution
                            quantity=abs(position.quantity)
                        )
                        try:
                            # Note: Liquidation might have higher fees or slippage (not modeled here)
                            self._execute_order(liq_order)
                        except ValueError as e:
                            print(f"Error during liquidation execution for {symbol}: {e}")
                            # Handle potential issues like insufficient funds for fees AFTER liquidation starts
                            # In a real system, this might lead to negative balance/insurance fund.

                    # Cancel all remaining limit orders after liquidation
                    limit_orders = market.get_limit_orders()
                    print(f"Cancelling {len(limit_orders)} limit orders post-liquidation.")
                    for order in limit_orders.values():
                        try:
                            self.cancel_order(order)
                        except Exception as e:
                            print(f"Error cancelling order {order.id} post-liquidation: {e}")

        # Check for negative equity after liquidations (Bankruptcy)
        final_equity = self._calculate_equity()
        if final_equity < 0:
            print(f"ACCOUNT BANKRUPT: Final Equity = {final_equity}")
            # Handle bankruptcy state if necessary (e.g., reset balance, halt trading)

    def _update_account(self):

        self._account.unrealized_pnl = self._calculate_unrealized_pnl()
        self._account.equity = self._calculate_equity()
        self._account.initial_margin_requirement = self._calculate_initial_margin_requirement()
        self._account.maintenance_margin_requirement = self._calculate_maintenance_margin_requirement()
        self._account.reserved_margin_requirement = self._calculate_reserved_margin_requirement()
        self._account.available_margin = self._calculate_available_margin()

    def _calculate_unrealized_pnl(self) -> Decimal:
        """
        Calculate the total unrealized profit and loss for all open positions.
        """
        return sum(
            (
                (self._markets[position.symbol].mark_price - position.entry_price) * position.quantity
                for symbol, positions in self._account.positions.items() for position in positions
            ),
            Decimal('0.0')
        )

    def _calculate_equity(self) -> Decimal:
        """
        Calculate the equity: cash balance + unrealized PnL.
        """
        return self._account.cash_balance + self._calculate_unrealized_pnl()

    def _calculate_initial_margin_requirement(self) -> Decimal:
        """
        Calculate the total initial margin required for all open positions.
        """
        initial_margin_requirement = Decimal("0.0")
        for symbol, positions in self._account.positions.items():
            market = self._markets[symbol]
            for position in positions:
                if position.quantity != Decimal("0"):
                    # Use current market price for position value
                    position_value = abs(position.quantity) * market.mark_price
                    initial_margin_rate = Decimal(str(market.initial_margin_rate))
                    initial_margin_requirement += position_value * initial_margin_rate

        return initial_margin_requirement

    def _calculate_maintenance_margin_requirement(self) -> Decimal:
        """
        Calculate the total maintenance margin required for all open positions.
        """
        maintenance_margin_requirement = Decimal("0.0")
        for symbol, positions in self._account.positions.items():
            for position in positions:
                if position.quantity != Decimal("0"):
                    market = self._markets[symbol]
                    # Use current market price for position value
                    position_value = abs(position.quantity) * market.mark_price
                    maintenance_margin_rate = Decimal(str(market.maintenance_margin_rate))
                    maintenance_margin_requirement += position_value * maintenance_margin_rate

        return maintenance_margin_requirement

    def _calculate_reserved_margin_requirement(self) -> Decimal:
        """
        Calculate the total initial margin potentially required by all open limit orders.
        """
        reserved_margin = Decimal("0.0")
        # A more sophisticated model would consider how orders affect existing positions (hedging)
        # This simple model assumes each order adds to the margin requirement independently.
        for market in self._markets.values():
            for order in market.get_limit_orders().values():
                # if order.type == OrderType.LIMIT:
                order_value = order.quantity * order.price  # Use limit price
                initial_margin_rate = Decimal(str(market.initial_margin_rate))
                reserved_margin += order_value * initial_margin_rate

        return reserved_margin

    def _calculate_available_margin(self) -> Decimal:
        """
        Calculate the available margin: equity - (initial margin requirement + reserved margin requirement).
        """
        equity = self._calculate_equity()
        initial_margin_req = self._calculate_initial_margin_requirement()
        # Also consider margin reserved for open orders
        reserved_margin_req = self._calculate_reserved_margin_requirement()

        # Available margin is equity minus margin used by positions AND reserved by limit orders
        return equity - (initial_margin_req + reserved_margin_req)

    @staticmethod
    def _validate_and_create_settings(config: dict) -> Settings:
        try:
            settings_data = config.get("settings")
            if not settings_data:
                raise ValueError("'settings' section missing in config")

            maker_fee = Decimal(settings_data["maker_fee"])
            if maker_fee < 0:
                raise ValueError("maker_fee cannot be negative")

            taker_fee = Decimal(settings_data["taker_fee"])
            if taker_fee < 0:
                raise ValueError("taker_fee cannot be negative")

            funding_interval = int(settings_data["funding_interval"])
            if funding_interval <= 0:
                raise ValueError("funding_interval must be positive")

            underlying_volatility = float(settings_data["underlying_volatility"])
            if underlying_volatility < 0:
                raise ValueError("underlying_volatility cannot be negative")

            step_duration = float(settings_data["step_duration"])
            if step_duration <= 0:
                raise ValueError("step_duration must be positive")
            seed = int(settings_data.get("seed", 42))

            return Settings(
                maker_fee=maker_fee,
                taker_fee=taker_fee,
                funding_interval=funding_interval,
                underlying_volatility=underlying_volatility,
                step_duration_seconds=step_duration,
                seed=seed
            )
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f"Invalid 'settings' configuration: {e}") from e

    @staticmethod
    def _validate_and_create_account(config: dict) -> Account:
        try:
            account_data = config.get("account")
            if not account_data:
                raise ValueError("'account' section missing in config")

            currency = account_data.get("currency")
            if not currency:
                raise ValueError("'currency' is required")

            initial_cash = Decimal(account_data.get("cash_balance"))
            if initial_cash and initial_cash < 0:
                raise ValueError("'cash_balance' cannot be negative")

            return Account(currency=currency, cash_balance=initial_cash)
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f"Invalid 'account' configuration: {e}") from e

    def _validate_and_create_markets(self, config: dict) -> dict[str, CorrelatedRandomWalkMarket]:
        markets: dict[str, CorrelatedRandomWalkMarket] = {}
        try:
            market_data = config.get("markets")
            if market_data is None:
                raise ValueError("'markets' section missing in config")
            if not isinstance(market_data, list):
                raise ValueError("'markets' section is not a list")
            if not market_data:  # Empty list check
                raise ValueError("'markets' list cannot be empty")

            market_symbols = set()
            for i, market_config in enumerate(market_data):
                if not isinstance(market_config, dict):
                    raise ValueError(f"Market configuration at index {i} must be a dictionary.")
                symbol = market_config.get("symbol")
                if not symbol:
                    raise ValueError(f"Market configuration at index {i} is missing 'symbol'.")
                if symbol in market_symbols:
                    raise ValueError(f"Duplicate market symbol found: {symbol}")

                # Pass a unique seed derived from the global seed
                market_seed = self._settings.seed + i
                markets[symbol] = CorrelatedRandomWalkMarket(
                    symbol=symbol,
                    quote_currency=market_config['quote_currency'],
                    tick_size=convert_to_decimal(market_config['tick_size']),
                    order_increment=convert_to_decimal(market_config['order_increment']),
                    initial_price=convert_to_decimal(market_config['initial_price']),
                    initial_margin_rate=convert_to_decimal(market_config['initial_margin_rate']),
                    maintenance_margin_rate=convert_to_decimal(market_config['maintenance_margin_rate']),
                    interest_rate=convert_to_decimal(market_config['interest_rate']),
                    volatility=market_config['volatility'],
                    premium_volatility=market_config['premium_volatility'],
                    funding_basis_ema_alpha=market_config['funding_basis_ema_alpha'],
                    correlation_factor=market_config['correlation_factor'],
                    premium_rate_min=convert_to_decimal(market_config['premium_rate_min']),
                    premium_rate_max=convert_to_decimal(market_config['premium_rate_max']),
                    seed=market_seed
                )
                market_symbols.add(symbol)
        except (KeyError, ValueError, TypeError) as e:
            # Catch errors from Market validation or list processing
            raise ValueError(f"Invalid 'markets' configuration: {e}") from e
        except Exception as e:
            # Catch any unexpected error during Market initialization
            raise ValueError(f"Unexpected error initializing markets: {e}") from e

        return markets
