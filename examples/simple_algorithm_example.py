from decimal import Decimal
from typing import Any

from pytrader import Algorithm, OrderSide, Exchange, IndicatorDefType, ExponentialMovingAverage, PositionType


class SimpleAlgorithm(Algorithm):

    def __init__(self, symbol: str, currency: str):
        super().__init__()

        self._symbol: str = symbol
        self._currency: str = currency

        self._has_terminated: bool = False
        self._state: dict[str, Any] = dict()
        self._action: dict[str, Any] = dict()

    def _build_indicator_defs(self) -> dict[str, IndicatorDefType]:

        defs = {
            "price_ema_60": (
                ExponentialMovingAverage(60),
                lambda exchange: (float(exchange.get_ticker(self._symbol).last_price),),
                0
            )
        }
        return defs

    def _get_is_ready(self) -> bool:
        return all([self._get_indicator(key).is_valid() for key in self._indicator_defs.keys()])

    def _get_has_terminated(self) -> bool:
        return self._has_terminated

    def _get_state(self) -> dict[str, Any]:
        return self._state

    def _get_action(self) -> dict[str, Any]:
        return self._action

    def on_startup(self, exchange: Exchange):
        print(f"[{self.__class__.__name__}] Startup for symbol {self._symbol}.")

    def on_shutdown(self, exchange: Exchange):
        account = exchange.get_account(self._currency)
        print(f"[{self.__class__.__name__}] Shutdown for symbol {self._symbol}. Final Equity: {account.equity:.2f}")

    def on_trade(self, exchange: Exchange):

        ticker = exchange.get_ticker(self._symbol)
        price_ema_60 = Decimal(str(self._get_indicator("price_ema_60").value))

        current_position = exchange.get_account(self._currency).get_position(self._symbol, PositionType.NET)
        position_quantity = current_position.quantity if current_position else Decimal("0")

        print(
            f"Symbol: {self._symbol}, Last Price: {ticker.last_price:.2f}, EMA(60): {price_ema_60:.2f}, "
            f"Position: {position_quantity}"
        )

        if ticker.last_price > Decimal("1.02") * price_ema_60:
            print(f"  SELL signal: {ticker.last_price:.2f} > 1.02 * {price_ema_60:.2f}. Placing SELL order.")
            exchange.create_market_order(self._symbol, OrderSide.SELL, Decimal(".5"))
        elif ticker.last_price < Decimal("0.98") * price_ema_60:
            print(f"  BUY signal: {ticker.last_price:.2f} < 0.98 * {price_ema_60:.2f}. Placing BUY order.")
            exchange.create_market_order(self._symbol, OrderSide.BUY, Decimal(".5"))


if __name__ == "__main__":

    import logging
    import json

    from pytrader.simulation import CorrelatedRandomWalkExchange, SimulatedTrader

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(threadName)s | %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    with open("config.json", 'r') as f:
        config = json.load(f)

    # Initialize the exchange with the loaded configuration
    exchange = CorrelatedRandomWalkExchange(config)

    symbol_to_trade = "BTC/USDT"
    currency = "USDT"

    algorithm = SimpleAlgorithm(symbol=symbol_to_trade, currency=currency)

    step_duration_seconds = config["settings"].get("step_duration", 1.0)
    steps_per_minute = int(60 / step_duration_seconds)
    trader = SimulatedTrader(exchange, algorithm, steps_per_trade=steps_per_minute)

    sim_duration_hours = 24
    total_seconds_in_duration = sim_duration_hours * 60 * 60
    max_steps = int(total_seconds_in_duration / step_duration_seconds)

    logging.info(f"Starting simulation for {sim_duration_hours} hours ({max_steps} steps)...")

    initial_account_state = exchange.get_account(currency)
    logging.info(f"Initial Account State: Cash: {initial_account_state.cash_balance}, Equity: {initial_account_state.equity}")

    trader.run(max_steps=max_steps)

    account = exchange.get_account(currency)

    # Display final account state
    print("Final account state:")
    ticker = exchange.get_ticker('BTC/USDT')
    print(f"  '{symbol_to_trade}' price = {ticker.last_price}")
    print(f"  Cash balance = {account.cash_balance}")
    print(f"  Unrealized P&L = {account.unrealized_pnl}")
    print(f"  Equity = {account.equity}")
    print(f"  Realized P&L = {account.realized_pnl}")
    print(f"  Available margin = {account.available_margin}")
    print(f"  Initial margin requirement = {account.initial_margin_requirement}")
    print(f"  Maintenance margin requirement = {account.maintenance_margin_requirement}")
    print(f"  Reserved margin requirement = {account.reserved_margin_requirement}")

    logging.info("Simulation finished.")
