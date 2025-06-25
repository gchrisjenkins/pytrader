from decimal import Decimal
from typing import Any

from pytrader import Algorithm, OrderSide, Exchange, ExponentialMovingAverage, PositionType, \
    Duration, TimeUnit, PriceEMA


class SimpleAlgorithm(Algorithm):

    def __init__(self, currency: str, symbol: str, trade_interval_seconds: int):
        super().__init__(currency, symbol, trade_interval_seconds, None)

        cls = type(self)
        self._logger: logging.Logger = logging.getLogger(f"{cls.__module__}.{cls.__name__}")

        self._has_terminated: bool = False
        self._state: dict[str, Any] = dict()
        self._action: dict[str, Any] = dict()

    def _build(self):

        symbol = self.symbols[0]

        price_ema_5 = PriceEMA(symbol, 5)
        self._add_indicator("price_ema_5", price_ema_5, 0)

        price_ema_15 = PriceEMA(symbol, 15)
        self._add_indicator("price_ema_15", price_ema_15, 1)

        price_ema_60 = PriceEMA(symbol, 60)
        self._add_indicator("price_ema_60", price_ema_60, 2)

        # self._add_indicator(
        #     "price_ema_5",
        #     ExponentialMovingAverage(5, lambda exchange: (float(exchange.get_ticker(self.symbols[0]).last_price),)),
        #     # lambda exchange: (float(exchange.get_ticker(self.symbols[0]).last_price),),
        #     0
        # )
        #
        # self._add_indicator(
        #     "price_ema_15",
        #     ExponentialMovingAverage(15, lambda exchange: (float(exchange.get_ticker(self.symbols[0]).last_price),)),
        #     # lambda exchange: (float(exchange.get_ticker(self.symbols[0]).last_price),),
        #     0
        # )
        #
        # self._add_indicator(
        #     "price_ema_60",
        #     ExponentialMovingAverage(60, lambda exchange: (float(exchange.get_ticker(self.symbols[0]).last_price),)),
        #     # lambda exchange: (float(exchange.get_ticker(self.symbols[0]).last_price),),
        #     0
        # )

    # def _build_indicator_defs(self) -> dict[str, IndicatorDefType]:
    #
    #     defs = {
    #         "price_ema_60": (
    #             ExponentialMovingAverage(60),
    #             lambda exchange: (float(exchange.get_ticker(self.symbols[0]).last_price),),
    #             0
    #         )
    #     }
    #     return defs

    def _get_is_ready(self) -> bool:
        return all([indicator.is_valid for indicator in self._indicators.values()])

    def _get_has_terminated(self) -> bool:
        return self._has_terminated

    def _get_state(self) -> dict[str, Any]:
        return self._state

    def _get_action(self) -> dict[str, Any]:
        return self._action

    def on_startup(self, exchange: Exchange):

        self._logger.debug(f"Startup of algorithm for currency {self.currency} and symbol {self.symbols[0]}.")

        self._logger.debug("Initial account details:")

        account = exchange.get_account(self.currency)
        ticker = exchange.get_ticker(self.symbols[0])

        self._logger.debug(f"  {self.symbols[0]} price = {ticker.last_price}")
        self._logger.debug(f"  Cash balance = {account.cash_balance}")
        self._logger.debug(f"  Unrealized P&L = {account.unrealized_pnl}")
        self._logger.debug(f"  Equity = {account.equity}")
        self._logger.debug(f"  Realized P&L = {account.realized_pnl}")
        self._logger.debug(f"  Available margin = {account.available_margin}")
        self._logger.debug(f"  Initial margin requirement = {account.initial_margin_requirement}")
        self._logger.debug(f"  Maintenance margin requirement = {account.maintenance_margin_requirement}")
        self._logger.debug(f"  Reserved margin requirement = {account.reserved_margin_requirement}")

    def on_shutdown(self, exchange: Exchange):

        self._logger.debug(f"Shutdown of algorithm for currency {self.currency} and symbol {self.symbols[0]}.")

        self._logger.debug("Final account details:")

        account = exchange.get_account(self.currency)
        ticker = exchange.get_ticker(self.symbols[0])

        self._logger.debug(f"  {self.symbols[0]} price = {ticker.last_price}")
        self._logger.debug(f"  Cash balance = {account.cash_balance}")
        self._logger.debug(f"  Unrealized P&L = {account.unrealized_pnl}")
        self._logger.debug(f"  Equity = {account.equity}")
        self._logger.debug(f"  Realized P&L = {account.realized_pnl}")
        self._logger.debug(f"  Available margin = {account.available_margin}")
        self._logger.debug(f"  Initial margin requirement = {account.initial_margin_requirement}")
        self._logger.debug(f"  Maintenance margin requirement = {account.maintenance_margin_requirement}")
        self._logger.debug(f"  Reserved margin requirement = {account.reserved_margin_requirement}")

    def on_trade(self, exchange: Exchange):

        account = exchange.get_account(self._currency)

        ticker = exchange.get_ticker(self.symbols[0])
        price_ema_60 = Decimal(str(self.get_indicator("price_ema_60").value))

        current_position = account.get_position(self.symbols[0], PositionType.NET)
        position_quantity = current_position.quantity if current_position else Decimal("0")

        self._logger.debug(
            f"Symbol: {self.symbols[0]}, Last Price: {ticker.last_price:.2f}, EMA(60): {price_ema_60:.2f}, "
            f"Position: {position_quantity}"
        )

        if ticker.last_price > Decimal("1.02") * price_ema_60 and position_quantity >= Decimal("0"):
            self._logger.debug(
                f"--> SELL signal: {ticker.last_price:.2f} > 1.02 * {price_ema_60:.2f}. Placing SELL order."
            )
            exchange.create_market_order(self.symbols[0], OrderSide.SELL, Decimal(".5"))
        elif ticker.last_price < Decimal("0.98") * price_ema_60 and position_quantity <= Decimal("0"):
            self._logger.debug(f"--> BUY signal: {ticker.last_price:.2f} < 0.98 * {price_ema_60:.2f}. Placing BUY order.")
            exchange.create_market_order(self.symbols[0], OrderSide.BUY, Decimal(".5"))


if __name__ == "__main__":

    import logging
    import json

    from pytrader.simulation import CorrelatedRandomWalkExchange, SimulatedTrader, TradingSimulator

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(threadName)s | %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("plotly").setLevel(logging.WARNING)
    logging.getLogger("Qt").setLevel(logging.WARNING)

    with open("config.json", 'r') as f:
        config = json.load(f)

    exchange = CorrelatedRandomWalkExchange(config)

    currency = "USDT"
    symbol_to_trade = "BTC/USDT"

    algorithm = SimpleAlgorithm(currency=currency, symbol=symbol_to_trade, trade_interval_seconds=60)

    trader = SimulatedTrader(exchange, algorithm)

    # simulation_duration_hours = 2
    # simulation_duration_seconds = simulation_duration_hours * 60 * 60
    simulation_duration = Duration(2, time_unit=TimeUnit.HOUR)
    # max_steps = max(1, int(simulation_duration_seconds / exchange.settings.step_duration_seconds) - 1)

    simulator = TradingSimulator(trader, max_duration=simulation_duration, is_viewer_enabled=True)
    simulator.run()
