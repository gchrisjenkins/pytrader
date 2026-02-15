from decimal import Decimal
from typing import Any, Literal

from pytrader.exchange import (
    Account,
    Exchange,
    Market,
    Order,
    Position,
    TradeMarketDataMessage,
    TradePayload,
)
from pytrader.trader_runner import (
    FixedIntervalScheduler,
    ReplayRunner,
    TraderRunnerStopReason,
    SchedulerClock,
)
from pytrader.trader import Strategy, Trader


def _build_market(symbol: str = "BTCUSDT") -> Market:
    settings = Market.Settings(
        symbol=symbol,
        base_asset="BTC",
        quote_asset="USDT",
        margin_asset="USDT",
        tick_size=Decimal("0.1"),
        order_increment=Decimal("0.001"),
        maker_fee=Decimal("0.0002"),
        taker_fee=Decimal("0.0004"),
    )
    return Market(settings)


def _trade_message(symbol: str, event_ts_ms: int, price: Decimal) -> TradeMarketDataMessage:
    return TradeMarketDataMessage(
        provider="test",
        symbol=symbol,
        event_ts_ms=event_ts_ms,
        recv_ts_ms=event_ts_ms,
        sequence=event_ts_ms,
        payload=TradePayload(
            symbol=symbol,
            price=price,
            quantity=Decimal("0.250"),
        ),
    )


class DummyExchange(Exchange["DummyExchange.Configuration", "DummyExchange.Credentials"]):

    class Configuration(Exchange.Configuration):
        provider: Literal["dummy"]

    class Credentials(Exchange.Credentials):
        pass

    class EnvOverrides(Exchange.EnvOverrides):
        pass

    def _startup(self):
        pass

    def _shutdown(self):
        pass

    def get_account(self, quote_currency: str) -> Account:
        raise NotImplementedError

    def get_market_settings(self, symbol: str) -> Market.Settings | None:
        market = self._markets.get(symbol)
        return market.settings if market else None

    def create_market_order(
        self, symbol: str, side: Order.Side, quantity: Decimal, position_side: Position.Side | None = None
    ) -> Order:
        raise NotImplementedError

    def create_limit_order(
        self, symbol: str, side: Order.Side, price: Decimal, quantity: Decimal, position_side: Position.Side | None = None
    ) -> Order:
        raise NotImplementedError

    def cancel_order(self, order: Order) -> Order:
        raise NotImplementedError


def _build_exchange(symbol: str = "BTCUSDT") -> DummyExchange:
    exchange = DummyExchange(
        config=DummyExchange.Configuration(provider="dummy", markets=[symbol]),
        credentials=DummyExchange.Credentials(),
    )
    exchange._markets[symbol] = _build_market(symbol=symbol)
    return exchange


class RecordingStrategy(Strategy):
    def __init__(
        self,
        symbol: str,
        *,
        trade_interval_seconds: int = 2,
        terminate_after_trades: int | None = None,
    ):
        self.startup_calls = 0
        self.update_calls = 0
        self.trade_calls = 0
        self.shutdown_calls = 0
        self.last_seen_price: Decimal | None = None
        self._terminate_after_trades = terminate_after_trades
        self._terminated = False

        super().__init__(
            currency="USDT",
            symbols=[symbol],
            trade_interval_seconds=trade_interval_seconds,
            name="RecordingStrategy",
        )

    def _build(self) -> None:
        return

    def _get_is_ready(self) -> bool:
        return True

    def _get_has_terminated(self) -> bool:
        return self._terminated

    def _get_state(self) -> dict[str, Any]:
        return {
            "updates": self.update_calls,
            "trades": self.trade_calls,
        }

    def _get_action(self) -> dict[str, Any]:
        return {"action": "none"}

    def on_startup(self, exchange: Exchange):
        _ = exchange
        self.startup_calls += 1

    def on_update(self, exchange: Exchange) -> None:
        self.update_calls += 1
        state = exchange.get_market_state(self.symbols[0])
        if state is not None and state.get("last_price") is not None:
            self.last_seen_price = state["last_price"]

    def on_shutdown(self, exchange: Exchange):
        _ = exchange
        self.shutdown_calls += 1

    def on_trade(self, exchange: Exchange):
        _ = exchange
        self.trade_calls += 1
        if self._terminate_after_trades is not None and self.trade_calls >= self._terminate_after_trades:
            self._terminated = True


def test_replay_runner_routes_updates_and_trades_by_event_time():
    exchange = _build_exchange()
    strategy = RecordingStrategy("BTCUSDT")
    trader = Trader(exchange, strategy)
    scheduler = FixedIntervalScheduler(
        interval_seconds=2.0,
        clock=SchedulerClock.EVENT_TIME,
        fire_immediately=False,
    )
    runner = ReplayRunner(trader, scheduler)

    messages = [
        _trade_message("BTCUSDT", 0, Decimal("50000.0")),
        _trade_message("BTCUSDT", 1_000, Decimal("50001.0")),
        _trade_message("BTCUSDT", 2_000, Decimal("50002.0")),
        _trade_message("BTCUSDT", 3_000, Decimal("50003.0")),
        _trade_message("BTCUSDT", 4_000, Decimal("50004.0")),
    ]

    result = runner.run(messages)

    assert result.stop_reason == TraderRunnerStopReason.COMPLETED
    assert result.processed_messages == 5
    assert result.routed_messages == 5
    assert result.updates_executed == 5
    assert result.trades_executed == 2

    assert strategy.startup_calls == 1
    assert strategy.update_calls == 5
    assert strategy.trade_calls == 2
    assert strategy.shutdown_calls == 1
    assert strategy.last_seen_price == Decimal("50004.0")


def test_replay_runner_stops_when_strategy_terminates():
    exchange = _build_exchange()
    strategy = RecordingStrategy("BTCUSDT", terminate_after_trades=1)
    trader = Trader(exchange, strategy)
    scheduler = FixedIntervalScheduler(
        interval_seconds=10.0,
        clock=SchedulerClock.EVENT_TIME,
        fire_immediately=True,
    )
    runner = ReplayRunner(trader, scheduler)

    messages = [
        _trade_message("BTCUSDT", 0, Decimal("50000.0")),
        _trade_message("BTCUSDT", 1_000, Decimal("50001.0")),
    ]

    result = runner.run(messages)

    assert result.stop_reason == TraderRunnerStopReason.STRATEGY_TERMINATED
    assert result.processed_messages == 1
    assert result.routed_messages == 1
    assert result.updates_executed == 1
    assert result.trades_executed == 1
    assert strategy.shutdown_calls == 1


def test_replay_runner_respects_max_messages():
    exchange = _build_exchange()
    strategy = RecordingStrategy("BTCUSDT")
    trader = Trader(exchange, strategy)
    scheduler = FixedIntervalScheduler(
        interval_seconds=2.0,
        clock=SchedulerClock.EVENT_TIME,
        fire_immediately=False,
    )
    runner = ReplayRunner(trader, scheduler)

    messages = [
        _trade_message("BTCUSDT", 0, Decimal("50000.0")),
        _trade_message("BTCUSDT", 1_000, Decimal("50001.0")),
        _trade_message("BTCUSDT", 2_000, Decimal("50002.0")),
        _trade_message("BTCUSDT", 3_000, Decimal("50003.0")),
    ]

    result = runner.run(messages, max_messages=3)

    assert result.stop_reason == TraderRunnerStopReason.MAX_MESSAGES
    assert result.processed_messages == 3
    assert result.routed_messages == 3
    assert result.updates_executed == 3
    assert strategy.shutdown_calls == 1
