import queue
import threading
import time
from datetime import timedelta
from decimal import Decimal
from types import SimpleNamespace
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
    TraderRunnerStopReason,
    SchedulerClock,
)
from pytrader.live_trader_runner import LiveTraderRunner
from pytrader.trader import Strategy, Trader
from pytrader.viewer.models import TraderViewDataUpdate


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
        self._is_ready.set()

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
    def __init__(self, symbol: str):
        self.startup_calls = 0
        self.update_calls = 0
        self.trade_calls = 0
        self.shutdown_calls = 0
        self.last_seen_price: Decimal | None = None

        super().__init__(
            currency="USDT",
            symbols=[symbol],
            trade_interval_seconds=1,
            name="RecordingStrategy",
        )

    def _build(self) -> None:
        return

    def _get_is_ready(self) -> bool:
        return True

    def _get_has_terminated(self) -> bool:
        return False

    def _get_state(self) -> dict[str, Any]:
        return {"updates": self.update_calls, "trades": self.trade_calls}

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


def test_live_runner_processes_queued_messages_and_trades():
    exchange = _build_exchange()
    strategy = RecordingStrategy("BTCUSDT")
    trader = Trader(exchange, strategy)
    scheduler = FixedIntervalScheduler(
        interval_seconds=2.0,
        clock=SchedulerClock.EVENT_TIME,
        fire_immediately=False,
    )
    runner = LiveTraderRunner(trader, scheduler, is_viewer_enabled=False)

    exchange.publish_market_data_message(_trade_message("BTCUSDT", 0, Decimal("50000.0")))
    exchange.publish_market_data_message(_trade_message("BTCUSDT", 1_000, Decimal("50001.0")))
    exchange.publish_market_data_message(_trade_message("BTCUSDT", 2_000, Decimal("50002.0")))
    exchange.publish_market_data_message(_trade_message("BTCUSDT", 3_000, Decimal("50003.0")))
    exchange.publish_market_data_message(_trade_message("BTCUSDT", 4_000, Decimal("50004.0")))

    result = runner.run(
        start_exchange=False,
        stop_exchange=False,
        ready_timeout_sec=None,
        poll_timeout_sec=0.0,
        max_messages=5,
    )

    assert result.stop_reason == TraderRunnerStopReason.MAX_MESSAGES
    assert result.processed_messages == 5
    assert result.routed_messages == 5
    assert result.updates_executed == 5
    assert result.trades_executed == 2
    assert strategy.startup_calls == 1
    assert strategy.update_calls == 5
    assert strategy.trade_calls == 2
    assert strategy.shutdown_calls == 1
    assert strategy.last_seen_price == Decimal("50004.0")


def test_live_runner_returns_ready_timeout_without_startup():
    exchange = _build_exchange()
    strategy = RecordingStrategy("BTCUSDT")
    trader = Trader(exchange, strategy)
    scheduler = FixedIntervalScheduler(interval_seconds=1.0)
    runner = LiveTraderRunner(trader, scheduler, is_viewer_enabled=False)

    result = runner.run(
        start_exchange=False,
        stop_exchange=False,
        ready_timeout_sec=0.01,
        poll_timeout_sec=0.0,
    )

    assert result.stop_reason == TraderRunnerStopReason.READY_TIMEOUT
    assert result.processed_messages == 0
    assert result.updates_executed == 0
    assert result.trades_executed == 0
    assert strategy.startup_calls == 0
    assert strategy.shutdown_calls == 0


def test_live_runner_can_be_stopped_externally():
    exchange = _build_exchange()
    strategy = RecordingStrategy("BTCUSDT")
    trader = Trader(exchange, strategy)
    scheduler = FixedIntervalScheduler(interval_seconds=1.0)
    runner = LiveTraderRunner(trader, scheduler, is_viewer_enabled=False)

    result_holder: dict[str, Any] = {}

    def _run() -> None:
        result_holder["result"] = runner.run(
            start_exchange=False,
            stop_exchange=False,
            ready_timeout_sec=None,
            poll_timeout_sec=0.05,
        )

    thread = threading.Thread(target=_run)
    thread.start()
    time.sleep(0.12)
    runner.request_stop()
    thread.join(timeout=2.0)

    assert thread.is_alive() is False
    result = result_holder["result"]
    assert result.stop_reason == TraderRunnerStopReason.STOP_REQUESTED
    assert strategy.startup_calls == 1
    assert strategy.shutdown_calls == 1


def test_live_runner_hands_off_exchange_market_data_routing_when_supported():
    exchange = _build_exchange()
    strategy = RecordingStrategy("BTCUSDT")
    trader = Trader(exchange, strategy)
    scheduler = FixedIntervalScheduler(interval_seconds=1.0)
    runner = LiveTraderRunner(trader, scheduler, is_viewer_enabled=False)

    handoff_calls = {"count": 0}

    def _handoff() -> bool:
        handoff_calls["count"] += 1
        return True

    exchange.handoff_market_data_routing_to_runner = _handoff  # type: ignore[attr-defined]

    exchange.publish_market_data_message(_trade_message("BTCUSDT", 1_000, Decimal("50000.0")))
    result = runner.run(
        start_exchange=False,
        stop_exchange=False,
        ready_timeout_sec=None,
        poll_timeout_sec=0.0,
        max_messages=1,
    )

    assert result.stop_reason == TraderRunnerStopReason.MAX_MESSAGES
    assert handoff_calls["count"] == 1


def test_live_runner_builds_view_update_only_when_emit_window_opens():
    exchange = _build_exchange()
    strategy = RecordingStrategy("BTCUSDT")
    trader = Trader(exchange, strategy)
    scheduler = FixedIntervalScheduler(interval_seconds=1.0)
    runner = LiveTraderRunner(trader, scheduler, is_viewer_enabled=False)

    calls: list[Any] = []

    class _Feed:
        @staticmethod
        def build_update(message: Any) -> TraderViewDataUpdate:
            calls.append(message)
            return TraderViewDataUpdate(symbol="BTCUSDT", time=timedelta(milliseconds=message.event_ts_ms))

    class _Viewer:
        def __init__(self):
            self.view_update_queue = queue.Queue()

    runner._view_feed = _Feed()
    runner._viewer = _Viewer()

    message_one = _trade_message("BTCUSDT", 1_000, Decimal("50000"))
    runner._pending_view_market_message = message_one
    runner._next_view_update_emit_monotonic_sec = time.monotonic() + 60.0
    runner._flush_pending_view_update()
    assert calls == []
    assert runner._viewer.view_update_queue.qsize() == 0

    message_two = _trade_message("BTCUSDT", 2_000, Decimal("50001"))
    runner._pending_view_market_message = message_two
    runner._flush_pending_view_update(force=True)
    assert calls == [message_two]
    assert runner._viewer.view_update_queue.qsize() == 1
    queued_update = runner._viewer.view_update_queue.get_nowait()
    assert queued_update.time == timedelta(milliseconds=2_000)


def test_live_runner_stop_viewer_skips_close_request_when_viewer_already_stopped():
    exchange = _build_exchange()
    strategy = RecordingStrategy("BTCUSDT")
    trader = Trader(exchange, strategy)
    scheduler = FixedIntervalScheduler(interval_seconds=1.0)
    runner = LiveTraderRunner(trader, scheduler, is_viewer_enabled=False)

    class _Viewer:
        def __init__(self):
            self.close_calls = 0

        def request_browser_close(self):
            self.close_calls += 1
            raise AssertionError("request_browser_close should not be called when viewer already stopped")

    class _Thread:
        def __init__(self):
            self.join_calls = 0
            self._alive = True

        def is_alive(self) -> bool:
            return self._alive

        def join(self, timeout: float | None = None):
            _ = timeout
            self.join_calls += 1
            self._alive = False

    viewer = _Viewer()
    thread = _Thread()

    runner._viewer = viewer
    runner._viewer_thread = thread
    runner._viewer_stopped_event.set()

    runner._stop_viewer()

    assert viewer.close_calls == 0
    assert thread.join_calls == 1
    assert runner._viewer is None
    assert runner._viewer_thread is None


def test_live_runner_stop_viewer_requests_server_shutdown_not_browser_close():
    exchange = _build_exchange()
    strategy = RecordingStrategy("BTCUSDT")
    trader = Trader(exchange, strategy)
    scheduler = FixedIntervalScheduler(interval_seconds=1.0)
    runner = LiveTraderRunner(trader, scheduler, is_viewer_enabled=False)

    class _Viewer:
        def __init__(self):
            self.server_shutdown_calls = 0
            self.browser_close_calls = 0

        def _request_server_shutdown_from_main(self):
            self.server_shutdown_calls += 1
            return True

        def request_browser_close(self):
            self.browser_close_calls += 1
            raise AssertionError("request_browser_close should not be called")

    class _Thread:
        def __init__(self):
            self.join_calls = 0
            self._alive = True

        def is_alive(self) -> bool:
            return self._alive

        def join(self, timeout: float | None = None):
            _ = timeout
            self.join_calls += 1
            self._alive = False

    viewer = _Viewer()
    thread = _Thread()

    runner._viewer = viewer
    runner._viewer_thread = thread
    runner._viewer_stopped_event.clear()

    runner._stop_viewer()

    assert viewer.server_shutdown_calls == 1
    assert viewer.browser_close_calls == 0
    assert thread.join_calls == 1
    assert runner._viewer is None
    assert runner._viewer_thread is None


def test_live_runner_processes_latest_viewer_symbol_request():
    exchange = _build_exchange()
    strategy = RecordingStrategy("BTCUSDT")
    trader = Trader(exchange, strategy)
    scheduler = FixedIntervalScheduler(interval_seconds=1.0)
    runner = LiveTraderRunner(trader, scheduler, is_viewer_enabled=False)

    calls: list[str] = []

    class _Exchange:
        @staticmethod
        def set_view_symbol(symbol: str) -> bool:
            calls.append(symbol)
            return True

    runner._on_viewer_symbol_selected("btcusdt")
    runner._on_viewer_symbol_selected("ethusdt")
    runner._process_viewer_symbol_requests(_Exchange())

    assert calls == ["ETHUSDT"]
    assert runner._last_view_symbol == "ETHUSDT"


def test_live_runner_processes_latest_candle_interval_request_and_primes_selected_symbol():
    exchange = _build_exchange()
    strategy = RecordingStrategy("BTCUSDT")
    trader = Trader(exchange, strategy)
    scheduler = FixedIntervalScheduler(interval_seconds=1.0)
    runner = LiveTraderRunner(trader, scheduler, is_viewer_enabled=False)
    runner._last_view_symbol = "BTCUSDT"

    class _Feed:
        def __init__(self):
            self.candle_interval = "1m"
            self.intervals: list[str] = []

        def set_candle_interval(self, interval: str) -> bool:
            self.intervals.append(interval)
            self.candle_interval = interval
            return True

        @staticmethod
        def build_seed_updates(symbol: str, candles: list[dict[str, Any]]):
            _ = candles
            return [TraderViewDataUpdate(symbol=symbol, time=timedelta(0), frame=None, reset_history=True)]

    class _Viewer:
        def __init__(self):
            self.view_update_queue = queue.Queue()

    class _Exchange:
        def __init__(self):
            self.calls: list[tuple[str, str, int]] = []

        def get_recent_klines(self, *, symbol: str, interval: str, limit: int):
            self.calls.append((symbol, interval, limit))
            return [
                {
                    "open_time_ms": 1_700_000_000_000,
                    "close_time_ms": 1_700_000_059_999,
                    "open": Decimal("50000"),
                    "high": Decimal("50100"),
                    "low": Decimal("49900"),
                    "close": Decimal("50050"),
                }
            ]

    runner._view_feed = _Feed()
    runner._viewer = _Viewer()
    exchange_stub = _Exchange()

    runner._on_viewer_candle_interval_selected("5m")
    runner._on_viewer_candle_interval_selected("15m")
    runner._process_viewer_candle_interval_requests(exchange_stub)

    assert runner._view_feed.intervals == ["15m"]
    assert exchange_stub.calls == [("BTCUSDT", "15m", 240)]
    update = runner._viewer.view_update_queue.get_nowait()
    assert update.reset_history is True
    assert runner._last_view_candle_interval == "15m"


def test_live_runner_warns_when_off_strategy_exposure_exists(caplog):
    exchange = _build_exchange()
    strategy = RecordingStrategy("BTCUSDT")
    trader = Trader(exchange, strategy)
    scheduler = FixedIntervalScheduler(interval_seconds=1.0)
    runner = LiveTraderRunner(trader, scheduler, is_viewer_enabled=False)

    off_strategy_order = Order(
        id="1",
        client_id="off-strategy-order",
        symbol="ETHUSDT",
        type=Order.Type.LIMIT,
        time_in_force=Order.TimeInForce.GTC,
        side=Order.Side.BUY,
        price=Decimal("2500"),
        quantity=Decimal("0.1"),
        timestamp=1_700_000_000_000,
        status=Order.Status.NEW,
    )

    account = SimpleNamespace(
        positions={
            "ETHUSDT": [
                Position(
                    symbol="ETHUSDT",
                    mode=Position.Mode.NET,
                    quantity=Decimal("0.2"),
                    entry_price=Decimal("2600"),
                )
            ]
        },
        orders={"ETHUSDT": [off_strategy_order]},
    )
    exchange.get_account = lambda quote_currency: account

    with caplog.at_level("WARNING"):
        runner._warn_off_strategy_exposure()

    assert "Open positions outside strategy symbols detected: ETHUSDT" in caplog.text
    assert "Open orders outside strategy symbols detected: ETHUSDT" in caplog.text


def test_prepare_viewer_startup_updates_returns_seed_updates_when_ready():
    exchange = _build_exchange()
    strategy = RecordingStrategy("BTCUSDT")
    trader = Trader(exchange, strategy)
    scheduler = FixedIntervalScheduler(interval_seconds=1.0)
    runner = LiveTraderRunner(trader, scheduler, is_viewer_enabled=False)

    ready_frame = SimpleNamespace(market=object(), account=object(), ohlc=object())
    ready_update = SimpleNamespace(frame=ready_frame)

    class _Feed:
        candle_interval = "1m"

        @staticmethod
        def build_seed_updates(symbol: str, candles: list[dict[str, Any]]):
            _ = symbol
            _ = candles
            return [ready_update]

        @staticmethod
        def build_snapshot_update(*, symbol: str, event_ts_ms: int):
            _ = symbol
            _ = event_ts_ms
            return None

    class _Exchange:
        def __init__(self):
            self.selected_symbols: list[str] = []

        def set_view_symbol(self, symbol: str) -> bool:
            self.selected_symbols.append(symbol)
            return True

        @staticmethod
        def get_recent_klines(*, symbol: str, interval: str, limit: int):
            _ = symbol
            _ = interval
            _ = limit
            return [
                {
                    "open_time_ms": 1_700_000_000_000,
                    "close_time_ms": 1_700_000_059_999,
                    "open": Decimal("50000"),
                    "high": Decimal("50100"),
                    "low": Decimal("49900"),
                    "close": Decimal("50050"),
                }
            ]

    runner._view_feed = _Feed()
    exchange_stub = _Exchange()

    updates = runner._prepare_viewer_startup_updates(exchange=exchange_stub, symbol="BTCUSDT")

    assert updates == [ready_update]
    assert exchange_stub.selected_symbols == ["BTCUSDT"]


def test_prepare_viewer_startup_updates_appends_snapshot_when_seed_not_ready():
    exchange = _build_exchange()
    strategy = RecordingStrategy("BTCUSDT")
    trader = Trader(exchange, strategy)
    scheduler = FixedIntervalScheduler(interval_seconds=1.0)
    runner = LiveTraderRunner(trader, scheduler, is_viewer_enabled=False)

    seed_update = SimpleNamespace(frame=SimpleNamespace(market=object(), account=object(), ohlc=None))
    snapshot_update = SimpleNamespace(frame=SimpleNamespace(market=object(), account=object(), ohlc=object()))

    class _Feed:
        candle_interval = "1m"

        @staticmethod
        def build_seed_updates(symbol: str, candles: list[dict[str, Any]]):
            _ = symbol
            _ = candles
            return [seed_update]

        @staticmethod
        def build_snapshot_update(*, symbol: str, event_ts_ms: int):
            _ = symbol
            _ = event_ts_ms
            return snapshot_update

    class _Exchange:
        @staticmethod
        def set_view_symbol(symbol: str) -> bool:
            _ = symbol
            return True

        @staticmethod
        def get_recent_klines(*, symbol: str, interval: str, limit: int):
            _ = symbol
            _ = interval
            _ = limit
            return [
                {
                    "open_time_ms": 1_700_000_000_000,
                    "close_time_ms": 1_700_000_059_999,
                    "open": Decimal("50000"),
                    "high": Decimal("50100"),
                    "low": Decimal("49900"),
                    "close": Decimal("50050"),
                }
            ]

    runner._view_feed = _Feed()

    updates = runner._prepare_viewer_startup_updates(exchange=_Exchange(), symbol="BTCUSDT")

    assert updates == [seed_update, snapshot_update]


def test_prepare_viewer_startup_updates_returns_empty_when_initial_symbol_cannot_be_set():
    exchange = _build_exchange()
    strategy = RecordingStrategy("BTCUSDT")
    trader = Trader(exchange, strategy)
    scheduler = FixedIntervalScheduler(interval_seconds=1.0)
    runner = LiveTraderRunner(trader, scheduler, is_viewer_enabled=False)

    class _Feed:
        candle_interval = "1m"

        @staticmethod
        def build_seed_updates(symbol: str, candles: list[dict[str, Any]]):
            _ = symbol
            _ = candles
            return []

        @staticmethod
        def build_snapshot_update(*, symbol: str, event_ts_ms: int):
            _ = symbol
            _ = event_ts_ms
            return None

    class _Exchange:
        @staticmethod
        def set_view_symbol(symbol: str) -> bool:
            _ = symbol
            return False

    runner._view_feed = _Feed()

    assert runner._prepare_viewer_startup_updates(exchange=_Exchange(), symbol="BTCUSDT") == []
