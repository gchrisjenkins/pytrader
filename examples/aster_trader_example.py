import argparse
import json
import logging
import signal
import sys
import time
from decimal import Decimal
from typing import Any

from pytrader.exchange import Exchange
from pytrader.live_trader_runner import LiveTraderRunner
from pytrader.trader import Strategy, Trader
from pytrader.trader_runner import (
    FixedIntervalScheduler,
    SchedulerClock,
    TraderRunnerStopReason,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Aster trader example with a no-op strategy.")
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=None,
        help="Optional max runtime in seconds before graceful shutdown.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without starting the TradingViewer window.",
    )
    return parser


def _format_decimal(value: Decimal | None) -> str:
    if value is None:
        return "-"
    return format(value, "f")


def _format_enum_like(value: Any) -> str:
    if value is None:
        return "-"
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


def _market_state_digest(state: dict[str, Any]) -> tuple[Any, ...]:
    return (
        state.get("status"),
        state.get("last_price"),
        state.get("last_quantity"),
        state.get("best_bid"),
        state.get("best_bid_quantity"),
        state.get("best_ask"),
        state.get("best_ask_quantity"),
        state.get("mark_price"),
        state.get("index_price"),
        state.get("funding_rate"),
        state.get("next_funding_time"),
        state.get("open_interest"),
        state.get("open_interest_value"),
        state.get("last_liquidation_side"),
        state.get("last_liquidation_price"),
        state.get("last_liquidation_quantity"),
        state.get("last_market_data_event_ts_ms"),
        state.get("last_market_data_recv_ts_ms"),
        state.get("last_market_data_sequence"),
    )


def _render_market_state(symbol: str, state: dict[str, Any]) -> str:
    status = _format_enum_like(state.get("status"))
    liq_side = _format_enum_like(state.get("last_liquidation_side"))
    return (
        f"[{symbol}] "
        f"status={status} "
        f"last={_format_decimal(state.get('last_price'))} "
        f"last_qty={_format_decimal(state.get('last_quantity'))} "
        f"bid={_format_decimal(state.get('best_bid'))}@{_format_decimal(state.get('best_bid_quantity'))} "
        f"ask={_format_decimal(state.get('best_ask'))}@{_format_decimal(state.get('best_ask_quantity'))} "
        f"mark={_format_decimal(state.get('mark_price'))} "
        f"index={_format_decimal(state.get('index_price'))} "
        f"funding={_format_decimal(state.get('funding_rate'))} "
        f"next_funding={state.get('next_funding_time') if state.get('next_funding_time') is not None else '-'} "
        f"open_interest={_format_decimal(state.get('open_interest'))} "
        f"open_interest_value={_format_decimal(state.get('open_interest_value'))} "
        f"liq={liq_side}@{_format_decimal(state.get('last_liquidation_price'))} "
        f"qty={_format_decimal(state.get('last_liquidation_quantity'))} "
        f"event_ts={state.get('last_market_data_event_ts_ms') if state.get('last_market_data_event_ts_ms') is not None else '-'} "
        f"seq={state.get('last_market_data_sequence') if state.get('last_market_data_sequence') is not None else '-'}"
    )


class NoOpStrategy(Strategy):
    def __init__(self, symbols: list[str], logger: logging.Logger, log_interval_sec: float = 1.0):
        self._logger = logger
        self._log_interval_sec = log_interval_sec
        self._next_log_ts = 0.0
        self._previous_market_digests: dict[str, tuple[Any, ...]] = {}

        super().__init__(
            currency="USDT",
            symbols=symbols,
            trade_interval_seconds=1,
            name="NoOpStrategy",
        )

    def _build(self) -> None:
        return

    def _get_is_ready(self) -> bool:
        return True

    def _get_has_terminated(self) -> bool:
        return False

    def _get_state(self) -> dict[str, Any]:
        return {}

    def _get_action(self) -> dict[str, Any]:
        return {}

    def on_startup(self, exchange: Exchange) -> None:
        _ = exchange
        self._logger.info("Exchange is ready.")
        self._logger.info("Streaming. Press Ctrl+C to stop.")

    def on_shutdown(self, exchange: Exchange) -> None:
        _ = exchange
        self._logger.info("Exchange stopped.")

    def on_trade(self, exchange: Exchange) -> None:
        _ = exchange

    def on_update(self, exchange: Exchange) -> None:
        now = time.monotonic()
        if now < self._next_log_ts:
            return
        self._next_log_ts = now + self._log_interval_sec

        for symbol in self.symbols:
            state = exchange.get_market_state(symbol)
            if state is None:
                continue

            digest = _market_state_digest(state)
            if self._previous_market_digests.get(symbol) != digest:
                self._logger.info(_render_market_state(symbol, state))
                self._previous_market_digests[symbol] = digest


def main() -> int:
    args = _build_parser().parse_args()
    if args.duration_sec is not None and args.duration_sec <= 0:
        raise ValueError("'--duration-sec' must be > 0 when provided")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(threadName)s | %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    logger = logging.getLogger("aster_trader_example")

    config_json = """
    {
      "provider": "aster",
      "rest_url": "https://fapi.asterdex.com",
      "ws_url": "wss://fstream.asterdex.com",
      "markets": ["BTCUSDT"]
    }
    """
    config = json.loads(config_json)

    logger.info(
        "Building trader from inline config. Credentials are loaded from env vars "
        "'aster_api_key' and 'aster_api_secret'.",
    )

    exchange = Exchange.build(config)

    symbols = [str(symbol).strip().upper() for symbol in config.get("markets", []) if str(symbol).strip()]
    if not symbols:
        raise ValueError("No markets configured")

    strategy = NoOpStrategy(symbols=symbols, logger=logger, log_interval_sec=1.0)
    trader = Trader(exchange, strategy)

    # A very long interval keeps trade callbacks effectively dormant for this no-op strategy.
    scheduler = FixedIntervalScheduler(
        interval_seconds=24.0 * 60.0 * 60.0,
        clock=SchedulerClock.EVENT_TIME,
        fire_immediately=False,
    )
    runner = LiveTraderRunner(trader, scheduler)

    shutdown_requested = False

    def request_shutdown(signum: int, _frame: Any):
        nonlocal shutdown_requested
        if shutdown_requested:
            return
        shutdown_requested = True
        signal_name = signal.Signals(signum).name if signum in signal.Signals._value2member_map_ else str(signum)
        logger.info("Received %s. Shutting down exchange...", signal_name)
        runner.request_stop()

    signal.signal(signal.SIGINT, request_shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_shutdown)

    result = runner.run(
        ready_timeout_sec=30.0,
        poll_timeout_sec=0.25,
        max_runtime_sec=args.duration_sec,
        headless=args.headless,
        start_exchange=True,
        stop_exchange=True,
        shutdown_on_exit=True,
    )

    if result.stop_reason == TraderRunnerStopReason.READY_TIMEOUT:
        logger.warning("Exchange did not report ready within 30 seconds.")
        return 1

    if result.stop_reason == TraderRunnerStopReason.EXCHANGE_STOPPED:
        logger.error("Exchange thread exited unexpectedly.")
        return 1

    if result.stop_reason == TraderRunnerStopReason.VIEWER_STARTUP_FAILED:
        logger.error("Viewer failed to start.")
        return 1

    if result.stop_reason == TraderRunnerStopReason.VIEWER_STOPPED:
        logger.info("Viewer was closed. Trader loop stopped.")
        return 0

    if result.stop_reason == TraderRunnerStopReason.MAX_RUNTIME and args.duration_sec is not None:
        logger.info("Duration reached (%.2f sec).", args.duration_sec)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
