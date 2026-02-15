from __future__ import annotations

import logging
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Any

from pytrader.exchange import Duration, TimeUnit
from pytrader.trader_runner import (
    ExchangeType,
    TraderRunner,
    TraderRunnerResult,
    TraderRunnerStopReason,
)

_SUPPORTED_CANDLE_INTERVALS: dict[str, int] = {
    "1m": 60,
    "5m": 5 * 60,
    "15m": 15 * 60,
    "1h": 60 * 60,
}


class LiveTraderRunner(TraderRunner[ExchangeType]):
    def __init__(
        self,
        trader,
        scheduler,
        *,
        is_viewer_enabled: bool = True,
        viewer_startup_timeout_sec: float = 30.0,
        viewer_default_candle_interval: str = "1m",
        viewer_candle_prime_limit: int = 240,
        viewer_update_frequency_hz: int = 5,
    ):
        super().__init__(trader, scheduler)

        if viewer_startup_timeout_sec <= 0:
            raise ValueError("'viewer_startup_timeout_sec' must be > 0")
        if viewer_candle_prime_limit < 1:
            raise ValueError("'viewer_candle_prime_limit' must be >= 1")
        if viewer_update_frequency_hz < 1:
            raise ValueError("'viewer_update_frequency_hz' must be >= 1")

        self._logger = logging.getLogger(type(self).__name__)
        self._is_viewer_enabled = bool(is_viewer_enabled)
        self._viewer_startup_timeout_sec = viewer_startup_timeout_sec
        self._viewer_default_candle_interval = self._normalize_candle_interval(viewer_default_candle_interval)
        self._viewer_candle_prime_limit = viewer_candle_prime_limit
        self._viewer_update_frequency_hz = viewer_update_frequency_hz
        self._viewer_update_interval_sec = 1.0 / float(viewer_update_frequency_hz)
        self._next_view_update_emit_monotonic_sec = 0.0
        self._pending_view_update: Any = None
        self._pending_view_market_message: Any = None
        self._viewer: Any = None
        self._viewer_thread: threading.Thread | None = None
        self._viewer_ready_event = threading.Event()
        self._viewer_stopped_event = threading.Event()
        self._viewer_exit_code: int | None = None
        self._viewer_lock = threading.Lock()
        self._view_feed: Any = None
        self._viewer_symbol_queue: queue.Queue[str] = queue.Queue()
        self._viewer_candle_interval_queue: queue.Queue[str] = queue.Queue()
        self._last_view_symbol: str | None = None
        self._last_view_candle_interval: str | None = self._viewer_default_candle_interval

    def run(
        self,
        *,
        ready_timeout_sec: float | None = 30.0,
        poll_timeout_sec: float = 0.25,
        max_runtime_sec: float | None = None,
        max_messages: int | None = None,
        headless: bool = False,
        start_exchange: bool = True,
        stop_exchange: bool = True,
        shutdown_on_exit: bool = True,
        close_viewer_on_exit: bool = True,
    ) -> TraderRunnerResult:
        if ready_timeout_sec is not None and ready_timeout_sec < 0:
            raise ValueError("'ready_timeout_sec' must be non-negative when provided")
        if poll_timeout_sec < 0:
            raise ValueError("'poll_timeout_sec' must be non-negative")
        if max_runtime_sec is not None and max_runtime_sec <= 0:
            raise ValueError("'max_runtime_sec' must be > 0 when provided")
        if max_messages is not None and max_messages < 1:
            raise ValueError("'max_messages' must be >= 1 when provided")

        exchange = self.trader.exchange
        self.scheduler.reset()
        self.clear_stop_request()
        self._last_view_symbol = None
        self._last_view_candle_interval = self._viewer_default_candle_interval
        self._pending_view_update = None
        self._pending_view_market_message = None
        self._next_view_update_emit_monotonic_sec = 0.0
        while True:
            try:
                self._viewer_symbol_queue.get_nowait()
            except queue.Empty:
                break
        while True:
            try:
                self._viewer_candle_interval_queue.get_nowait()
            except queue.Empty:
                break

        processed_messages = 0
        routed_messages = 0
        updates_executed = 0
        trades_executed = 0
        stop_reason = TraderRunnerStopReason.COMPLETED
        trader_started = False
        viewer_started = False
        viewer_enabled = self._is_viewer_enabled and not headless
        viewer_preload_thread: threading.Thread | None = None

        started_at_utc = datetime.now(timezone.utc)
        start_monotonic_sec = time.monotonic()

        if start_exchange:
            exchange.start()

        if viewer_enabled:
            viewer_preload_thread = threading.Thread(
                target=self._prepare_viewer_runtime,
                name="TradingViewerPreloadThread",
                daemon=True,
            )
            viewer_preload_thread.start()

        if ready_timeout_sec is not None:
            if not exchange.wait_until_ready(timeout_sec=ready_timeout_sec):
                stop_reason = TraderRunnerStopReason.READY_TIMEOUT
                ended_at_utc = datetime.now(timezone.utc)
                if stop_exchange:
                    exchange.stop()
                if viewer_preload_thread is not None:
                    viewer_preload_thread.join(timeout=1.0)
                    if viewer_preload_thread.is_alive():
                        self._logger.warning(
                            "Viewer preload thread did not stop during ready-timeout cleanup; skipping viewer teardown"
                        )
                    else:
                        self._stop_viewer()
                else:
                    self._stop_viewer()
                return TraderRunnerResult(
                    stop_reason=stop_reason,
                    started_at_utc=started_at_utc,
                    ended_at_utc=ended_at_utc,
                    processed_messages=processed_messages,
                    routed_messages=routed_messages,
                    updates_executed=updates_executed,
                    trades_executed=trades_executed,
                )

        self.trader.startup()
        trader_started = True
        self._warn_off_strategy_exposure()

        if viewer_enabled:
            viewer_start_monotonic_sec = time.monotonic()
            if not self._start_viewer(exchange=exchange):
                stop_reason = TraderRunnerStopReason.VIEWER_STARTUP_FAILED
                ended_at_utc = datetime.now(timezone.utc)
                if shutdown_on_exit and trader_started:
                    self.trader.shutdown()
                if stop_exchange:
                    exchange.stop()
                if close_viewer_on_exit:
                    self._stop_viewer()
                return TraderRunnerResult(
                    stop_reason=stop_reason,
                    started_at_utc=started_at_utc,
                    ended_at_utc=ended_at_utc,
                    processed_messages=processed_messages,
                    routed_messages=routed_messages,
                    updates_executed=updates_executed,
                    trades_executed=trades_executed,
                )
            self._logger.info(
                "Viewer startup completed in %.2f sec",
                time.monotonic() - viewer_start_monotonic_sec,
            )
            viewer_started = True

        self._handoff_exchange_market_data_routing(exchange)

        try:
            while True:
                if viewer_enabled and self._viewer_stopped_event.is_set():
                    stop_reason = TraderRunnerStopReason.VIEWER_STOPPED
                    break

                if viewer_enabled:
                    self._process_viewer_candle_interval_requests(exchange)
                    self._process_viewer_symbol_requests(exchange)
                    self._flush_pending_view_update()

                if self._stop_requested.is_set():
                    stop_reason = TraderRunnerStopReason.STOP_REQUESTED
                    break

                if max_runtime_sec is not None and (time.monotonic() - start_monotonic_sec) >= max_runtime_sec:
                    stop_reason = TraderRunnerStopReason.MAX_RUNTIME
                    break

                if max_messages is not None and processed_messages >= max_messages:
                    stop_reason = TraderRunnerStopReason.MAX_MESSAGES
                    break

                if start_exchange and not exchange.is_alive():
                    stop_reason = TraderRunnerStopReason.EXCHANGE_STOPPED
                    break

                message = exchange.poll_market_data_message(timeout_sec=poll_timeout_sec)
                if message is None:
                    continue

                processed_messages += 1

                try:
                    routed = exchange.route_market_data_message(message)
                finally:
                    exchange.mark_market_data_message_done()

                if not routed:
                    continue

                routed_messages += 1
                self.trader.update()
                updates_executed += 1

                if viewer_enabled and self._view_feed is not None:
                    active_view_symbol = self._last_view_symbol or self._get_initial_view_symbol()
                    if active_view_symbol is not None and message.symbol == active_view_symbol:
                        self._pending_view_market_message = message
                        if self._viewer is not None:
                            self._flush_pending_view_update()

                if self.trader.has_terminated:
                    stop_reason = TraderRunnerStopReason.STRATEGY_TERMINATED
                    break

                if self.trader.is_ready and self.scheduler.should_trade(event_ts_ms=message.event_ts_ms):
                    self.trader.trade()
                    trades_executed += 1
                    self.scheduler.mark_trade_executed(event_ts_ms=message.event_ts_ms)

                    if self.trader.has_terminated:
                        stop_reason = TraderRunnerStopReason.STRATEGY_TERMINATED
                        break
        finally:
            if viewer_enabled:
                self._flush_pending_view_update(force=True)
            if shutdown_on_exit and trader_started:
                self.trader.shutdown()
            if stop_exchange:
                exchange.stop()
            if close_viewer_on_exit and viewer_started:
                self._stop_viewer()

        ended_at_utc = datetime.now(timezone.utc)
        return TraderRunnerResult(
            stop_reason=stop_reason,
            started_at_utc=started_at_utc,
            ended_at_utc=ended_at_utc,
            processed_messages=processed_messages,
            routed_messages=routed_messages,
            updates_executed=updates_executed,
            trades_executed=trades_executed,
        )

    def on_dash_app_started(self, app: Any) -> None:
        if app is self._viewer:
            self._viewer_ready_event.set()

    def on_dash_app_stopped(self, app: Any, exit_code: int) -> None:
        if app is self._viewer:
            self._viewer_exit_code = exit_code
            self._viewer_stopped_event.set()
            self.request_stop()

    def _prepare_viewer_runtime(self) -> bool:
        with self._viewer_lock:
            if self._viewer is not None and self._view_feed is not None:
                return True

            import_started_at_monotonic_sec = time.monotonic()
            try:
                from pytrader.viewer import (
                    LiveTraderViewFeed,
                    LiveTraderViewFeedConfiguration,
                    TraderViewConfiguration,
                    TradingViewer,
                )
            except Exception:
                self._logger.exception("Failed to import viewer dependencies")
                return False
            self._logger.info(
                "Viewer dependency import completed in %.2f sec",
                time.monotonic() - import_started_at_monotonic_sec,
            )

            build_started_at_monotonic_sec = time.monotonic()
            try:
                self._view_feed = LiveTraderViewFeed(
                    trader=self.trader,
                    config=LiveTraderViewFeedConfiguration(
                        default_candle_interval=self._viewer_default_candle_interval,
                    ),
                )
                self._viewer = TradingViewer(
                    listener=self,
                    config=TraderViewConfiguration(
                        strategy_name=self.trader.strategy.name,
                        symbols=self.trader.strategy.symbols,
                        currency=self.trader.strategy.currency,
                        max_duration=Duration(value=1, time_unit=TimeUnit.DAY),
                        time_unit=TimeUnit.MINUTE,
                        max_candle_window_size=600,
                        view_refresh_frequency_hz=self._viewer_update_frequency_hz,
                        default_candle_interval=self._viewer_default_candle_interval,
                        candle_interval_options=sorted(_SUPPORTED_CANDLE_INTERVALS.keys()),
                    ),
                    on_symbol_selected=self._on_viewer_symbol_selected,
                    on_candle_interval_selected=self._on_viewer_candle_interval_selected,
                )
            except Exception:
                self._logger.exception("Failed to initialize viewer")
                self._viewer = None
                self._view_feed = None
                return False
            self._logger.info(
                "Viewer object construction completed in %.2f sec",
                time.monotonic() - build_started_at_monotonic_sec,
            )

            return True

    def _start_viewer(self, *, exchange: Any) -> bool:
        initial_symbol: str | None = None
        startup_updates: list[Any] = []
        queue_startup_updates: list[Any] = []

        runtime_started_at_monotonic_sec = time.monotonic()
        self._logger.info("Preparing viewer runtime")
        if not self._prepare_viewer_runtime():
            return False
        self._logger.info(
            "Viewer runtime prepared in %.2f sec",
            time.monotonic() - runtime_started_at_monotonic_sec,
        )

        with self._viewer_lock:
            self._viewer_ready_event.clear()
            self._viewer_stopped_event.clear()
            self._viewer_exit_code = None

            if self._viewer is None:
                self._logger.error("Viewer runtime is unavailable")
                return False

            if self._viewer_thread and self._viewer_thread.is_alive():
                return True

            initial_symbol = self._get_initial_view_symbol()
            if initial_symbol is not None:
                bootstrap_started_at_monotonic_sec = time.monotonic()
                self._logger.info("Preparing viewer startup snapshot for symbol '%s'", initial_symbol)
                startup_updates = self._prepare_viewer_startup_updates(exchange=exchange, symbol=initial_symbol)
                if not startup_updates:
                    self._logger.error(
                        "Could not prepare viewer startup snapshot for symbol '%s' within %.2f sec",
                        initial_symbol,
                        self._viewer_startup_timeout_sec,
                    )
                    self._viewer = None
                    self._view_feed = None
                    return False
                self._logger.info(
                    "Viewer startup snapshot for symbol '%s' prepared in %.2f sec (%d updates)",
                    initial_symbol,
                    time.monotonic() - bootstrap_started_at_monotonic_sec,
                    len(startup_updates),
                )

            queue_startup_updates = list(startup_updates)
            apply_startup_updates = getattr(self._viewer, "apply_startup_updates", None)
            if queue_startup_updates and callable(apply_startup_updates):
                try:
                    applied_count = int(apply_startup_updates(queue_startup_updates))
                except Exception:
                    self._logger.warning("Failed applying startup updates to viewer state", exc_info=True)
                else:
                    self._logger.info("Applied %d startup updates directly before viewer launch", applied_count)
                    if applied_count >= len(queue_startup_updates):
                        queue_startup_updates = []
                    elif applied_count > 0:
                        queue_startup_updates = queue_startup_updates[applied_count:]

            self._viewer_thread = threading.Thread(
                target=self._viewer.run_forever,
                name="TradingViewerThread",
                daemon=True,
            )
            self._viewer_thread.start()

        ready_wait_started_at_monotonic_sec = time.monotonic()
        if not self._viewer_ready_event.wait(timeout=self._viewer_startup_timeout_sec):
            self._logger.error("Viewer did not report ready within %.2f sec", self._viewer_startup_timeout_sec)
            self._stop_viewer()
            return False
        self._logger.info(
            "Viewer runtime signaled ready in %.2f sec",
            time.monotonic() - ready_wait_started_at_monotonic_sec,
        )

        self._last_view_candle_interval = self._viewer_default_candle_interval
        self._next_view_update_emit_monotonic_sec = 0.0
        self._pending_view_update = None
        self._pending_view_market_message = None
        if initial_symbol is not None:
            self._last_view_symbol = initial_symbol
        viewer = self._viewer
        if viewer is not None:
            for update in queue_startup_updates:
                viewer.view_update_queue.put(update)

        return True

    def _stop_viewer(self) -> None:
        with self._viewer_lock:
            viewer = self._viewer
            viewer_thread = self._viewer_thread
            viewer_already_stopped = self._viewer_stopped_event.is_set()

        should_request_shutdown = (
            viewer is not None
            and not viewer_already_stopped
            and viewer_thread is not None
            and viewer_thread.is_alive()
        )

        if should_request_shutdown:
            request_server_shutdown = getattr(viewer, "_request_server_shutdown_from_main", None)
            if callable(request_server_shutdown):
                try:
                    request_server_shutdown()
                except Exception:
                    self._logger.warning(
                        "Viewer server shutdown request raised an error; continuing with shutdown",
                        exc_info=True,
                    )

        if viewer_thread is not None and viewer_thread.is_alive():
            viewer_thread.join(timeout=2.0)
            if viewer_thread.is_alive():
                self._logger.warning("Viewer thread did not stop within 2.00 sec; continuing shutdown")

        with self._viewer_lock:
            self._viewer = None
            self._viewer_thread = None
            self._view_feed = None
            self._last_view_symbol = None
            self._last_view_candle_interval = self._viewer_default_candle_interval
            self._pending_view_update = None
            self._pending_view_market_message = None
            self._next_view_update_emit_monotonic_sec = 0.0

    def _handoff_exchange_market_data_routing(self, exchange: Any) -> None:
        handoff = getattr(exchange, "handoff_market_data_routing_to_runner", None)
        if not callable(handoff):
            return

        try:
            handed_off = bool(handoff())
        except Exception:
            self._logger.warning(
                "Failed to hand off exchange market-data routing to LiveTraderRunner",
                exc_info=True,
            )
            return

        if handed_off:
            self._logger.info("Exchange market-data routing handed off to LiveTraderRunner")

    @staticmethod
    def _normalize_symbol(symbol: str | None) -> str | None:
        if not isinstance(symbol, str):
            return None
        normalized = symbol.strip().upper()
        return normalized if normalized else None

    @staticmethod
    def _normalize_candle_interval(interval: str | None) -> str:
        if not isinstance(interval, str):
            raise ValueError("Candle interval must be a string")
        normalized = interval.strip().lower()
        if normalized not in _SUPPORTED_CANDLE_INTERVALS:
            supported = ", ".join(sorted(_SUPPORTED_CANDLE_INTERVALS.keys()))
            raise ValueError(f"Unsupported candle interval '{interval}'. Supported intervals: {supported}")
        return normalized

    def _on_viewer_symbol_selected(self, symbol: str) -> None:
        normalized = self._normalize_symbol(symbol)
        if normalized is None:
            return
        self._viewer_symbol_queue.put(normalized)

    def _on_viewer_candle_interval_selected(self, candle_interval: str) -> None:
        try:
            normalized = self._normalize_candle_interval(candle_interval)
        except ValueError:
            self._logger.warning("Ignoring unsupported viewer candle interval '%s'", candle_interval)
            return
        self._viewer_candle_interval_queue.put(normalized)

    def _process_viewer_candle_interval_requests(self, exchange: Any) -> None:
        selected_interval: str | None = None
        while True:
            try:
                selected_interval = self._viewer_candle_interval_queue.get_nowait()
            except queue.Empty:
                break

        if selected_interval is None or selected_interval == self._last_view_candle_interval:
            return

        view_feed = self._view_feed
        if view_feed is None:
            self._last_view_candle_interval = selected_interval
            return

        set_candle_interval = getattr(view_feed, "set_candle_interval", None)
        interval_changed = False
        if callable(set_candle_interval):
            try:
                interval_changed = bool(set_candle_interval(selected_interval))
            except Exception:
                self._logger.warning(
                    "Failed to apply viewer candle interval '%s'",
                    selected_interval,
                    exc_info=True,
                )
                return
        else:
            interval_changed = True

        if not interval_changed:
            return

        self._last_view_candle_interval = selected_interval
        symbol = self._last_view_symbol or self._get_initial_view_symbol()
        if symbol is not None:
            self._prime_view_symbol_candles(exchange=exchange, symbol=symbol)

    def _flush_pending_view_update(self, *, force: bool = False) -> None:
        viewer = self._viewer
        if viewer is None:
            return

        now = time.monotonic()
        if not force and now < self._next_view_update_emit_monotonic_sec:
            return

        if self._pending_view_update is None and self._pending_view_market_message is not None:
            view_feed = self._view_feed
            if view_feed is not None:
                try:
                    self._pending_view_update = view_feed.build_update(self._pending_view_market_message)
                except Exception:
                    self._logger.warning("Failed to build viewer update from pending market message", exc_info=True)
            self._pending_view_market_message = None

        if self._pending_view_update is None:
            if force:
                self._pending_view_market_message = None
            return

        viewer.view_update_queue.put(self._pending_view_update)
        self._pending_view_update = None
        self._next_view_update_emit_monotonic_sec = now + self._viewer_update_interval_sec

    def _process_viewer_symbol_requests(self, exchange: Any) -> None:
        selected_symbol: str | None = None
        while True:
            try:
                selected_symbol = self._viewer_symbol_queue.get_nowait()
            except queue.Empty:
                break

        if selected_symbol is None or selected_symbol == self._last_view_symbol:
            return

        set_view_symbol = getattr(exchange, "set_view_symbol", None)
        if not callable(set_view_symbol):
            self._last_view_symbol = selected_symbol
            self._prime_view_symbol_candles(exchange=exchange, symbol=selected_symbol)
            return

        try:
            if bool(set_view_symbol(selected_symbol)):
                self._last_view_symbol = selected_symbol
                self._prime_view_symbol_candles(exchange=exchange, symbol=selected_symbol)
        except Exception:
            self._logger.warning(
                "Failed to switch viewer market stream to symbol '%s'",
                selected_symbol,
                exc_info=True,
            )

    def _get_initial_view_symbol(self) -> str | None:
        for symbol in getattr(self.trader.strategy, "symbols", []):
            normalized = self._normalize_symbol(symbol)
            if normalized is not None:
                return normalized
        return None

    def _prime_view_symbol_candles(self, *, exchange: Any, symbol: str) -> None:
        viewer = self._viewer
        if viewer is None:
            return

        updates = self._build_seed_view_updates(exchange=exchange, symbol=symbol)
        if not updates:
            return

        # Drop stale incremental updates so primed snapshot is authoritative.
        self._pending_view_update = None
        self._pending_view_market_message = None
        self._next_view_update_emit_monotonic_sec = 0.0
        for update in updates:
            viewer.view_update_queue.put(update)

    def _prepare_viewer_startup_updates(self, *, exchange: Any, symbol: str) -> list[Any]:
        normalized_symbol = self._normalize_symbol(symbol)
        if normalized_symbol is None:
            return []

        set_view_symbol = getattr(exchange, "set_view_symbol", None)
        if callable(set_view_symbol):
            try:
                if not bool(set_view_symbol(normalized_symbol)):
                    self._logger.warning("Could not set initial view symbol '%s' on exchange", normalized_symbol)
                    return []
            except Exception:
                self._logger.warning(
                    "Failed setting initial view symbol '%s' on exchange",
                    normalized_symbol,
                    exc_info=True,
                )
                return []

        seed_updates = self._build_seed_view_updates(exchange=exchange, symbol=normalized_symbol)
        if seed_updates and self._is_viewer_startup_snapshot_ready(seed_updates[-1]):
            return seed_updates

        deadline_monotonic_sec = time.monotonic() + self._viewer_startup_timeout_sec
        while time.monotonic() < deadline_monotonic_sec:
            snapshot_update = self._build_snapshot_view_update(symbol=normalized_symbol)
            if snapshot_update is None:
                if self._stop_requested.wait(timeout=0.05):
                    return []
                continue
            if self._is_viewer_startup_snapshot_ready(snapshot_update):
                if seed_updates:
                    seed_updates.append(snapshot_update)
                    return seed_updates
                return [snapshot_update]
            if self._stop_requested.wait(timeout=0.05):
                return []

        return []

    def _build_seed_view_updates(self, *, exchange: Any, symbol: str) -> list[Any]:
        view_feed = self._view_feed
        if view_feed is None:
            return []

        get_recent_klines = getattr(exchange, "get_recent_klines", None)
        if not callable(get_recent_klines):
            return []

        candle_interval = getattr(view_feed, "candle_interval", self._viewer_default_candle_interval)
        try:
            candles = get_recent_klines(
                symbol=symbol,
                interval=candle_interval,
                limit=self._viewer_candle_prime_limit,
            )
        except Exception:
            self._logger.warning(
                "Failed to fetch recent klines for symbol '%s' interval '%s'",
                symbol,
                candle_interval,
                exc_info=True,
            )
            return []

        if not isinstance(candles, list) or not candles:
            return []

        build_seed_updates = getattr(view_feed, "build_seed_updates", None)
        if not callable(build_seed_updates):
            return []

        try:
            updates = build_seed_updates(symbol=symbol, candles=candles)
        except Exception:
            self._logger.warning(
                "Failed to build seeded viewer updates for symbol '%s'",
                symbol,
                exc_info=True,
            )
            return []

        if not isinstance(updates, list):
            return []
        return [update for update in updates if update is not None]

    def _build_snapshot_view_update(self, *, symbol: str) -> Any:
        view_feed = self._view_feed
        if view_feed is None:
            return None

        build_snapshot_update = getattr(view_feed, "build_snapshot_update", None)
        if not callable(build_snapshot_update):
            return None

        try:
            return build_snapshot_update(symbol=symbol, event_ts_ms=int(time.time() * 1000))
        except Exception:
            self._logger.warning(
                "Failed to build startup viewer snapshot update for symbol '%s'",
                symbol,
                exc_info=True,
            )
            return None

    @staticmethod
    def _is_viewer_startup_snapshot_ready(update: Any) -> bool:
        if update is None:
            return False

        resolve_frame = getattr(update, "resolve_frame", None)
        if callable(resolve_frame):
            frame = resolve_frame()
        else:
            frame = getattr(update, "frame", None)
        if frame is None:
            return False

        return (
            getattr(frame, "market", None) is not None
            and getattr(frame, "account", None) is not None
            and getattr(frame, "ohlc", None) is not None
        )

    @staticmethod
    def _is_order_open(order: Any) -> bool:
        status = getattr(order, "status", None)
        status_name = getattr(status, "name", None)
        if isinstance(status_name, str):
            normalized = status_name.strip().upper()
        else:
            status_value = getattr(status, "value", status)
            normalized = str(status_value).strip().upper()
        return normalized in {"NEW", "PARTIALLY_FILLED"}

    def _warn_off_strategy_exposure(self) -> None:
        strategy_symbols = {
            normalized
            for normalized in (
                self._normalize_symbol(symbol) for symbol in getattr(self.trader.strategy, "symbols", [])
            )
            if normalized is not None
        }

        try:
            account = self.trader.exchange.get_account(self.trader.strategy.currency)
        except Exception:
            self._logger.debug("Could not fetch account state for off-strategy exposure check", exc_info=True)
            return

        off_strategy_positions: set[str] = set()
        for symbol, positions in account.positions.items():
            normalized_symbol = self._normalize_symbol(symbol)
            if normalized_symbol is None or normalized_symbol in strategy_symbols:
                continue
            if any(getattr(position, "quantity", 0) != 0 for position in positions):
                off_strategy_positions.add(normalized_symbol)

        off_strategy_orders: set[str] = set()
        for symbol, orders in account.orders.items():
            normalized_symbol = self._normalize_symbol(symbol)
            if normalized_symbol is None or normalized_symbol in strategy_symbols:
                continue
            if any(self._is_order_open(order) for order in orders):
                off_strategy_orders.add(normalized_symbol)

        if off_strategy_positions:
            self._logger.warning(
                "Open positions outside strategy symbols detected: %s",
                ", ".join(sorted(off_strategy_positions)),
            )
        if off_strategy_orders:
            self._logger.warning(
                "Open orders outside strategy symbols detected: %s",
                ", ".join(sorted(off_strategy_orders)),
            )
