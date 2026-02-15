from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Generic, Iterable, TypeVar

from pytrader.exchange import Exchange, MarketDataMessage, NormalizedMarketDataMessage
from pytrader.trader import Trader

ExchangeType = TypeVar("ExchangeType", bound=Exchange)
ReplayMessage = MarketDataMessage | NormalizedMarketDataMessage


class SchedulerClock(str, Enum):
    EVENT_TIME = "event_time"
    MONOTONIC = "monotonic"


class Scheduler(ABC):
    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def should_trade(self, *, event_ts_ms: int | None = None) -> bool:
        pass

    @abstractmethod
    def mark_trade_executed(self, *, event_ts_ms: int | None = None) -> None:
        pass


class FixedIntervalScheduler(Scheduler):
    def __init__(
        self,
        interval_seconds: float,
        *,
        clock: SchedulerClock = SchedulerClock.MONOTONIC,
        fire_immediately: bool = False,
    ):
        if not isinstance(interval_seconds, (int, float)):
            raise TypeError(f"'interval_seconds' must be numeric, but found: {type(interval_seconds)}")
        if interval_seconds <= 0:
            raise ValueError("'interval_seconds' must be > 0")

        self._interval_ms = float(interval_seconds) * 1000.0
        self._clock = clock
        self._fire_immediately = fire_immediately
        self._last_trade_marker_ms: float | None = None

    def reset(self) -> None:
        self._last_trade_marker_ms = None

    def should_trade(self, *, event_ts_ms: int | None = None) -> bool:
        marker_ms = self._resolve_marker_ms(event_ts_ms=event_ts_ms)

        if self._last_trade_marker_ms is None:
            if self._fire_immediately:
                return True
            self._last_trade_marker_ms = marker_ms
            return False

        return (marker_ms - self._last_trade_marker_ms) >= self._interval_ms

    def mark_trade_executed(self, *, event_ts_ms: int | None = None) -> None:
        self._last_trade_marker_ms = self._resolve_marker_ms(event_ts_ms=event_ts_ms)

    def _resolve_marker_ms(self, *, event_ts_ms: int | None) -> float:
        if self._clock == SchedulerClock.EVENT_TIME:
            if event_ts_ms is None:
                raise ValueError("'event_ts_ms' is required when scheduler clock is EVENT_TIME")
            return float(event_ts_ms)
        return time.monotonic() * 1000.0


class TraderRunnerStopReason(str, Enum):
    COMPLETED = "completed"
    STRATEGY_TERMINATED = "strategy_terminated"
    STOP_REQUESTED = "stop_requested"
    MAX_MESSAGES = "max_messages"
    MAX_RUNTIME = "max_runtime"
    READY_TIMEOUT = "ready_timeout"
    EXCHANGE_STOPPED = "exchange_stopped"
    VIEWER_STARTUP_FAILED = "viewer_startup_failed"
    VIEWER_STOPPED = "viewer_stopped"


@dataclass(frozen=True)
class TraderRunnerResult:
    stop_reason: TraderRunnerStopReason
    started_at_utc: datetime
    ended_at_utc: datetime
    processed_messages: int
    routed_messages: int
    updates_executed: int
    trades_executed: int


class TraderRunner(ABC, Generic[ExchangeType]):
    def __init__(self, trader: Trader[ExchangeType], scheduler: Scheduler):
        if trader is None:
            raise ValueError("'trader' is required")
        if scheduler is None:
            raise ValueError("'scheduler' is required")

        self._trader = trader
        self._scheduler = scheduler
        self._stop_requested = threading.Event()

    @property
    def trader(self) -> Trader[ExchangeType]:
        return self._trader

    @property
    def scheduler(self) -> Scheduler:
        return self._scheduler

    def request_stop(self) -> None:
        self._stop_requested.set()

    def clear_stop_request(self) -> None:
        self._stop_requested.clear()

    @abstractmethod
    def run(self, *args, **kwargs) -> TraderRunnerResult:
        pass


class ReplayRunner(TraderRunner[ExchangeType]):
    def run(
        self,
        messages: Iterable[ReplayMessage],
        *,
        max_messages: int | None = None,
        shutdown_on_exit: bool = True,
    ) -> TraderRunnerResult:
        if messages is None:
            raise ValueError("'messages' is required")
        if max_messages is not None and max_messages < 1:
            raise ValueError("'max_messages' must be >= 1 when provided")

        exchange = self.trader.exchange
        self.scheduler.reset()
        self.clear_stop_request()

        processed_messages = 0
        routed_messages = 0
        updates_executed = 0
        trades_executed = 0
        stop_reason = TraderRunnerStopReason.COMPLETED

        started_at_utc = datetime.now(timezone.utc)
        self.trader.startup()

        try:
            for message in messages:
                if self._stop_requested.is_set():
                    stop_reason = TraderRunnerStopReason.STOP_REQUESTED
                    break

                exchange.publish_market_data_message(message)
                processed_messages += 1

                routed_now = exchange.drain_and_route_market_data_messages(max_messages=1, timeout_sec=0.0)
                routed_messages += routed_now

                if routed_now > 0:
                    self.trader.update()
                    updates_executed += 1

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

                if max_messages is not None and processed_messages >= max_messages:
                    stop_reason = TraderRunnerStopReason.MAX_MESSAGES
                    break
        finally:
            if shutdown_on_exit:
                self.trader.shutdown()

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
