import dataclasses
import copy
import json
import logging
import queue
import threading
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

from dash import Input, Output, dcc, html, no_update
from dash.development.base_component import Component

from dashqt import EmbeddedDashApplication, EmbeddedDashApplicationListener
from pytrader.exchange import Duration, TimeUnit
from pytrader.viewer.models import (
    Ohlc,
    StrategyIndicator,
    TraderViewDataUpdate,
    TraderViewFrame,
)

_SUPPORTED_CANDLE_INTERVALS: tuple[str, ...] = ("1m", "5m", "15m", "1h")
_CANDLE_INTERVAL_SECONDS: dict[str, int] = {
    "1m": 60,
    "5m": 5 * 60,
    "15m": 15 * 60,
    "1h": 60 * 60,
}
_SUPPORTED_MARKET_TABS: tuple[str, ...] = ("orderbook", "trades")
_SUPPORTED_BOTTOM_TABS: tuple[str, ...] = ("open_orders", "positions", "assets", "trader")


@dataclasses.dataclass
class TraderViewConfiguration:
    strategy_name: str
    symbols: list[str]
    currency: str
    max_duration: Duration
    time_unit: TimeUnit
    max_candle_window_size: int
    view_refresh_frequency_hz: int = 5
    default_candle_interval: str = "1m"
    candle_interval_options: list[str] | None = None


class TradingViewer(EmbeddedDashApplication):

    def __init__(self,
        listener: EmbeddedDashApplicationListener | None,
        config: TraderViewConfiguration,
        on_symbol_selected: Callable[[str], None] | None = None,
        on_candle_interval_selected: Callable[[str], None] | None = None,
    ):
        assets_folder = Path(__file__).with_name("assets")
        super().__init__(listener, "Trading Viewer", assets_folder=assets_folder)

        cls = type(self)
        self._logger: logging.Logger = logging.getLogger(f"{cls.__module__}.{cls.__name__}")

        self._config: TraderViewConfiguration = self._validate_config(config)
        self._max_history_points: int = self._config.max_candle_window_size

        self._view_update_queue: queue.Queue[TraderViewDataUpdate] = queue.Queue()

        self._view_data_history: dict[str, list[TradingViewer._ViewData]] = {}
        self._latest_frames_by_symbol: dict[str, TraderViewFrame] = {}
        self._latest_account_frame: TraderViewFrame | None = None
        self._on_symbol_selected = on_symbol_selected
        self._on_candle_interval_selected = on_candle_interval_selected
        initial_symbol = self._config.symbols[0].strip().upper()
        initial_interval = self._config.default_candle_interval

        self._view = TradingViewer._View(
            symbol=initial_symbol,
            currency=self._config.currency,
            candle_interval=initial_interval,
            market_tab="orderbook",
            bottom_tab="open_orders",
        )
        self._available_symbols: set[str] = {
            symbol.strip().upper()
            for symbol in self._config.symbols
            if isinstance(symbol, str) and symbol.strip()
        }
        self._available_candle_intervals: list[str] = list(self._config.candle_interval_options or [])

        self._cached_figure: dict[str, Any] | None = None
        self._max_updates_per_drain: int = 200
        self._force_full_render_next_tick: bool = True
        self._startup_monotonic_sec: float = time.monotonic()
        self._first_render_logged: bool = False
        self._update_callback_lock = threading.Lock()
        self._base_figure_layout: dict[str, Any] = {
            "yaxis_title": f"Price ({self._view.currency})",
            "xaxis_title": "Time (UTC)",
            "template": "plotly_dark",
            "margin": {"l": 50, "r": 20, "t": 40, "b": 40},
            "paper_bgcolor": "rgba(17,17,17,1)",
            "plot_bgcolor": "rgba(17,17,17,1)",
            "xaxis": {
                "type": "date",
                "fixedrange": False,
                "rangeslider": {"visible": False},
                "gridcolor": "#444444",
                "linecolor": "#CCCCCC",
                "tickcolor": "#CCCCCC",
                "zerolinecolor": "#444444",
            },
            "yaxis": {"gridcolor": "#444444", "linecolor": "#CCCCCC", "tickcolor": "#CCCCCC", "zerolinecolor": "#444444"},
            "legend": {"font": {"color": "#CCCCCC"}},
            "hovermode": "x unified",
            "hoverlabel": {"font_family": "monospace", "font_size": 11, "align": "left"},
        }

    @property
    def view_update_queue(self) -> queue.Queue[TraderViewDataUpdate]:
        return self._view_update_queue

    def apply_startup_updates(self, updates: list[TraderViewDataUpdate]) -> int:
        applied = 0
        is_view_dirty = False
        for update in updates:
            if update is None:
                continue
            try:
                if self._apply_view_update(update):
                    is_view_dirty = True
            except Exception:
                self._logger.exception("Failed applying startup viewer update for symbol='%s'", update.symbol)
                continue
            applied += 1
        if is_view_dirty:
            self._force_full_render_next_tick = True
        return applied

    # Override dashqt defaults so viewer-owned threads cannot block process exit
    # when Qt shutdown callbacks fail on some PySide builds.
    def _start_server(self) -> bool:
        try:
            self._server.layout = self._build_layout()
            for outputs, inputs, func in self._build_callbacks():
                self._server.callback(outputs, inputs)(func)

            self._server_port = self._find_available_port()
            self._logger.debug("Starting Dash server on 127.0.0.1:%s", self._server_port)

            self._server_thread = threading.Thread(
                target=self._run_server,
                name="DashThread",
                daemon=True,
            )
            self._server_thread.start()
            return self._wait_for_server_ready(max_wait_seconds=15.0, retry_interval_seconds=0.25)
        except Exception:
            self._logger.exception("Failed to start Dash server")
            if self._server_thread and self._server_thread.is_alive():
                self._request_server_shutdown_from_main()
            return False

    # Same rationale as _start_server: keep browser runtime from pinning interpreter exit.
    def _start_browser(self) -> bool:
        try:
            self._browser_thread = threading.Thread(
                target=self._run_browser,
                name="EmbeddedBrowserThread",
                daemon=True,
            )
            self._browser_thread.start()
            return True
        except Exception:
            self._logger.exception("Failed to start browser thread")
            return False


    def _build_layout(self) -> Component | list[Component]:
        self._logger.debug("Building viewer layout")
        return html.Div(
            style={
                "backgroundColor": "#111111",
                "color": "#E2E8F0",
                "margin": "0",
                "padding": "8px 10px",
                "height": "100vh",
                "display": "flex",
                "flexDirection": "column",
                "gap": "10px",
                "fontFamily": "'IBM Plex Sans', 'Segoe UI', sans-serif",
            },
            children=[
                dcc.Interval(
                    id="viewer-update-interval",
                    interval=1000.0 / self._config.view_refresh_frequency_hz,
                    n_intervals=0,
                ),
                html.Div(
                    style={
                        "display": "flex",
                        "alignItems": "stretch",
                        "justifyContent": "space-between",
                        "gap": "12px",
                        "backgroundColor": "#14171C",
                        "border": "1px solid #242A33",
                        "borderRadius": "6px",
                        "padding": "8px 10px",
                    },
                    children=[
                        html.Div(
                            style={"display": "flex", "alignItems": "center", "gap": "10px"},
                            children=[
                                html.Div(
                                    id="viewer-title",
                                    children=[self._build_viewer_title()],
                                    style={"fontWeight": "600", "fontSize": "16px", "whiteSpace": "nowrap"},
                                ),
                                html.Label("Symbol", htmlFor="viewer-symbol-selector", style={"color": "#94A3B8"}),
                                dcc.Dropdown(
                                    id="viewer-symbol-selector",
                                    options=self._build_symbol_options(),
                                    value=self._view.symbol,
                                    clearable=False,
                                    searchable=True,
                                    style={"minWidth": "180px", "color": "#111111"},
                                ),
                                html.Label("Candle", htmlFor="viewer-candle-interval-selector", style={"color": "#94A3B8"}),
                                dcc.Dropdown(
                                    id="viewer-candle-interval-selector",
                                    options=self._build_candle_interval_options(),
                                    value=self._view.candle_interval,
                                    clearable=False,
                                    searchable=False,
                                    style={"width": "120px", "color": "#111111"},
                                ),
                            ],
                        ),
                        html.Div(
                            id="viewer-market-strip",
                            style={
                                "display": "grid",
                                "gridTemplateColumns": "repeat(6, minmax(96px, auto))",
                                "gap": "8px",
                                "alignItems": "center",
                            },
                        ),
                    ],
                ),
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "minmax(0, 3.1fr) minmax(320px, 1fr)",
                        "gap": "10px",
                        "flex": "1 1 auto",
                        "minHeight": "0",
                    },
                    children=[
                        html.Div(
                            style={
                                "display": "flex",
                                "flexDirection": "column",
                                "minHeight": "0",
                                "backgroundColor": "#14171C",
                                "border": "1px solid #242A33",
                                "borderRadius": "6px",
                                "padding": "6px",
                                "gap": "6px",
                            },
                            children=[
                                dcc.Graph(
                                    id="viewer-candlestick-graph",
                                    style={"flex": "1 1 auto", "minHeight": "0", "touchAction": "none"},
                                    config={
                                        "scrollZoom": True,
                                        "doubleClick": "reset",
                                        "displaylogo": False,
                                        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                                    },
                                ),
                                html.Pre(
                                    id="viewer-status-text",
                                    style={
                                        "margin": "0",
                                        "padding": "6px 8px",
                                        "fontFamily": "monospace",
                                        "fontSize": "11px",
                                        "border": "1px solid #262C35",
                                        "borderRadius": "4px",
                                        "color": "#94A3B8",
                                        "backgroundColor": "#101319",
                                    },
                                ),
                            ],
                        ),
                        html.Div(
                            style={
                                "display": "flex",
                                "flexDirection": "column",
                                "gap": "10px",
                                "minHeight": "0",
                            },
                            children=[
                                html.Div(
                                    style={
                                        "display": "flex",
                                        "flexDirection": "column",
                                        "minHeight": "0",
                                        "flex": "1 1 58%",
                                        "backgroundColor": "#14171C",
                                        "border": "1px solid #242A33",
                                        "borderRadius": "6px",
                                        "overflow": "hidden",
                                    },
                                    children=[
                                        dcc.Tabs(
                                            id="viewer-market-tabs",
                                            value=self._view.market_tab,
                                            children=[
                                                dcc.Tab(label="Order Book", value="orderbook"),
                                                dcc.Tab(label="Trades", value="trades"),
                                            ],
                                            colors={
                                                "border": "#242A33",
                                                "primary": "#2F7EF7",
                                                "background": "#14171C",
                                            },
                                        ),
                                        html.Div(
                                            id="viewer-market-tab-content",
                                            style={
                                                "flex": "1 1 auto",
                                                "minHeight": "0",
                                                "overflow": "auto",
                                                "padding": "8px",
                                            },
                                        ),
                                    ],
                                ),
                                html.Div(
                                    style={
                                        "display": "flex",
                                        "flexDirection": "column",
                                        "flex": "1 1 42%",
                                        "minHeight": "0",
                                        "backgroundColor": "#14171C",
                                        "border": "1px solid #242A33",
                                        "borderRadius": "6px",
                                    },
                                    children=[
                                        html.Div(
                                            "Account",
                                            style={
                                                "padding": "8px 10px 0 10px",
                                                "fontWeight": "600",
                                                "fontSize": "13px",
                                            },
                                        ),
                                        html.Div(
                                            id="viewer-account-panel",
                                            style={
                                                "flex": "1 1 auto",
                                                "padding": "8px 10px 10px 10px",
                                                "overflow": "auto",
                                            },
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    style={
                        "display": "flex",
                        "flexDirection": "column",
                        "minHeight": "220px",
                        "maxHeight": "34vh",
                        "backgroundColor": "#14171C",
                        "border": "1px solid #242A33",
                        "borderRadius": "6px",
                        "overflow": "hidden",
                    },
                    children=[
                        dcc.Tabs(
                            id="viewer-bottom-tabs",
                            value=self._view.bottom_tab,
                            children=[
                                dcc.Tab(label="Open Orders", value="open_orders"),
                                dcc.Tab(label="Positions", value="positions"),
                                dcc.Tab(label="Assets", value="assets"),
                                dcc.Tab(label="Trader", value="trader"),
                            ],
                            colors={
                                "border": "#242A33",
                                "primary": "#2F7EF7",
                                "background": "#14171C",
                            },
                        ),
                        html.Div(
                            id="viewer-bottom-tab-content",
                            style={
                                "flex": "1 1 auto",
                                "minHeight": "0",
                                "padding": "8px",
                                "overflow": "auto",
                            },
                        ),
                    ],
                ),
            ],
        )

    def _build_callbacks(self) -> list[tuple]:
        self._logger.debug("Building TradingViewer callbacks")
        return [(
            (
                Output("viewer-title", "children"),
                Output("viewer-candlestick-graph", "figure"),
                Output("viewer-status-text", "children"),
                Output("viewer-market-strip", "children"),
                Output("viewer-account-panel", "children"),
                Output("viewer-market-tab-content", "children"),
                Output("viewer-bottom-tab-content", "children"),
                Output("viewer-symbol-selector", "options"),
                Output("viewer-symbol-selector", "value"),
                Output("viewer-candle-interval-selector", "options"),
                Output("viewer-candle-interval-selector", "value"),
            ),
            [
                Input("viewer-update-interval", "n_intervals"),
                Input("viewer-symbol-selector", "value"),
                Input("viewer-candle-interval-selector", "value"),
                Input("viewer-market-tabs", "value"),
                Input("viewer-bottom-tabs", "value"),
            ],
            self._update_view
        )]

    def _update_view(
        self,
        n_intervals: int,
        selected_symbol: str | None,
        selected_candle_interval: str | None,
        selected_market_tab: str | None,
        selected_bottom_tab: str | None,
    ):
        if not self._update_callback_lock.acquire(blocking=False):
            return self._no_update_result()

        callback_started_at_monotonic_sec = time.monotonic()
        try:
            _ = n_intervals

            selection_changed = self._set_selected_symbol(selected_symbol)
            interval_changed = self._set_selected_candle_interval(selected_candle_interval)
            market_tab_changed = self._set_selected_market_tab(selected_market_tab)
            bottom_tab_changed = self._set_selected_bottom_tab(selected_bottom_tab)
            queue_depth_before_drain = self._view_update_queue.qsize()
            drain_started_at_monotonic_sec = time.monotonic()
            is_view_dirty, drained_updates = self._drain_view_update_queue()
            drain_duration_sec = time.monotonic() - drain_started_at_monotonic_sec
            force_render = self._force_full_render_next_tick
            if force_render:
                self._force_full_render_next_tick = False

            selected_symbol_value = self._view.symbol
            latest = (
                self._latest_frames_by_symbol.get(selected_symbol_value)
                if selected_symbol_value is not None
                else None
            )
            history = (
                self._view_data_history.get(selected_symbol_value, [])
                if selected_symbol_value is not None
                else []
            )
            history_size = len(history)

            if not force_render and not is_view_dirty and not selection_changed and not interval_changed:
                if not market_tab_changed and not bottom_tab_changed:
                    if not self._first_render_logged:
                        if selected_symbol_value is not None and (latest is not None or history_size > 0):
                            # Startup updates may already be applied directly, so force one render.
                            force_render = True
                        else:
                            return self._no_update_result()
                    else:
                        return self._no_update_result()

            if selected_symbol_value is None:
                return self._no_update_result()

            account_frame = self._latest_account_frame if self._latest_account_frame is not None else latest
            data_window = [
                point for point in history[-self._config.max_candle_window_size:]
                if point.ohlc is not None
            ]

            if latest is not None:
                self._view.time = latest.time

            figure_started_at_monotonic_sec = time.monotonic()
            figure_output: dict[str, Any] | Any = no_update
            status_text = "Waiting for market data..."
            if data_window:
                figure_output = self._build_figure(data_window)
                self._cached_figure = figure_output
                self._view.data_window = data_window
                unit = self._config.time_unit.abbreviation
                last_ohlc = data_window[-1].ohlc
                status_text = (
                    f"Time: {self._view.time.total_seconds() / self._config.time_unit.seconds:.1f} {unit}\n"
                    f"Last Price: {self._format_decimal(last_ohlc.close)}"
                )
            elif self._cached_figure is not None:
                figure_output = self._cached_figure
            elif selection_changed or interval_changed or force_render:
                # Clear stale chart data immediately when switching symbols, and ensure startup renders.
                figure_output = self._empty_figure()
            figure_duration_sec = time.monotonic() - figure_started_at_monotonic_sec

            if not self._first_render_logged:
                self._first_render_logged = True
                self._logger.info(
                    "Viewer first render completed in %.2f sec (symbol=%s, history=%d, latest=%s, queue=%d)",
                    time.monotonic() - self._startup_monotonic_sec,
                    selected_symbol_value,
                    len(data_window),
                    "yes" if latest is not None else "no",
                    self._view_update_queue.qsize(),
                )

            format_started_at_monotonic_sec = time.monotonic()
            result = (
                self._build_viewer_title(),
                figure_output,
                status_text,
                self._build_market_strip(latest),
                self._build_account_panel(account_frame),
                self._build_market_tab_content(self._view.market_tab, latest),
                self._build_bottom_tab_content(self._view.bottom_tab, account_frame, latest),
                self._build_symbol_options(),
                self._view.symbol,
                self._build_candle_interval_options(),
                self._view.candle_interval,
            )
            format_duration_sec = time.monotonic() - format_started_at_monotonic_sec

            callback_duration_sec = time.monotonic() - callback_started_at_monotonic_sec
            if callback_duration_sec >= 0.5:
                self._logger.warning(
                    "Slow viewer callback: %.3f sec (symbol=%s, drained=%s/%d, queue_before=%d, queue_after=%d, drain=%.3f sec, figure=%.3f sec, format=%.3f sec, history=%d)",
                    callback_duration_sec,
                    selected_symbol_value,
                    "yes" if is_view_dirty else "no",
                    drained_updates,
                    queue_depth_before_drain,
                    self._view_update_queue.qsize(),
                    drain_duration_sec,
                    figure_duration_sec,
                    format_duration_sec,
                    len(data_window),
                )

            return result
        finally:
            self._update_callback_lock.release()

    def _drain_view_update_queue(self) -> tuple[bool, int]:
        is_view_dirty = False
        processed = 0
        while processed < self._max_updates_per_drain:
            try:
                view_update = self._view_update_queue.get_nowait()
            except queue.Empty:
                break
            processed += 1
            try:
                if self._apply_view_update(view_update):
                    is_view_dirty = True
            except Exception:
                self._logger.exception("Error processing viewer update for symbol='%s'", view_update.symbol)
            finally:
                self._view_update_queue.task_done()
        if processed >= self._max_updates_per_drain:
            self._logger.debug(
                "Viewer update drain capped at %d updates; remaining updates will be processed next tick",
                self._max_updates_per_drain,
            )
        return is_view_dirty, processed

    def _apply_view_update(self, view_update: TraderViewDataUpdate) -> bool:
        symbol = view_update.symbol.strip().upper()
        if view_update.reset_history:
            if view_update.seed_points:
                seeded_history: list[TradingViewer._ViewData] = []
                for event_ts_ms, ohlc in view_update.seed_points:
                    seeded_history.append(
                        TradingViewer._ViewData(
                            time=timedelta(milliseconds=event_ts_ms),
                            ohlc=self._copy_ohlc(ohlc),
                            indicators=[],
                        )
                    )
                self._view_data_history[symbol] = seeded_history
            else:
                self._view_data_history[symbol] = []
            self._cached_figure = None

        if view_update.frame is None and not view_update.data:
            self._available_symbols.add(symbol)
            return True

        frame = view_update.resolve_frame()
        symbol = frame.symbol.strip().upper()

        self._latest_account_frame = frame
        self._latest_frames_by_symbol[symbol] = frame
        self._register_symbols_from_frame(frame)
        history = self._view_data_history.setdefault(symbol, [])

        if frame.ohlc is not None:
            if history and history[-1].ohlc is not None and history[-1].ohlc.time == frame.ohlc.time:
                history[-1].time = frame.time
                history[-1].ohlc = self._copy_ohlc(frame.ohlc)
                history[-1].indicators = list(frame.indicators)
            else:
                history.append(
                    TradingViewer._ViewData(
                        time=frame.time,
                        ohlc=self._copy_ohlc(frame.ohlc),
                        indicators=list(frame.indicators),
                    )
                )

        self._prune_history(symbol, frame.time)
        return True

    def _prune_history(self, symbol: str, current_time: timedelta) -> None:
        history = self._view_data_history.get(symbol)
        if history is None:
            return

        max_age = timedelta(seconds=self._config.max_duration.seconds)
        min_time = current_time - max_age
        while history and history[0].time < min_time:
            history.pop(0)

        if len(history) > self._max_history_points:
            del history[:-self._max_history_points]

    @staticmethod
    def _validate_config(config: TraderViewConfiguration) -> TraderViewConfiguration:
        normalized_symbols = [
            symbol.strip().upper()
            for symbol in config.symbols
            if isinstance(symbol, str) and symbol.strip()
        ]
        if not normalized_symbols:
            raise ValueError("'symbols' cannot be empty")
        config.symbols = normalized_symbols
        if config.max_candle_window_size < 1:
            raise ValueError("'max_candle_window_size' must be >= 1")
        if config.view_refresh_frequency_hz < 1:
            raise ValueError("'view_refresh_frequency_hz' must be >= 1")
        if config.max_duration.seconds <= 0:
            raise ValueError("'max_duration' must be > 0")
        if config.time_unit.seconds <= 0:
            raise ValueError("'time_unit' must be > 0")
        normalized_default_interval = TradingViewer._normalize_candle_interval(config.default_candle_interval)
        if config.candle_interval_options is None:
            config.candle_interval_options = list(_SUPPORTED_CANDLE_INTERVALS)
        else:
            normalized_options = [
                TradingViewer._normalize_candle_interval(interval)
                for interval in config.candle_interval_options
            ]
            config.candle_interval_options = sorted(set(normalized_options), key=normalized_options.index)
        if normalized_default_interval not in config.candle_interval_options:
            config.candle_interval_options.insert(0, normalized_default_interval)
        config.default_candle_interval = normalized_default_interval
        return config

    def _build_viewer_title(self) -> str:
        symbol = self._view.symbol if self._view.symbol is not None else "-"
        return f"{self._config.strategy_name} ({symbol})"

    def _build_symbol_options(self) -> list[dict[str, str]]:
        return [{"label": symbol, "value": symbol} for symbol in sorted(self._available_symbols)]

    def _build_candle_interval_options(self) -> list[dict[str, str]]:
        return [{"label": interval.upper(), "value": interval} for interval in self._available_candle_intervals]

    @staticmethod
    def _normalize_candle_interval(value: str | None) -> str:
        if not isinstance(value, str):
            raise ValueError("Candle interval must be a string")
        normalized = value.strip().lower()
        if normalized not in _SUPPORTED_CANDLE_INTERVALS:
            raise ValueError(
                f"Unsupported candle interval '{value}'. Supported intervals: {', '.join(_SUPPORTED_CANDLE_INTERVALS)}"
            )
        return normalized

    def _set_selected_symbol(self, selected_symbol: str | None) -> bool:
        if selected_symbol is None:
            return False
        if not isinstance(selected_symbol, str):
            return False
        normalized = selected_symbol.strip().upper()
        if not normalized:
            return False
        self._available_symbols.add(normalized)
        if self._view.symbol == normalized:
            return False
        self._view.symbol = normalized
        self._cached_figure = None
        if self._on_symbol_selected is not None:
            try:
                self._on_symbol_selected(normalized)
            except Exception:
                self._logger.exception("Failed to notify selected symbol '%s'", normalized)
        return True

    def _set_selected_candle_interval(self, selected_interval: str | None) -> bool:
        if selected_interval is None:
            return False

        try:
            normalized = self._normalize_candle_interval(selected_interval)
        except ValueError:
            self._logger.warning("Ignoring unsupported candle interval '%s'", selected_interval)
            return False

        if normalized == self._view.candle_interval:
            return False

        self._view.candle_interval = normalized
        self._cached_figure = None
        self._view_data_history.clear()
        if normalized not in self._available_candle_intervals:
            self._available_candle_intervals.append(normalized)

        if self._on_candle_interval_selected is not None:
            try:
                self._on_candle_interval_selected(normalized)
            except Exception:
                self._logger.exception("Failed to notify selected candle interval '%s'", normalized)
        return True

    def _set_selected_market_tab(self, selected_tab: str | None) -> bool:
        if selected_tab not in _SUPPORTED_MARKET_TABS:
            return False
        if self._view.market_tab == selected_tab:
            return False
        self._view.market_tab = selected_tab
        return True

    def _set_selected_bottom_tab(self, selected_tab: str | None) -> bool:
        if selected_tab not in _SUPPORTED_BOTTOM_TABS:
            return False
        if self._view.bottom_tab == selected_tab:
            return False
        self._view.bottom_tab = selected_tab
        return True

    def _register_symbols_from_frame(self, frame: TraderViewFrame) -> None:
        self._available_symbols.add(frame.symbol.strip().upper())
        account = frame.account
        if account is None:
            return
        for position in account.positions:
            symbol = position.symbol.strip().upper()
            if symbol:
                self._available_symbols.add(symbol)
        for order in account.orders:
            symbol = order.symbol.strip().upper()
            if symbol:
                self._available_symbols.add(symbol)

    @staticmethod
    def _copy_ohlc(ohlc: Ohlc) -> Ohlc:
        return Ohlc(
            time=ohlc.time,
            open=ohlc.open,
            high=ohlc.high,
            low=ohlc.low,
            close=ohlc.close,
        )

    @staticmethod
    def _build_panel(title: str, body_id: str) -> html.Div:
        return html.Div(
            style={
                "border": "1px solid #333333",
                "borderRadius": "4px",
                "padding": "8px",
                "backgroundColor": "#151515",
            },
            children=[
                html.Div(title, style={"fontWeight": "bold", "marginBottom": "6px"}),
                html.Pre(
                    id=body_id,
                    style={
                        "margin": "0",
                        "fontFamily": "monospace",
                        "fontSize": "11px",
                        "whiteSpace": "pre-wrap",
                    },
                ),
            ],
        )

    @staticmethod
    def _format_decimal(value: Decimal | None) -> str:
        if value is None:
            return "-"
        return format(value, "f")

    @staticmethod
    def _format_enum_like(value: Any) -> str:
        if value is None:
            return "-"
        enum_value = getattr(value, "value", None)
        if enum_value is not None:
            return str(enum_value)
        return str(value)

    def _empty_figure(self) -> dict[str, Any]:
        layout = copy.deepcopy(self._base_figure_layout)
        layout["yaxis_title"] = f"Price ({self._view.currency})"
        layout["xaxis_title"] = "Time (UTC)"
        layout["xaxis"]["tickformat"] = self._xaxis_tickformat_for_interval()
        layout["xaxis"]["hoverformat"] = "%Y-%m-%d %H:%M"
        layout["uirevision"] = self._figure_uirevision()
        return {
            "data": [],
            "layout": layout,
        }

    def _build_figure(self, data_window: list["TradingViewer._ViewData"]) -> dict[str, Any]:
        candle_interval_seconds = _CANDLE_INTERVAL_SECONDS[self._view.candle_interval]
        time_values = [
            datetime.fromtimestamp(data.ohlc.time * candle_interval_seconds, tz=timezone.utc).isoformat()
            for data in data_window
        ]
        ohlc_data = {
            "time": time_values,
            "open": [float(data.ohlc.open) for data in data_window],
            "high": [float(data.ohlc.high) for data in data_window],
            "low": [float(data.ohlc.low) for data in data_window],
            "close": [float(data.ohlc.close) for data in data_window],
        }

        hover_texts = [
            (
                f"<b>Open:</b> {open_:.2f}<br><b>High:</b> {high_:.2f}<br>"
                f"<b>Low:</b> {low_:.2f}<br><b>Close:</b> {close_:.2f}<br>"
                f"<b>Time:</b> {time_}"
            )
            for time_, open_, high_, low_, close_ in zip(
                ohlc_data["time"],
                ohlc_data["open"],
                ohlc_data["high"],
                ohlc_data["low"],
                ohlc_data["close"],
            )
        ]

        traces: list[dict[str, Any]] = [{
            "type": "candlestick",
            "x": ohlc_data["time"],
            "open": ohlc_data["open"],
            "high": ohlc_data["high"],
            "low": ohlc_data["low"],
            "close": ohlc_data["close"],
            "name": self._view.symbol,
            "hovertext": hover_texts,
            "hoverinfo": "text",
        }]

        layout = copy.deepcopy(self._base_figure_layout)
        layout["yaxis_title"] = f"Price ({self._view.currency})"
        layout["xaxis_title"] = "Time (UTC)"
        layout["xaxis"]["tickformat"] = self._xaxis_tickformat_for_interval()
        layout["xaxis"]["hoverformat"] = "%Y-%m-%d %H:%M"
        layout["uirevision"] = self._figure_uirevision()
        return {
            "data": traces,
            "layout": layout,
        }

    def _figure_uirevision(self) -> str:
        symbol = self._view.symbol or "-"
        return f"{symbol}:{self._view.candle_interval}"

    def _xaxis_tickformat_for_interval(self) -> str:
        interval_seconds = _CANDLE_INTERVAL_SECONDS[self._view.candle_interval]
        # Values after '\n' are only shown when they change, so month labels appear at month boundaries.
        if interval_seconds <= 15 * 60:
            return "%H:%M\n%b"
        return "%H\n%b"

    @staticmethod
    def _compact_value(value: Decimal | int | float | str | None, places: int = 4) -> str:
        if value is None:
            return "-"
        if isinstance(value, Decimal):
            return f"{value:.{places}f}"
        if isinstance(value, (int, float)):
            return f"{value:.{places}f}".rstrip("0").rstrip(".")
        return str(value)

    @staticmethod
    def _metric_card(title: str, value: str) -> html.Div:
        return html.Div(
            style={
                "backgroundColor": "#101319",
                "border": "1px solid #242A33",
                "borderRadius": "4px",
                "padding": "4px 8px",
                "lineHeight": "1.25",
            },
            children=[
                html.Div(title, style={"fontSize": "10px", "color": "#94A3B8"}),
                html.Div(value, style={"fontSize": "12px", "fontWeight": "600", "color": "#F8FAFC"}),
            ],
        )

    @staticmethod
    def _table_styles() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
        table_style = {
            "width": "100%",
            "borderCollapse": "collapse",
            "fontSize": "12px",
            "lineHeight": "1.35",
        }
        header_cell_style = {
            "textAlign": "left",
            "fontWeight": "600",
            "padding": "8px 8px",
            "color": "#94A3B8",
            "borderBottom": "1px solid #242A33",
            "position": "sticky",
            "top": "0",
            "backgroundColor": "#14171C",
            "zIndex": "1",
            "whiteSpace": "nowrap",
        }
        body_cell_style = {
            "padding": "8px 8px",
            "borderBottom": "1px solid #1C212A",
            "whiteSpace": "nowrap",
            "color": "#E2E8F0",
        }
        return table_style, header_cell_style, body_cell_style

    def _build_table(self, *, headers: list[str], rows: list[list[str]], empty_text: str) -> Component:
        if not rows:
            return html.Div(empty_text, style={"color": "#94A3B8", "padding": "10px 6px"})

        table_style, header_cell_style, body_cell_style = self._table_styles()
        return html.Table(
            style=table_style,
            children=[
                html.Thead(
                    html.Tr([html.Th(header, style=header_cell_style) for header in headers])
                ),
                html.Tbody(
                    [
                        html.Tr([html.Td(value, style=body_cell_style) for value in row])
                        for row in rows
                    ]
                ),
            ],
        )

    def _build_market_strip(self, frame: TraderViewFrame | None) -> list[Component]:
        if frame is None or frame.market is None:
            return [self._metric_card("Status", "Waiting for market data")]

        market = frame.market
        return [
            self._metric_card("Last", self._compact_value(market.last_price, places=2)),
            self._metric_card("Mark", self._compact_value(market.mark_price, places=2)),
            self._metric_card("Index", self._compact_value(market.index_price, places=2)),
            self._metric_card("Funding", self._compact_value(market.funding_rate, places=6)),
            self._metric_card("Open Interest", self._compact_value(market.open_interest, places=3)),
            self._metric_card("OI Value", self._compact_value(market.open_interest_value, places=2)),
        ]

    def _build_account_panel(self, frame: TraderViewFrame | None) -> Component:
        if frame is None or frame.account is None:
            return html.Div("No account data yet.", style={"color": "#94A3B8"})

        account = frame.account
        rows = [
            ("Quote Asset", account.quote_asset),
            ("Margin Asset", account.margin_asset),
            ("Balance", self._compact_value(account.balance, places=6)),
            ("Equity", self._compact_value(account.equity, places=6)),
            ("Unrealized PnL", self._compact_value(account.unrealized_pnl, places=6)),
            ("Available Margin", self._compact_value(account.available_margin_balance, places=6)),
            ("Initial Margin", self._compact_value(account.initial_margin_requirement, places=6)),
            ("Maintenance Margin", self._compact_value(account.maintenance_margin_requirement, places=6)),
            ("Open Positions", str(len(account.positions))),
            ("Open Orders", str(len(account.orders))),
        ]
        return html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "6px 10px"},
            children=[
                html.Div(
                    style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "borderBottom": "1px dotted #252B35",
                        "paddingBottom": "4px",
                    },
                    children=[
                        html.Span(label, style={"color": "#94A3B8", "fontSize": "11px"}),
                        html.Span(value, style={"color": "#E2E8F0", "fontSize": "11px", "fontWeight": "600"}),
                    ],
                )
                for label, value in rows
            ],
        )

    def _build_market_tab_content(self, tab: str, frame: TraderViewFrame | None) -> Component:
        if tab == "trades":
            return self._build_trades_panel(frame)
        return self._build_order_book_panel(frame)

    def _build_order_book_panel(self, frame: TraderViewFrame | None) -> Component:
        if frame is None or frame.market is None:
            return html.Div("Order book feed pending wiring.", style={"color": "#94A3B8"})

        market = frame.market
        spread = None
        if market.best_bid is not None and market.best_ask is not None:
            spread = market.best_ask - market.best_bid
        rows = [
            ["Best Bid", self._compact_value(market.best_bid, places=2), self._compact_value(market.best_bid_quantity, places=4)],
            ["Best Ask", self._compact_value(market.best_ask, places=2), self._compact_value(market.best_ask_quantity, places=4)],
            ["Spread", self._compact_value(spread, places=2), "-"],
        ]
        return self._build_table(
            headers=["Level", "Price", "Size"],
            rows=rows,
            empty_text="Order book unavailable.",
        )

    def _build_trades_panel(self, frame: TraderViewFrame | None) -> Component:
        if frame is None or frame.market is None:
            return html.Div("Trades feed pending wiring.", style={"color": "#94A3B8"})

        market = frame.market
        rows = [[
            str(market.event_ts_ms or "-"),
            self._compact_value(market.last_price, places=2),
            self._compact_value(market.last_quantity, places=4),
            self._format_enum_like(market.last_liquidation_side),
        ]]
        return self._build_table(
            headers=["Event Time", "Price", "Qty", "Side"],
            rows=rows,
            empty_text="No trades yet.",
        )

    def _resolve_mark_price_for_symbol(self, symbol: str, selected_market: TraderViewFrame | None) -> Decimal | None:
        normalized_symbol = symbol.strip().upper()
        if selected_market is not None and selected_market.market is not None:
            if selected_market.market.symbol.strip().upper() == normalized_symbol:
                return selected_market.market.mark_price
        frame = self._latest_frames_by_symbol.get(normalized_symbol)
        if frame is not None and frame.market is not None:
            return frame.market.mark_price
        return None

    def _estimate_position_upnl(self, *, position_qty: Decimal, position_side: str | None, entry_price: Decimal, mark_price: Decimal | None) -> Decimal | None:
        if mark_price is None:
            return None
        qty = position_qty
        if qty == 0:
            return Decimal("0")
        side_value = (position_side or "").strip().lower()
        if side_value == "long":
            signed_qty = abs(qty)
        elif side_value == "short":
            signed_qty = -abs(qty)
        else:
            signed_qty = qty
        return (mark_price - entry_price) * signed_qty

    def _build_bottom_tab_content(
        self,
        tab: str,
        account_frame: TraderViewFrame | None,
        market_frame: TraderViewFrame | None,
    ) -> Component:
        if tab == "positions":
            return self._build_positions_tab(account_frame, market_frame)
        if tab == "assets":
            return self._build_assets_tab(account_frame)
        if tab == "trader":
            return self._build_trader_tab(account_frame)
        return self._build_open_orders_tab(account_frame)

    def _build_open_orders_tab(self, frame: TraderViewFrame | None) -> Component:
        if frame is None or frame.account is None:
            return html.Div("No order data yet.", style={"color": "#94A3B8"})

        orders = sorted(frame.account.orders, key=lambda item: item.timestamp, reverse=True)
        rows = [
            [
                str(order.timestamp),
                order.symbol,
                order.side,
                order.status,
                order.type,
                self._compact_value(order.price, places=4),
                self._compact_value(order.quantity, places=4),
                self._compact_value(order.filled_quantity, places=4),
                self._compact_value(order.average_fill_price, places=4),
            ]
            for order in orders
        ]
        return self._build_table(
            headers=["Time", "Symbol", "Side", "Status", "Type", "Price", "Qty", "Filled", "Avg Fill"],
            rows=rows,
            empty_text="No open orders.",
        )

    def _build_positions_tab(self, frame: TraderViewFrame | None, market_frame: TraderViewFrame | None) -> Component:
        if frame is None or frame.account is None:
            return html.Div("No position data yet.", style={"color": "#94A3B8"})

        rows: list[list[str]] = []
        for position in frame.account.positions:
            mark_price = self._resolve_mark_price_for_symbol(position.symbol, market_frame)
            upnl = self._estimate_position_upnl(
                position_qty=position.quantity,
                position_side=position.side,
                entry_price=position.entry_price,
                mark_price=mark_price,
            )
            rows.append(
                [
                    position.symbol,
                    position.side or "-",
                    self._compact_value(position.quantity, places=4),
                    self._compact_value(position.entry_price, places=4),
                    self._compact_value(mark_price, places=4),
                    self._compact_value(upnl, places=4),
                    position.mode,
                ]
            )
        return self._build_table(
            headers=["Symbol", "Side", "Size", "Entry", "Mark", "uPnL", "Mode"],
            rows=rows,
            empty_text="No active positions.",
        )

    def _build_assets_tab(self, frame: TraderViewFrame | None) -> Component:
        if frame is None or frame.account is None:
            return html.Div("No asset data yet.", style={"color": "#94A3B8"})

        account = frame.account
        rows = []
        for wallet in account.wallets:
            if wallet.asset == account.quote_asset:
                value = wallet.equity
            else:
                value = None
            rows.append(
                [
                    wallet.asset,
                    self._compact_value(wallet.balance, places=6),
                    self._compact_value(wallet.equity, places=6),
                    self._compact_value(wallet.available_margin_balance, places=6),
                    self._compact_value(value, places=6),
                    self._compact_value(wallet.unrealized_pnl, places=6),
                ]
            )
        return self._build_table(
            headers=["Asset", "Balance", "Equity", "Available", "Value", "uPnL"],
            rows=rows,
            empty_text="No assets available.",
        )

    def _build_trader_tab(self, frame: TraderViewFrame | None) -> Component:
        strategy_state = "{}"
        strategy_action = "{}"
        if frame is not None and frame.strategy is not None:
            strategy_state = self._format_json(frame.strategy.state)
            strategy_action = self._format_json(frame.strategy.action)

        return html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "10px", "minHeight": "0"},
            children=[
                html.Div(
                    style={
                        "backgroundColor": "#101319",
                        "border": "1px solid #242A33",
                        "borderRadius": "4px",
                        "padding": "8px",
                        "overflow": "auto",
                    },
                    children=[
                        html.Div("Strategy State", style={"fontSize": "12px", "fontWeight": "600", "marginBottom": "6px"}),
                        html.Pre(
                            strategy_state,
                            style={"margin": "0", "fontSize": "11px", "fontFamily": "monospace", "whiteSpace": "pre-wrap"},
                        ),
                    ],
                ),
                html.Div(
                    style={
                        "backgroundColor": "#101319",
                        "border": "1px solid #242A33",
                        "borderRadius": "4px",
                        "padding": "8px",
                        "overflow": "auto",
                    },
                    children=[
                        html.Div("Strategy Action", style={"fontSize": "12px", "fontWeight": "600", "marginBottom": "6px"}),
                        html.Pre(
                            strategy_action,
                            style={"margin": "0", "fontSize": "11px", "fontFamily": "monospace", "whiteSpace": "pre-wrap"},
                        ),
                    ],
                ),
            ],
        )

    def _format_market_summary(self, frame: TraderViewFrame | None) -> str:
        if frame is None or frame.market is None:
            return "No market data yet."
        market = frame.market
        return (
            f"symbol={market.symbol}\n"
            f"status={self._format_enum_like(market.status)}\n"
            f"last={self._format_decimal(market.last_price)} qty={self._format_decimal(market.last_quantity)}\n"
            f"bid={self._format_decimal(market.best_bid)} @ {self._format_decimal(market.best_bid_quantity)}\n"
            f"ask={self._format_decimal(market.best_ask)} @ {self._format_decimal(market.best_ask_quantity)}\n"
            f"mark={self._format_decimal(market.mark_price)} index={self._format_decimal(market.index_price)}\n"
            f"funding={self._format_decimal(market.funding_rate)} next={market.next_funding_time or '-'}\n"
            f"open_interest={self._format_decimal(market.open_interest)}\n"
            f"open_interest_value={self._format_decimal(market.open_interest_value)}\n"
            f"liq={self._format_enum_like(market.last_liquidation_side)} @ "
            f"{self._format_decimal(market.last_liquidation_price)} qty={self._format_decimal(market.last_liquidation_quantity)}\n"
            f"event_ts={market.event_ts_ms or '-'} recv_ts={market.recv_ts_ms or '-'} seq={market.sequence or '-'}"
        )

    def _format_account_summary(self, frame: TraderViewFrame | None) -> str:
        if frame is None or frame.account is None:
            return "No account data yet."
        account = frame.account
        return (
            f"quote_asset={account.quote_asset}\n"
            f"margin_asset={account.margin_asset}\n"
            f"balance={self._format_decimal(account.balance)}\n"
            f"equity={self._format_decimal(account.equity)}\n"
            f"uPnL={self._format_decimal(account.unrealized_pnl)}\n"
            f"available_margin={self._format_decimal(account.available_margin_balance)}\n"
            f"init_margin={self._format_decimal(account.initial_margin_requirement)}\n"
            f"maint_margin={self._format_decimal(account.maintenance_margin_requirement)}"
        )

    def _format_wallets(self, frame: TraderViewFrame | None) -> str:
        if frame is None or frame.account is None or not frame.account.wallets:
            return "-"

        rows = [["asset", "balance", "equity", "available", "uPnL"]]
        for wallet in frame.account.wallets:
            rows.append(
                [
                    wallet.asset,
                    self._format_decimal(wallet.balance),
                    self._format_decimal(wallet.equity),
                    self._format_decimal(wallet.available_margin_balance),
                    self._format_decimal(wallet.unrealized_pnl),
                ]
            )
        return self._format_table(rows)

    def _format_positions(self, frame: TraderViewFrame | None) -> str:
        if frame is None or frame.account is None or not frame.account.positions:
            return "-"

        rows = [["symbol", "mode", "side", "qty", "entry"]]
        for position in frame.account.positions:
            rows.append(
                [
                    position.symbol,
                    position.mode,
                    position.side or "-",
                    self._format_decimal(position.quantity),
                    self._format_decimal(position.entry_price),
                ]
            )
        return self._format_table(rows)

    def _format_orders(self, frame: TraderViewFrame | None) -> str:
        if frame is None or frame.account is None or not frame.account.orders:
            return "-"

        rows = [["ts", "symbol", "side", "status", "type", "qty", "filled", "price", "avg"]]
        for order in frame.account.orders:
            rows.append(
                [
                    str(order.timestamp),
                    order.symbol,
                    order.side,
                    order.status,
                    order.type,
                    self._format_decimal(order.quantity),
                    self._format_decimal(order.filled_quantity),
                    self._format_decimal(order.price),
                    self._format_decimal(order.average_fill_price),
                ]
            )
        return self._format_table(rows)

    @staticmethod
    def _format_table(rows: list[list[str]]) -> str:
        if not rows:
            return "-"
        widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
        lines = []
        for row in rows:
            padded = [cell.ljust(widths[idx]) for idx, cell in enumerate(row)]
            lines.append(" | ".join(padded))
        return "\n".join(lines)

    def _format_strategy_state(self, frame: TraderViewFrame | None) -> str:
        if frame is None or frame.strategy is None:
            return "{}"
        return self._format_json(frame.strategy.state)

    def _format_strategy_action(self, frame: TraderViewFrame | None) -> str:
        if frame is None or frame.strategy is None:
            return "{}"
        return self._format_json(frame.strategy.action)

    @staticmethod
    def _format_json(value: Any) -> str:
        try:
            return json.dumps(value, indent=2, sort_keys=True, default=str)
        except Exception:
            return str(value)

    @staticmethod
    def _no_update_result() -> tuple[Any, ...]:
        return (
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
            no_update,
        )

    @dataclasses.dataclass
    class _ViewData:
        time: timedelta
        ohlc: Ohlc | None = None
        indicators: list[StrategyIndicator] = dataclasses.field(default_factory=list)

    @dataclasses.dataclass
    class _View:
        symbol: str | None = None
        candle_interval: str = "1m"
        market_tab: str = "orderbook"
        bottom_tab: str = "open_orders"
        currency: str | None = None
        time: timedelta = timedelta(0)
        data_window: list['TradingViewer._ViewData'] | None = None
