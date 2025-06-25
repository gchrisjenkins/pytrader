import dataclasses
import logging
import queue
from datetime import timedelta
from decimal import Decimal
from typing import Any

import plotly.graph_objects as go
from dash import Output, Input, dcc, html, no_update
from dash.development.base_component import Component

from dashqt import EmbeddedDashApplication, EmbeddedDashApplicationListener
from pytrader import Duration, TimeUnit


@dataclasses.dataclass
class TraderViewConfiguration:
    algorithm_name: str
    symbols: list[str]
    currency: str
    max_duration: Duration
    time_unit: TimeUnit
    max_candle_window_size: int
    view_refresh_frequency_hz: int = 5


@dataclasses.dataclass
class Ohlc:
    time: int
    open: Decimal
    high: Decimal | None = None
    low: Decimal | None = None
    close: Decimal | None = None

    def __post_init__(self) -> None:

        if self.high is None:
            self.high = self.open
        if self.low is None:
            self.low = self.open
        if self.close is None:
            self.close = self.open


@dataclasses.dataclass
class Indicator:
    name: str
    value: float | None
    is_price_indicator: bool = False


@dataclasses.dataclass
class TraderViewDataUpdate:
    symbol: str
    time: timedelta
    data: dict[str, Any] = dataclasses.field(default_factory=dict)


class TradingViewer(EmbeddedDashApplication):

    def __init__(self,
        listener: EmbeddedDashApplicationListener | None,
        config: TraderViewConfiguration
    ):
        super().__init__(listener, "Trading Viewer")

        cls = type(self)
        self._logger: logging.Logger = logging.getLogger(f"{cls.__module__}.{cls.__name__}")

        self._config: TraderViewConfiguration = config

        self._view_update_queue: queue.Queue[TraderViewDataUpdate] = queue.Queue()

        self._view_data: dict[str, TradingViewer._ViewData] = {}
        self._view_data_history: dict[str, list[TradingViewer._ViewData]] = {}

        self._view: TradingViewer._View = TradingViewer._View()
        self._view.symbol = self._config.symbols[0]
        self._view.currency = self._config.currency

        self._cached_figure: go.Figure | None = None

    @property
    def view_update_queue(self) -> queue.Queue[TraderViewDataUpdate]:
        return self._view_update_queue

    def _build_layout(self) -> Component | list[Component]:

        self._logger.debug("Building viewer layout")

        return html.Div(
            style={
                'backgroundColor': '#111111',  # Dark background
                'color': '#ffffff',  # White text for visibility
                'margin': '0',  # Remove default margins
                'padding': '0px',  # Add internal padding (adjust as needed)
                # 'minHeight': '100vh',         # Full viewport height
                # 'width': '100vw',             # Full viewport width
                # 'boxSizing': 'border-box'     # Ensure padding doesn’t increase size
            },
            children=[
                dcc.Interval(
                    id='viewer-update-interval',
                    interval=1000. / self._config.view_refresh_frequency_hz,
                    n_intervals=0
                ),
                html.H3(
                    id='viewer-title',
                    children=[self._build_viewer_title()],
                    style={'textAlign': 'center', 'flexShrink': '0', 'margin': '10px'}
                ),
                html.Div(
                    style={'flexGrow': '1', 'minHeight': '400px', 'overflow': 'hidden'},
                    children=[
                        dcc.Graph(id='viewer-candlestick-graph', style={'height': '100%'})
                    ]
                ),
                html.Div(
                    id='viewer-status-text', style={'padding': '10px', 'textAlign': 'center', 'flexShrink': '0'}
                )
            ]
        )

    def _build_callbacks(self) -> list[tuple]:
        self._logger.debug("Building ExchangeViewer callbacks")
        return [(
            (
                Output('viewer-title', 'children'),
                Output('viewer-candlestick-graph', 'figure'),
                Output('viewer-status-text', 'children')
            ),
            Input('viewer-update-interval', 'n_intervals'),
            self._update_view
        )]

    def _update_view(self, n_intervals: int):

        is_view_dirty: bool = False
        while not self._view_update_queue.empty():
            try:
                view_update = self._view_update_queue.get_nowait()

                symbol = view_update.symbol
                if self._view.symbol == symbol:
                    is_view_dirty = True

                data = view_update.data
                if (
                    self._view_data.get(symbol) is None or
                    self._view_data[symbol].ohlc.time != data['ohlc']['time']
                ):
                    self._view_data[symbol] = TradingViewer._ViewData(view_update.time)

                    if self._view_data_history.get(symbol) is None:
                        self._view_data_history[symbol] = []
                    self._view_data_history[symbol].append(self._view_data[symbol])

                view_data = self._view_data[symbol]
                view_data.time = view_update.time

                view_data.ohlc = Ohlc(
                    time=data['ohlc']['time'],
                    open=data['ohlc']['open'],
                    high=data['ohlc']['high'],
                    low=data['ohlc']['low'],
                    close=data['ohlc']['close']
                )

                view_data.indicators = [
                    Indicator(indicator['name'], indicator['value'], indicator['is_price_indicator']) for indicator in data['indicators']
                ]

                self._view_update_queue.task_done()

            except Exception as e:
                self._logger.error(f"Error processing update from queue: {e}")
                break  # Stop processing queue on error

        # Optimization: Avoid redrawing if no data has changed
        if not is_view_dirty and (self._cached_figure or self._view_data_history.get(self._view.symbol) is None):
            # self.__logger.debug("No new data, returning cached figure/no_update.")
            # Use no_update to tell Dash not to change these outputs
            return no_update, no_update, no_update

        fig = go.Figure()

        self._view.time = timedelta(seconds=self._view_data_history[self._view.symbol][-1].time.total_seconds())
        self._view.data_window = self._view_data_history[self._view.symbol][-self._config.max_candle_window_size:]

        ohlc_data = {
            'time': [data.ohlc.time for data in self._view.data_window],
            'open': [float(data.ohlc.open) for data in self._view.data_window],
            'high': [float(data.ohlc.high) for data in self._view.data_window],
            'low': [float(data.ohlc.low) for data in self._view.data_window],
            'close': [float(data.ohlc.close) for data in self._view.data_window],
        }

        hover_texts = [
            (f"<b>Open:</b>  {o:.2f}<br><b>High:</b>  {h:.2f}<br><b>Low:</b>   {l:.2f}<br><b>Close:</b> {c:.2f}<br>"
             f"<b>Time:  </b>{t:.0f}")
            for t, o, h, l, c in zip(
                ohlc_data['time'], ohlc_data['open'], ohlc_data['high'], ohlc_data['low'], ohlc_data['close']
            )
        ]

        fig.add_trace(
            go.Candlestick(
                x=ohlc_data['time'],
                open=ohlc_data['open'],
                high=ohlc_data['high'],
                low=ohlc_data['low'],
                close=ohlc_data['close'],
                name=self._view.symbol,
                hovertext=hover_texts,
                hoverinfo="text"
            )
        )

        # Add traces for price-based indicators
        if self._view.data_window and self._view.data_window[-1].indicators:
            # Identify which indicators to plot based on the last data point
            price_indicators_to_plot = [
                ind for ind in self._view.data_window[-1].indicators
                if ind.is_price_indicator and ind.name
            ]

            # For each identified indicator, create a time series and plot it
            for indicator_template in price_indicators_to_plot:
                indicator_name = indicator_template.name
                indicator_values = []

                # Build the list of y-values for the indicator across the data window
                for data_point in self._view.data_window:
                    # Find the corresponding indicator in the current data point
                    found_indicator = next(
                        (ind for ind in data_point.indicators if ind.name == indicator_name), None
                    )

                    # Add its value, or None if it's missing or the value is None
                    # Plotly correctly handles `None` by creating gaps in the line
                    if found_indicator and found_indicator.value is not None:
                        indicator_values.append(found_indicator.value)
                    else:
                        indicator_values.append(None)

                # Add the line trace for the indicator to the figure
                if any(v is not None for v in indicator_values):  # Avoid adding empty traces
                    fig.add_trace(
                        go.Scatter(
                            x=ohlc_data['time'],
                            y=indicator_values,
                            mode='lines',
                            name=indicator_name,
                            line=dict(width=1.5)  # Style the line to be thinner than candles
                        )
                    )

        # Common layout updates
        fig.update_layout(
            yaxis_title=f"Price ({self._view.currency})",
            xaxis_title="Time (min)",
            template='plotly_dark',
            xaxis_rangeslider_visible=False,
            margin=dict(l=50, r=20, t=40, b=40),
            paper_bgcolor='rgba(17,17,17,1)',
            plot_bgcolor='rgba(17,17,17,1)',
            xaxis=dict(gridcolor='#444444', linecolor='#CCCCCC', tickcolor='#CCCCCC', zerolinecolor='#444444'),
            yaxis=dict(gridcolor='#444444', linecolor='#CCCCCC', tickcolor='#CCCCCC', zerolinecolor='#444444'),
            legend=dict(font=dict(color='#CCCCCC')),
            hovermode='x unified', hoverlabel=dict(font_family="monospace", font_size=11, align='left')
        )

        # Cache the generated figure if valid
        if fig.data:
            self._cached_figure = fig
        else:  # Don't cache an empty/error figure
            self._cached_figure = None

        # Regenerate status text
        status = (
            # f"Step: {self._view.step} / {self._config.max_steps} | "
            f"Time: {self._view.time.total_seconds() / self._config.time_unit.seconds:.1f} min | "
            f"Last Price: {self._view.data_window[-1].ohlc.close:.2f}"
        )

        return self._build_viewer_title(), fig, status

    def _build_viewer_title(self) -> str:
        return f"{self._config.algorithm_name} Simulation ({self._view.symbol})"

    @dataclasses.dataclass
    class _ViewData:
        time: timedelta
        ohlc: Ohlc | None = None
        indicators: list[Indicator] = dataclasses.field(default_factory=list)

    @dataclasses.dataclass
    class _View:
        symbol: str | None = None
        currency: str | None = None
        time: timedelta = timedelta(0)
        data_window: list['TradingViewer._ViewData'] | None = None
