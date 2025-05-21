import logging
import queue
from collections import deque

import plotly.graph_objects as go
from dash import Output, Input, dcc, html, no_update
from dash.development.base_component import Component

from dashqt import EmbeddedDashApplication, EmbeddedDashApplicationListener


class ExchangeViewer(EmbeddedDashApplication):
    """
    An EmbeddedDashApplication that displays exchange simulation state received
    via a queue and notifies a listener about its lifecycle events.
    """

    def __init__(self,
        view_update_queue: queue.Queue,
        listener: EmbeddedDashApplicationListener | None,
        symbol: str,
        plot_window_candles: int,
        view_update_interval_ms: int,
        title: str | None = None
    ):
        """
        Initializes the ExchangeViewer.

        Args:
            view_update_queue: Queue to receive view update dictionaries from.
            listener: An object implementing EmbeddedDashApplicationListener.
            symbol: The market symbol for the main OHLC plot.
            plot_window_candles: Max number of candles to display in the plot window.
            view_update_interval_ms: How often the Dash UI checks the queue (milliseconds).
            title: Optional title for the browser window.
        """
        # Pass listener up to the base class constructor
        super().__init__(title or "Exchange Viewer", listener=listener)

        cls = type(self)
        self.__logger: logging.Logger = logging.getLogger(f"{cls.__module__}.{cls.__name__}")

        # Store specific attributes for the viewer
        self.__symbol = symbol
        self.__view_update_queue = view_update_queue
        self.__view_update_interval_ms = view_update_interval_ms

        # UI State
        self.__view_data: deque[dict[str, dict | object]] = deque(maxlen=plot_window_candles)

        # # Cache figure to optimize updates when no new data arrives
        self.__figure_cache: go.Figure | None = None

    def _build_layout(self) -> Component | list[Component]:
        self.__logger.debug("Building ExchangeViewer layout")

        return html.Div(
            style={
                'backgroundColor': '#111111',  # Dark background
                'color': '#ffffff',           # White text for visibility
                'margin': '0',                # Remove default margins
                'padding': '0px',            # Add internal padding (adjust as needed)
                # 'minHeight': '100vh',         # Full viewport height
                # 'width': '100vw',             # Full viewport width
                # 'boxSizing': 'border-box'     # Ensure padding doesn’t increase size
            },
            children=[
                dcc.Interval(
                    id='viewer-update-interval',
                    interval=self.__view_update_interval_ms,
                    n_intervals=0
                ),
                html.H1(
                    id='viewer-title',
                    children=[f"{self.__symbol} Simulation"],
                    style={'textAlign': 'center', 'flexShrink': '0', 'margin': '10px'}),
                # Use a Div wrapper for the Graph to potentially control size/styles better
                html.Div(
                    style={'flexGrow': '1', 'minHeight': '400px', 'overflow': 'hidden'},
                    children=[
                        dcc.Graph(id='viewer-candlestick-graph', style={'height': '100%'})
                    ]
                ),
                html.Div(
                    id='viewer-status-text', style={'padding': '10px', 'textAlign': 'center', 'flexShrink': '0'}
                )
            ])

    # --- Dash Callbacks Implementation ---
    def _build_callbacks(self) -> list[tuple]:
        self.__logger.debug("Building ExchangeViewer callbacks")
        return [(
            (
                Output('viewer-candlestick-graph', 'figure'),
                Output('viewer-status-text', 'children'),
                Output('viewer-title', 'children')
                # Add Output('viewer-account-text', 'children') if using account div
            ),
            Input('viewer-update-interval', 'n_intervals'),
            self._update_view  # Instance method as the callback
        )]

    # --- Callback Implementation ---
    def _update_view(self, n_intervals: int):
        """
        Callback triggered by the interval timer. Drains the queue, updates
        internal state, and regenerates the plot and status text.
        """
        # self.__logger.debug(f"Update interval {n_intervals} triggered.")
        processed_count = 0
        while not self.__view_update_queue.empty():
            try:
                update = self.__view_update_queue.get_nowait()
                self.__view_data.append(update)
                processed_count += 1

                self.__view_update_queue.task_done()

            except queue.Empty:
                break  # Should not happen with check, but safe
            except Exception as e:
                self.__logger.error(f"Error processing queue item: {e}")
                break  # Stop processing queue on error

        # --- Optimization: Avoid redrawing if no data changed ---
        if not self.__view_data or (processed_count == 0 and self.__figure_cache is not None):
            # self.__logger.debug("No new data, returning cached figure/no_update.")
            # Use no_update to tell Dash not to change these outputs
            return no_update, no_update, no_update
            # return no_update, no_update

        # --- Regenerate Figure ---
        fig = go.Figure()
        view_data_list = list(self.__view_data)
        plot_title = f"{self.__symbol} Simulation (Step {view_data_list[-1]['step']})"
        quote_ccy = view_data_list[-1]['quote_currency']
        y_axis_title = f"Price ({quote_ccy})"

        try:
            ohlc_data = {
                'time_min': [d['time_min'] for d in view_data_list],
                'open': [d['market']['ohlc']['open'] for d in view_data_list],
                'high': [d['market']['ohlc']['high'] for d in view_data_list],
                'low': [d['market']['ohlc']['low'] for d in view_data_list],
                'close': [d['market']['ohlc']['close'] for d in view_data_list],
            }

            hover_texts = [(
                f"<b>Time: </b>{tm:.0f}<br><b>O:</b> {o:.2f}<br><b>H:</b> {h:.2f}<br><b>L:</b>"
                f" {l:.2f}<br><b>C:</b> {c:.2f}"
                for tm, o, h, l, c in
                zip(ohlc_data['time_min'], ohlc_data['open'], ohlc_data['high'], ohlc_data['low'], ohlc_data['close'])
            )]

            fig.add_trace(
                go.Candlestick(
                    x=ohlc_data['time_min'],
                    open=ohlc_data['open'], high=ohlc_data['high'],
                    low=ohlc_data['low'], close=ohlc_data['close'],
                    name=self.__symbol,
                    hovertext=hover_texts, hoverinfo="text"
                )
            )
        except Exception as e:
            self.__logger.error(f"Error creating candlestick trace: {e}", exc_info=True)
            # Keep fig empty or add annotation
            fig.update_layout(title_text="Error displaying chart data")

        # Common layout updates
        fig.update_layout(
            yaxis_title=y_axis_title, xaxis_title='Time (min)',
            template='plotly_dark', xaxis_rangeslider_visible=False,
            margin=dict(l=50, r=20, t=40, b=40), paper_bgcolor='rgba(17,17,17,1)',
            plot_bgcolor='rgba(17,17,17,1)',
            xaxis=dict(gridcolor='#444444', linecolor='#CCCCCC', tickcolor='#CCCCCC', zerolinecolor='#444444'),
            yaxis=dict(gridcolor='#444444', linecolor='#CCCCCC', tickcolor='#CCCCCC', zerolinecolor='#444444'),
            legend=dict(font=dict(color='#CCCCCC')),
            hovermode='x unified', hoverlabel=dict(font_family="monospace", font_size=11, align='left')
        )

        # Cache the generated figure if it's valid
        if fig.data:
            self.__figure_cache = fig
        else:  # Don't cache an empty/error figure
            self.__figure_cache = None

        # # --- Regenerate Status Text ---
        last_view_data = view_data_list[-1]
        status = (
            f"Step: {last_view_data['step']} / {last_view_data['total_steps']} | "
            f"Time: {last_view_data['time_min']:.1f} min | "
            f"Last Price: {last_view_data['market']['last_price']:.2f}"
        )

        return fig, status, plot_title
