import json
import logging
import queue
import threading
import time
from decimal import Decimal

from pytrader import Exchange
from pytrader.simulation import ExchangeViewer
from dashqt import EmbeddedDashApplicationListener, EmbeddedDashApplication


class TradingSimulator(EmbeddedDashApplicationListener):
    """
    Manages and runs an Exchange simulation, optionally displaying state
    via an ExchangeViewer. Implements EmbeddedDashApplicationListener to react to UI events.
    """

    def __init__(self,
        config_filepath: str,
        total_steps: int,
        steps_per_update: int,
        max_updates_per_second: float,
        run_with_ui: bool = True,
        symbol: str | None = None,
        ui_plot_candles: int = 60,
        ui_update_ms: int = 100
    ):
        """
        Initializes the TradingSimulator.

        Args:
            config_filepath: Path to the configuration JSON file.
            total_steps: Total number of simulation steps to run.
            steps_per_update: How many simulation steps between sending updates.
            max_updates_per_second: Rate limit for sending updates to the queue/UI.
            run_with_ui: If True, creates and runs the ExchangeViewer.
            symbol: The market symbol to plot if UI is enabled. Required if run_with_ui is True.
            ui_plot_candles: Max candles for the UI plot window.
            ui_update_ms: Update interval for the UI callback.
        """
        self._config_filepath = config_filepath
        self._total_steps = total_steps
        self._steps_per_update = max(1, steps_per_update)
        self._max_updates_per_second = max_updates_per_second
        self._run_with_ui = run_with_ui
        self._symbol = symbol
        self._ui_plot_candles = ui_plot_candles
        self._ui_update_ms = ui_update_ms

        cls = type(self)
        self._logger: logging.Logger = logging.getLogger(f"{cls.__module__}.{cls.__name__}")

        if self._run_with_ui and not self._symbol:
            raise ValueError("ui_symbol must be provided if run_with_ui is True.")

        # Simulation State
        self._exchange: Exchange | None = None
        self._exchange_state_queue = queue.Queue()
        self._min_update_delay_seconds: float = (
            1.0 / self._max_updates_per_second if self._max_updates_per_second > 0 else 0.0
        )
        self._last_update_put_time: float = 0.0

        # UI Management (if applicable)
        self._viewer: ExchangeViewer | None = None
        self._viewer_thread: threading.Thread | None = None
        self._is_viewer_ready = threading.Event()
        self._is_viewer_stopped = threading.Event()

        # Control
        self._is_stopping = threading.Event()
        self._exit_code: int = 0

    def run(self) -> int:
        """
        Runs the entire simulation lifecycle, including starting the optional UI
        and the main simulation loop. Blocks until completion. Returns final exit code.
        """
        self._logger.info("TradingSimulator starting...")
        viewer_start_successful = not self._run_with_ui  # Assume OK if no UI

        try:
            # --- Load Config and Init Exchange ---
            self._logger.info(f"Loading configuration from {self._config_filepath}")
            with open(self._config_filepath, 'r') as f:
                config = json.load(f)
            self._exchange = Exchange(config)
            self._logger.info("Exchange initialized.")

            if self._run_with_ui and self._symbol not in self._exchange.markets:
                raise ValueError(f"ui_symbol '{self._symbol}' not found in configured markets.")

            # --- Start UI Thread (if applicable) ---
            if self._run_with_ui:
                self._logger.info("Starting ExchangeViewer UI...")
                self._viewer = ExchangeViewer(
                    view_update_queue=self._exchange_state_queue,
                    listener=self,  # Pass self as the listener
                    symbol=self._symbol,
                    plot_window_candles=self._ui_plot_candles,
                    view_update_interval_ms=self._ui_update_ms
                )
                self._viewer_thread = threading.Thread(
                    target=self._viewer.run_forever,
                    name="ExchangeViewerThread",
                    daemon=False  # Non-daemon so we can join
                )
                self._viewer_thread.start()

                # Wait for the viewer to signal it's ready
                self._logger.info("Waiting for viewer to become ready...")
                if not self._is_viewer_ready.wait(timeout=30.0):  # Wait up to 30s
                    self._logger.error("Viewer did not become ready within timeout. Stopping simulation.")
                    self.stop()  # Signal self to stop
                    self._exit_code = 1
                    viewer_start_successful = False  # Mark UI as failed
                    # Proceed to finally block for cleanup
                else:
                    self._logger.info("Viewer is ready.")
                    viewer_start_successful = True  # Mark UI as started
            else:
                self._logger.info("Running simulation headless (no UI).")

            # --- Run Simulation Loop (only if viewer started OK or no UI) ---
            if viewer_start_successful:
                self._run_simulation_loop()
            else:
                # If viewer failed to start, ensure we stop gracefully
                self._is_stopping.set()

        except KeyboardInterrupt:
            self._logger.warning("KeyboardInterrupt (Ctrl+C) caught, initiating stop...")
            self._exit_code = 130  # Standard exit code for SIGINT
            self.stop()  # Trigger the graceful shutdown using the event
            # Do not re-raise, allow finally block to execute

        except Exception as e:
            # Avoid logging error if stop was requested (e.g., by Ctrl+C)
            # Log other exceptions only if not already stopping gracefully
            if not self._is_stopping.is_set():
                self._logger.error(f"Fatal error during TradingSimulator run: {e}", exc_info=True)
                self._exit_code = 1
            elif self._exit_code == 0:  # If stopped, but exception occurred during shutdown
                self._logger.warning(f"Caught exception during stop sequence: {e}", exc_info=True)
                self._exit_code = 1  # Still indicate an issue occurred

            self.stop()  # Ensure stop is signaled on error

        finally:
            self._logger.info("TradingSimulator run method ending.")

            # Wait for the viewer thread to complete ONLY IF the UI was started
            # and the simulation finished normally (or was stopped gracefully).
            # If there was a fatal error causing an early exit from the try block,
            # we might already be trying to shut down the viewer via stop().
            if self._run_with_ui and self._viewer_thread:
                if self._viewer_thread.is_alive():
                    self._logger.info("Simulation complete. Waiting for UI window to be closed...")
                    # Wait indefinitely for the viewer thread to finish
                    # This thread only finishes when the user closes the UI window
                    self._viewer_thread.join()
                    self._logger.info("Viewer thread finished after UI close.")
                else:
                    # If the viewer thread isn't alive here, it might have stopped
                    # due to an error reported by the listener, or shut down early.
                    self._logger.info("Viewer thread already finished when checked in finally block.")
            else:
                # Update headless completion log message
                if not self._run_with_ui:
                    completion_status = "completed"
                    # Check exit code set by KeyboardInterrupt or other errors
                    if self._exit_code == 130:
                        completion_status = "stopped by signal (KeyboardInterrupt)"
                    elif self._exit_code != 0:
                        completion_status = f"finished with error code {self._exit_code}"
                    self._logger.info(f"Headless simulation {completion_status}.")

            self._logger.info("Trading simulation stopped")

            return self._exit_code

    def _run_simulation_loop(self):
        """
        Internal method containing the main simulation step loop.
        """
        self._logger.info(f"Starting simulation loop for {self._total_steps} steps.")

        current_step = 0
        steps_in_batch = 0
        step_duration_sec = self._exchange.settings.step_duration

        # OHLC Aggregation State
        ohlc_open = None
        ohlc_high = Decimal('-Infinity')
        ohlc_low = Decimal('Infinity')
        ohlc_close = None

        try:
            while current_step < self._total_steps and not self._is_stopping.is_set():

                # --- Perform one simulation step ---
                state: ExchangeState = self._exchange.step()
                current_step = state.step
                steps_in_batch += 1

                market_state: MarketState = state.markets[self._symbol]
                account_state: AccountState = state.account

                # --- Aggregate Data for Update ---
                # OHLC Aggregation
                # if self._run_with_ui:
                # if market_state:
                last_price = market_state.last_price
                if last_price is not None:
                    if ohlc_open is None:
                        ohlc_open, ohlc_high, ohlc_low = last_price, last_price, last_price
                        # ohlc_start_step = current_step
                    else:
                        ohlc_high = max(ohlc_high, last_price)
                        ohlc_low = min(ohlc_low, last_price)  # Corrected: min(ohlc_low, ...)
                    ohlc_close = last_price

                # --- Check if an update should be pushed ---
                if steps_in_batch >= self._steps_per_update:  # or current_step == self._total_steps:

                    # --- Apply rate limiting ---
                    if self._run_with_ui:
                        if self._enforce_rate_limiting():
                            break  # stop event was signaled

                    # --- Prepare and Send/Log Updates ---
                    time_sec = current_step * step_duration_sec  # Simulation time

                    state_update = {
                        'step': current_step,
                        'total_steps': self._total_steps,
                        'time_min': time_sec / 60.0,
                        'symbol': self._symbol,
                        'quote_currency': self._exchange.markets[self._symbol].quote_currency,
                        'market': {
                            'last_price': market_state.last_price,
                            'ohlc': {
                                'open': ohlc_open, 'high': ohlc_high, 'low': ohlc_low, 'close': ohlc_close
                            },
                            'mark_price': market_state.mark_price,
                            'index_price': market_state.index_price,
                            'funding_rate': market_state.funding_rate,
                            'limit_orders': market_state.limit_orders
                        },
                        'account': {
                            'cash_balance': account_state.cash_balance,
                            'realized_pnl': account_state.realized_pnl,
                            'total_equity': account_state.total_equity,
                            'available_margin': account_state.available_margin,
                            'positions': {
                                key: {
                                    'entry_price': position.entry_price,
                                    'quantity': position.quantity
                                }
                                for key, position in account_state.positions.items()
                                if key == self._symbol
                            }
                        }
                    }
                    if self._run_with_ui:
                        self._exchange_state_queue.put(state_update)

                    ohlc = state_update['market']['ohlc']
                    account = state_update['account']
                    status_log_msg = (
                        f"Step {current_step}/{self._total_steps} | Time: {state_update['time_min']:.1f} min | "
                        f"{self._symbol} | OHLC: {ohlc['open']:.2f}/{ohlc['high']:.2f}/{ohlc['low']:.2f}/"
                        f"{ohlc['close']:.2f} | Cash Balance: {account['cash_balance']:.2f} | "
                        f"Realized P&L: {account['realized_pnl']:.2f} | Equity: {account['total_equity']:.2f} | "
                        f"Available Margin: {account['available_margin']:.2f}"
                    )

                    # Reset OHLC aggregation
                    ohlc_open, ohlc_high, ohlc_low, ohlc_close = (
                        None, Decimal('-Infinity'), Decimal('Infinity'), None
                    )

                    # Log combined status update
                    self._logger.info(status_log_msg)

                    # Reset batch counter
                    steps_in_batch = 0

                # Add optional brief sleep to yield CPU when not updating which helps to prevent 100% CPU usage in
                # fast loops and improves responsiveness
                else:
                    if self._is_stopping.is_set():   # Check stop event before sleep
                        break
                    time.sleep(0.005)  # Yield briefly (adjust as needed)

        except Exception as e:
            # Log error only if not caused by stop event
            if not self._is_stopping.is_set():
                self._logger.error(f"Error during simulation loop: {e}", exc_info=True)
                # Set error code only if not already set (e.g., by KeyboardInterrupt)
            else:
                # Log as warning if exception occurs during stop sequence
                self._logger.warning(f"Caught exception during loop shutdown: {e}", exc_info=True)
            if self._exit_code == 0:
                self._exit_code = 1
        finally:
            # Ensure stop is set reliably upon exiting the loop, regardless of reason
            self._is_stopping.set()  # Set the event FIRST
            # Now log the state
            loop_end_reason = "completed"
            # Check if total steps were reached ONLY if the event wasn't set externally before this finally block
            # This is tricky to determine perfectly, but we know if the event is set now, it was stopped.
            if current_step < self._total_steps:
                loop_end_reason = "stopped"  # Assume stopped if loop exited early
            self._logger.info(
                f"Simulation loop finished ({loop_end_reason}). Stop event is set: {self._is_stopping.is_set()}."
            )

    def _enforce_rate_limiting(self) -> bool:
        current_time = time.monotonic()
        time_since_last = current_time - self._last_update_put_time
        if time_since_last < self._min_update_delay_seconds:
            sleep_time = self._min_update_delay_seconds - time_since_last
            if self._is_stopping.is_set():
                return True  # Check before sleep attempt
            wakeup_time = time.monotonic() + sleep_time
            while time.monotonic() < wakeup_time and not self._is_stopping.is_set():
                remaining_sleep = wakeup_time - time.monotonic()
                # Sleep for short intervals checking the stop event
                time.sleep(min(remaining_sleep, 0.05))  # e.g., sleep up to 50ms
            if self._is_stopping.is_set():
                return True  # Check again after sleep attempt
        self._last_update_put_time = time.monotonic()
        return False

    def stop(self):
        """Signals the simulation loop and the viewer (if running) to stop."""
        # Check if stop hasn't already been signaled to avoid redundant actions/logs
        if not self._is_stopping.is_set():
            self._logger.info("Stop requested.")
            self._is_stopping.set()
            # Request viewer close *only if* it seems to still be running
            # (Check instance and potentially thread liveness)
            # The check ensures we don't try to close a viewer that never started or already stopped
            if (self._run_with_ui and self._viewer and self._viewer_thread
                and self._viewer_thread.is_alive()
            ):
                self._logger.info("Requesting viewer to close.")
                self._viewer.request_browser_close()
        else:
            # Log at debug level if stop is called when already stopping
            self._logger.debug("Stop already requested or in progress.")

    def on_dash_app_started(self, app: EmbeddedDashApplication):
        # Check if it's the viewer instance we manage
        if app is self._viewer:
            self._logger.info("Listener: Viewer reported started.")
            self._is_viewer_ready.set()

    def on_dash_app_stopped(self, app: EmbeddedDashApplication, exit_code: int):
        # Check if it's the viewer instance we manage
        if app is self._viewer:
            # Trigger the simulator's stop mechanism when the viewer stops.
            self._logger.info("Viewer stopped, initiating simulator stop.")
            # START CHANGE: Set exit code based on viewer if simulator hasn't already set one
            if exit_code != 0 and self._exit_code == 0:
                self._logger.warning(
                    f"Viewer stopped with non-zero exit code: {exit_code}. Setting simulator exit code."
                    )
                # Use viewer's code if it's numeric, otherwise default error code 1
                self._exit_code = exit_code if isinstance(exit_code, int) else 1
            # END CHANGE: Set exit code based on viewer if simulator hasn't already set one
            self.stop()  # Call own stop method to halt simulation loop


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(threadName)s | %(levelname)s | %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    # Optionally reduce noise from libraries
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("plotly").setLevel(logging.INFO)
    logging.getLogger("Qt").setLevel(logging.INFO)
    # Set levels for *your* specific loggers using FQDN if needed
    logging.getLogger("Qt").setLevel(logging.WARNING)  # Usually WARNING is enough for Qt

    # --- Simulation Parameters ---
    CONFIG_FILE = "../../../examples/config.json"  # Make sure this file exists
    # Adjust total steps for reasonable runtime
    TOTAL_SIM_HOURS = 2.0  # e.g., 2 hours simulation time
    # Assuming 1 step = 1 simulated second for simplicity
    # TODO: Define step duration more explicitly if needed
    SIMULATION_STEP_DURATION_SECONDS = 1.0
    TOTAL_STEPS = int(TOTAL_SIM_HOURS * 3600 / SIMULATION_STEP_DURATION_SECONDS)

    STEPS_PER_UPDATE = 60  # Send UI/log update every 60 steps (e.g., 1 minute)
    MAX_UPDATES_PER_SECOND = 1.0  # Rate limit at which updates are pushed
    RUN_WITH_UI = True  # Set to False to run headless
    SYMBOL = "BTC/USDT"  # Must exist in config.json

    main_logger = logging.getLogger(__name__)  # Logger for this main block
    main_logger.info("--- Starting Trading Simulation ---")
    main_logger.info(f"Config File: {CONFIG_FILE}")
    main_logger.info(f"Total Steps: {TOTAL_STEPS}")
    main_logger.info(f"Steps per Update: {STEPS_PER_UPDATE}")
    main_logger.info(f"Max Updates/sec: {MAX_UPDATES_PER_SECOND}")
    main_logger.info(f"Run with UI: {RUN_WITH_UI}")
    if RUN_WITH_UI:
        main_logger.info(f"Symbol: {SYMBOL}")

    # --- Create and Run Simulator ---
    simulator = TradingSimulator(
        config_filepath=CONFIG_FILE,
        total_steps=TOTAL_STEPS,
        steps_per_update=STEPS_PER_UPDATE,
        max_updates_per_second=MAX_UPDATES_PER_SECOND,
        run_with_ui=RUN_WITH_UI,
        symbol=SYMBOL,
        ui_plot_candles=60,  # Display last 60 candles
        ui_update_ms=100  # Check queue slightly less often
    )

    final_exit_code = simulator.run()  # This blocks until completion

    print(f"\n--- Main script finished with final exit code: {final_exit_code} ---")
    exit(final_exit_code)
