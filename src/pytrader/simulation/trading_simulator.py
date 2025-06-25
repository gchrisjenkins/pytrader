import copy
import logging
import queue
import threading
import time
from datetime import timedelta

from dashqt import EmbeddedDashApplicationListener, EmbeddedDashApplication
from pytrader.ui import TradingViewer, TraderViewConfiguration, TraderViewDataUpdate, Ohlc
from pytrader import TimeUnit, Duration
from pytrader.simulation import SimulatedTrader


class TradingSimulator(EmbeddedDashApplicationListener):

    def __init__(self, trader: SimulatedTrader, max_duration: Duration, max_trade_frequency_hz: float = 1.0,
        is_viewer_enabled: bool = False, max_candle_window_size: int = 60, view_refresh_frequency_hz: int = 5
    ):
        super().__init__()

        cls = type(self)
        self._logger: logging.Logger = logging.getLogger(f"{cls.__module__}.{cls.__name__}")

        self._trader: SimulatedTrader = trader
        self._max_duration: Duration = max_duration
        self._max_trade_frequency_hz: float = max_trade_frequency_hz
        self._is_viewer_enabled: bool = is_viewer_enabled
        self._max_candle_window_size: int = max_candle_window_size
        self._view_refresh_frequency_hz: int = view_refresh_frequency_hz

        # self._view_update_queue: queue.Queue[TraderViewDataUpdate] = queue.Queue()

        step_duration_seconds = trader.exchange.settings.step_duration_seconds
        trade_interval_seconds = trader.algorithm.trade_interval_seconds
        steps_per_trade = int(trade_interval_seconds / step_duration_seconds)

        self._max_steps: int = int(max_duration.seconds / step_duration_seconds)
        self._max_step_frequency_hz = self._max_trade_frequency_hz * steps_per_trade
        self._min_step_delay_seconds: float = max(0., 1. / self._max_step_frequency_hz)

        self._last_step_time: int = 0
        self._viewer_data: dict[str, dict] = {}

        self._viewer: TradingViewer | None = None
        self._viewer_thread: threading.Thread | None = None
        self._is_viewer_ready = threading.Event()
        self._is_stopping = threading.Event()
        self._exit_code: int = 0

    def run(self) -> int:

        try:
            self._logger.info("Starting trading simulation...")

            was_viewer_startup_successful = False
            if self._is_viewer_enabled:
                self._logger.info("Running simulation with viewer enabled")
                self._viewer = TradingViewer(
                    listener=self,
                    config=TraderViewConfiguration(
                        algorithm_name=self._trader.algorithm.name,
                        symbols=self._trader.algorithm.symbols,
                        currency=self._trader.algorithm.currency,
                        max_duration=self._max_duration,
                        # max_time=timedelta(
                        #     seconds=self._max_steps * self._trader.exchange.settings.step_duration_seconds
                        # ),
                        time_unit=TimeUnit.MINUTE,
                        # algorithm=self._trader.algorithm,
                        max_candle_window_size=self._max_candle_window_size,
                        # max_candles_displayed=self._candle_window_size,
                        # max_steps=self._max_steps,
                        view_refresh_frequency_hz=self._view_refresh_frequency_hz,
                    )
                )
                self._viewer_thread = threading.Thread(
                    target=self._viewer.run_forever,
                    name="TradingViewerThread",
                    daemon=False
                )
                self._viewer_thread.start()

                # Wait for the viewer to signal it's ready
                self._logger.info("Waiting for viewer to become ready...")
                if not self._is_viewer_ready.wait(timeout=30.0):  # Wait up to 30s
                    self._logger.error("Viewer did not become ready within timeout. Stopping simulation.")
                    self._shutdown()
                    self._exit_code = 1
                    # Proceed to finally block for cleanup
                else:
                    self._logger.info("Viewer is ready")
                    was_viewer_startup_successful = True
            else:
                self._logger.info("Running simulation in headless mode")

            # Run simulation loop only if viewer started successfully or in headless mode
            if not self._is_viewer_enabled or was_viewer_startup_successful:
                self._simulation_loop()
            else:
                # If viewer failed to start, ensure we stop gracefully
                self._is_stopping.set()

        except KeyboardInterrupt:

            self._logger.warning("KeyboardInterrupt (Ctrl+C) caught, initiating shutdown...")
            self._exit_code = 130  # Standard exit code for SIGINT
            self._shutdown()
            # Do not re-raise, allow finally block to execute

        except Exception as e:

            # Avoid logging error if stop was requested (e.g., by Ctrl+C)
            # Log other exceptions only if not already stopping gracefully
            if not self._is_stopping.is_set():
                self._logger.error(f"Fatal exception occurred during simulation: {e}", exc_info=True)
                self._exit_code = 1
            elif self._exit_code == 0:  # If stopped, but exception occurred during shutdown
                self._logger.warning(f"Exception occurred during shutdown sequence: {e}", exc_info=True)
                self._exit_code = 1  # Still indicate an issue occurred

            self._shutdown()  # Ensure shutdown is signaled on error

        finally:

            # Wait for the viewer thread to complete ONLY IF the viewer was started
            # and the simulation finished normally (or was stopped gracefully).
            # If there was a fatal error causing an early exit from the try block,
            # we might already be trying to shut down the viewer via _shutdown().
            if self._is_viewer_enabled and self._viewer_thread:
                if self._viewer_thread.is_alive():
                    self._logger.info("Simulation complete. Waiting for viewer window to be closed...")
                    # Wait indefinitely for the viewer thread to finish
                    # This thread only finishes when the user closes the viewer window
                    self._viewer_thread.join()
                    self._logger.info("Viewer thread finished after viewer window closed")
                else:
                    # If the viewer thread isn't alive here, it might have stopped
                    # due to an error reported by the listener, or shut down early.
                    self._logger.info("Viewer thread already finished when checked in finally block")
            else:
                # Update headless completion log message
                if not self._is_viewer_enabled:
                    completion_status = "completed"
                    # Check exit code set by KeyboardInterrupt or other errors
                    if self._exit_code == 130:
                        completion_status = "stopped by signal (KeyboardInterrupt)"
                    elif self._exit_code != 0:
                        completion_status = f"finished with error code {self._exit_code}"
                    self._logger.info(f"Headless simulation {completion_status}")

            self._logger.info("Trading simulation stopped")

            return self._exit_code

    def _simulation_loop(self):

        self._logger.info(f"Running simulation loop for a max of {self._max_duration.value} {self._max_duration.time_unit}...")
        self._trader.algorithm.on_startup(self._trader.exchange)

        try:
            while self._trader.exchange.current_step < self._max_steps:

                if self._is_viewer_enabled:
                    if self._enforce_rate_limiting():
                        break  # stop event was signaled
                else:
                    # Add brief sleep to yield CPU when not updating which helps to prevent 100% CPU usage in
                    # fast loops and improves responsiveness
                    if self._is_stopping.is_set():   # Check stop event before sleep
                        break
                    time.sleep(0.005)  # Yield briefly (5 ms)

                # Take a single step in the exchange simulation
                if self._trader.step():
                    self._is_stopping.set()
                    self._logger.info("Simulation loop stopping early!")
                    break

                if self._is_viewer_enabled:
                    for update in self._build_view_updates():
                        self._viewer.view_update_queue.put(update)

        except Exception as e:

            if not self._is_stopping.is_set():
                # Log error only if not caused by stop event
                self._logger.error(f"Error occurred while running simulation loop: {e}", exc_info=True)
            else:
                # Log as warning if exception occurs during stop sequence
                self._logger.warning(f"Error occurred during simulation loop shutdown: {e}", exc_info=True)

            # Set error code only if not already set (e.g., by KeyboardInterrupt)
            if self._exit_code == 0:
                self._exit_code = 1

        finally:

            # Ensure stop is set reliably upon exiting the loop, regardless of reason
            self._is_stopping.set()

        self._trader.algorithm.on_shutdown(self._trader.exchange)
        self._logger.info(f"Simulation loop stopped after {self._trader.exchange.current_step} of {self._max_steps} steps.")

    def _build_view_updates(self) -> list[TraderViewDataUpdate]:

        current_step = self._trader.exchange.current_step
        step_duration_seconds = self._trader.exchange.settings.step_duration_seconds
        steps_per_candle = int(TimeUnit.MINUTE.seconds / step_duration_seconds)

        current_time = timedelta(seconds=(current_step - 1) * step_duration_seconds)

        updates = []
        for symbol in self._trader.algorithm.symbols:

            last_price = self._trader.exchange.get_ticker(symbol).last_price

            if (current_step - 1) % steps_per_candle == 0:
                self._viewer_data[symbol] = {
                    'ohlc': {
                        'time': int(current_time.total_seconds() / TimeUnit.MINUTE.seconds),
                        'open': last_price,
                        'high': last_price,
                        'low': last_price
                    }
                    # 'time': current_time,
                    # 'ohlc': Ohlc(
                    #     time=int(current_time.total_seconds() / TimeUnit.MINUTE.seconds),
                    #     open=last_price
                    # )
                    # 'ohlc': Ohlc(
                    #     time=int(current_time.total_seconds() / TimeUnit.MINUTE.seconds),
                    #     open=last_price, high=last_price, low=last_price, close=last_price
                    # )
                }
            viewer_data = self._viewer_data[symbol]

            # viewer_data['time'] = current_time
            viewer_data['ohlc']['high'] = max(viewer_data['ohlc']['high'], last_price)
            viewer_data['ohlc']['low'] = min(viewer_data['ohlc']['low'], last_price)
            viewer_data['ohlc']['close'] = last_price

            # viewer_data['ohlc'].high = max(viewer_data['ohlc'].high, last_price)
            # viewer_data['ohlc'].low = min(viewer_data['ohlc'].low, last_price)
            # viewer_data['ohlc'].close = last_price

            viewer_data['indicators'] = []
            for name, indicator in self._trader.algorithm.indicators.items():
                viewer_data['indicators'].append({
                    'name': name,
                    'value': indicator.value,
                    'is_price_indicator': True
                })

            # price_ema_60 = self._trader.algorithm.get_indicator("price_ema_60").value

            updates.append(TraderViewDataUpdate(
                symbol=symbol,
                time=current_time,
                data=copy.deepcopy(viewer_data)
            ))

        return updates

    def _enforce_rate_limiting(self) -> bool:

        # ---> what about long running steps???

        current_time = time.monotonic()
        time_delta = current_time - self._last_step_time

        if time_delta < self._min_step_delay_seconds:
            sleep_time = self._min_step_delay_seconds - time_delta
            if self._is_stopping.is_set():
                return True  # Check before sleep attempt
            wakeup_time = time.monotonic() + sleep_time
            while time.monotonic() < wakeup_time and not self._is_stopping.is_set():
                remaining_sleep = min(.05, max(.0, wakeup_time - time.monotonic()))  # sleep up to 50ms
                # Sleep for short intervals checking the stop event
                time.sleep(remaining_sleep)
            if self._is_stopping.is_set():
                return True  # Check again after sleep attempt

        self._last_step_time = time.monotonic()

        return False

    def _shutdown(self):

        # Check if stop hasn't already been signaled to avoid redundant actions/logs
        if not self._is_stopping.is_set():
            self._logger.info("Stop requested")
            self._is_stopping.set()
            # Request viewer close *only if* it seems to still be running
            # (Check instance and potentially thread liveness)
            # The check ensures we don't try to close a viewer that never started or already stopped
            if (self._is_viewer_enabled and self._viewer and self._viewer_thread
                and self._viewer_thread.is_alive()
            ):
                self._logger.info("Requesting viewer to close.")
                self._viewer.request_browser_close()
        else:
            # Log at debug level if stop is called when already stopping
            self._logger.debug("Stop already requested or in progress.")

    def on_dash_app_started(self, app: EmbeddedDashApplication):

        # Check if app is the viewer instance being managed
        if app is self._viewer:
            self._logger.info("Listener: Viewer reported started.")
            self._is_viewer_ready.set()

    def on_dash_app_stopped(self, app: EmbeddedDashApplication, exit_code: int):

        # Check if app is the viewer instance being managed
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
            self._shutdown()  # Call own _shutdown method to halt simulation loop
