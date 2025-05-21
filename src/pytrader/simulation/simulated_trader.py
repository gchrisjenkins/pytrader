from pytrader import Algorithm, SimulatedExchange
from pytrader import Trader


class SimulatedTrader(Trader[SimulatedExchange]):

    def __init__(self, exchange: SimulatedExchange, algorithm: Algorithm, steps_per_trade: int):
        super().__init__(exchange, algorithm)

        self._steps_per_trade: int = steps_per_trade

    def step(self) -> None:

        while True:
            current_step = self._exchange.step()
            if self._algorithm.has_terminated:
                break

            if current_step % self._steps_per_trade == 0:
                self._algorithm.on_update(self._exchange)
                if self._algorithm.is_ready:
                    self._algorithm.on_trade(self._exchange)
                break

    def run(self, max_steps: int):

        self._algorithm.on_startup(self._exchange)

        for _ in range(max_steps):
            self.step()
            if self._algorithm.has_terminated:
                break

        self._algorithm.on_shutdown(self._exchange)
