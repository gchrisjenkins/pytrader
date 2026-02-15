from decimal import Decimal
from typing import Literal

from pytrader.exchange import (
    Exchange,
    Market,
    MarketDataMessage,
    MarketDataType,
    Order,
    Position,
    Account,
    TradeMarketDataMessage,
    TradePayload,
)


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


def _message(symbol: str, type_: MarketDataType, sequence: int) -> MarketDataMessage:
    return MarketDataMessage(
        provider="test",
        type=type_,
        symbol=symbol,
        event_ts_ms=sequence,
        recv_ts_ms=sequence,
        sequence=sequence,
        payload={},
    )


class DummyExchange(Exchange["DummyExchange.Configuration", "DummyExchange.Credentials"]):

    class Configuration(Exchange.Configuration):
        provider: Literal["dummy"]

    class Credentials(Exchange.Credentials):
        pass

    class EnvOverrides(Exchange.EnvOverrides):
        pass

    def _startup(self):
        pass

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


def _build_exchange_with_market(symbol: str = "BTCUSDT") -> DummyExchange:
    exchange = DummyExchange(
        config=DummyExchange.Configuration(provider="dummy", markets=[symbol]),
        credentials=DummyExchange.Credentials(),
    )
    exchange._markets[symbol] = _build_market(symbol=symbol)
    return exchange


def test_market_backpressure_drops_non_critical_when_oldest_is_critical():
    market = _build_market()
    market.configure_market_data_queue(
        maxsize=1,
        drop_oldest_on_full=True,
        critical_types={MarketDataType.MARKET_STATUS},
    )

    critical = _message("BTCUSDT", MarketDataType.MARKET_STATUS, sequence=1)
    non_critical = _message("BTCUSDT", MarketDataType.TRADE, sequence=2)
    market.enqueue_market_data_message(critical)
    market.enqueue_market_data_message(non_critical)

    assert market.get_pending_market_data_count() == 1
    assert market.get_dropped_market_data_count() == 1

    kept = market._peek_oldest_market_data_message()
    assert kept is not None
    assert kept.type == MarketDataType.MARKET_STATUS
    assert kept.sequence == 1


def test_exchange_backpressure_keeps_critical_when_queue_is_full():
    exchange = _build_exchange_with_market()
    exchange.configure_market_data_queue(
        maxsize=1,
        drop_oldest_on_full=True,
        critical_types={MarketDataType.MARKET_STATUS},
    )

    critical = _message("BTCUSDT", MarketDataType.MARKET_STATUS, sequence=1)
    non_critical = _message("BTCUSDT", MarketDataType.TRADE, sequence=2)
    replacement = _message("BTCUSDT", MarketDataType.MARKET_STATUS, sequence=3)

    exchange.publish_market_data_message(critical)
    exchange.publish_market_data_message(non_critical)
    exchange.publish_market_data_message(replacement)

    assert exchange.get_dropped_market_data_count() == 2

    kept = exchange._peek_oldest_market_data_message()
    assert kept is not None
    assert kept.type == MarketDataType.MARKET_STATUS
    assert kept.sequence == 3


def test_route_uses_market_worker_queue_when_worker_running():
    exchange = _build_exchange_with_market()
    market = exchange._markets["BTCUSDT"]
    market.start_market_data_worker(poll_timeout_sec=0.05)

    message = TradeMarketDataMessage(
        provider="test",
        symbol="BTCUSDT",
        event_ts_ms=100,
        recv_ts_ms=101,
        payload=TradePayload(
            symbol="BTCUSDT",
            price=Decimal("50000"),
            quantity=Decimal("0.25"),
        ),
    )

    routed = exchange.route_market_data_message(message)
    assert routed is True

    # Wait briefly for worker application.
    for _ in range(20):
        if market.last_price is not None:
            break
        import time

        time.sleep(0.01)

    market.stop_market_data_worker(timeout_sec=1.0)

    assert market.last_price == Decimal("50000.0")
    assert market.last_quantity == Decimal("0.250")
