import logging
import threading
from decimal import Decimal
from types import SimpleNamespace

import pytest

from pytrader.exchange import (
    MarketDataType,
    MarketDataSource,
    BookTopPayload,
    BookTopMarketDataMessage,
    TradePayload,
    TradeMarketDataMessage,
    MarkFundingPayload,
    MarkFundingMarketDataMessage,
    OpenInterestPayload,
    OpenInterestMarketDataMessage,
)
from pytrader.exchange.aster.aster_exchange import AsterExchange, AsterMarket


def _build_market(symbol: str = "BTCUSDT") -> AsterMarket:
    return AsterMarket(
        AsterMarket.Settings(
            symbol=symbol,
            base_asset="BTC",
            quote_asset="USDT",
            margin_asset="USDT",
            tick_size=Decimal("0.1"),
            order_increment=Decimal("0.001"),
            min_quantity=Decimal("0.001"),
            max_quantity=Decimal("1000"),
            min_notional=Decimal("5"),
            open_interest_value_multiplier=Decimal("2"),
            maker_fee=Decimal("0.0002"),
            taker_fee=Decimal("0.0004"),
        )
    )


def _build_exchange_for_warmup_tests() -> AsterExchange:
    exchange = object.__new__(AsterExchange)
    exchange._logger = logging.getLogger("tests.exchange.aster.test_aster_startup_warmup")
    exchange._markets = {"BTCUSDT": _build_market()}
    exchange._is_running = threading.Event()
    exchange._is_running.set()
    exchange._is_stopping = threading.Event()
    exchange._config = SimpleNamespace(
        provider="aster",
        open_interest_value_multiplier=Decimal("2"),
        market_data_init_timeout_sec=0.05,
        market_data_init_poll_interval_sec=0.001,
    )
    return exchange


def test_prime_symbol_market_data_from_rest_publishes_core_messages_and_derives_open_interest_value():
    exchange = _build_exchange_for_warmup_tests()

    class _DummyClient:
        @staticmethod
        def get_book_ticker(symbol: str):
            return {
                "symbol": symbol,
                "bidPrice": "100.1",
                "bidQty": "3.5",
                "askPrice": "100.2",
                "askQty": "2.0",
                "u": 42,
                "time": 1_700_000_000_001,
            }

        @staticmethod
        def get_recent_trades(symbol: str, limit: int = 1):
            _ = symbol
            _ = limit
            return [
                {
                    "id": 99,
                    "price": "100.3",
                    "qty": "0.125",
                    "time": 1_700_000_000_002,
                    "isBuyerMaker": False,
                }
            ]

        @staticmethod
        def get_mark_price(symbol: str):
            return {
                "symbol": symbol,
                "markPrice": "101.0",
                "indexPrice": "101.2",
                "lastFundingRate": "0.00001",
                "nextFundingTime": 1_700_000_360_000,
                "time": 1_700_000_000_003,
            }

        @staticmethod
        def get_open_interest(symbol: str):
            return {
                "symbol": symbol,
                "openInterest": "10",
                "time": 1_700_000_000_004,
            }

    exchange._client = _DummyClient()
    published: list = []
    exchange.publish_market_data_message = published.append

    exchange._prime_symbol_market_data_from_rest("BTCUSDT")

    assert [message.type for message in published] == [
        MarketDataType.BOOK_TOP,
        MarketDataType.TRADE,
        MarketDataType.MARK_FUNDING,
        MarketDataType.OPEN_INTEREST,
    ]

    open_interest_message = next(message for message in published if message.type == MarketDataType.OPEN_INTEREST)
    assert open_interest_message.source == MarketDataSource.REST
    assert open_interest_message.payload.open_interest == Decimal("10")
    assert open_interest_message.payload.open_interest_value == Decimal("2020")


def test_wait_for_market_data_initialization_times_out_when_required_fields_missing():
    exchange = _build_exchange_for_warmup_tests()

    with pytest.raises(RuntimeError, match="Timed out waiting for initial market data"):
        exchange._wait_for_market_data_initialization()


def test_wait_for_market_data_initialization_succeeds_when_required_fields_are_present():
    exchange = _build_exchange_for_warmup_tests()
    symbol = "BTCUSDT"
    market = exchange._markets[symbol]

    market.apply_market_data_message(
        BookTopMarketDataMessage(
            provider="test",
            symbol=symbol,
            event_ts_ms=1_700_000_000_010,
            recv_ts_ms=1_700_000_000_010,
            source=MarketDataSource.REST,
            payload=BookTopPayload(
                symbol=symbol,
                best_bid=Decimal("100.1"),
                best_bid_quantity=Decimal("1.0"),
                best_ask=Decimal("100.2"),
                best_ask_quantity=Decimal("1.5"),
            ),
        )
    )
    market.apply_market_data_message(
        TradeMarketDataMessage(
            provider="test",
            symbol=symbol,
            event_ts_ms=1_700_000_000_011,
            recv_ts_ms=1_700_000_000_011,
            source=MarketDataSource.REST,
            payload=TradePayload(
                symbol=symbol,
                price=Decimal("100.15"),
                quantity=Decimal("0.2"),
            ),
        )
    )
    market.apply_market_data_message(
        MarkFundingMarketDataMessage(
            provider="test",
            symbol=symbol,
            event_ts_ms=1_700_000_000_012,
            recv_ts_ms=1_700_000_000_012,
            source=MarketDataSource.REST,
            payload=MarkFundingPayload(
                symbol=symbol,
                mark_price=Decimal("100.16"),
                index_price=Decimal("100.20"),
                funding_rate=Decimal("0.00001"),
                next_funding_time=1_700_000_360_000,
            ),
        )
    )

    exchange._wait_for_market_data_initialization()


def test_open_interest_value_recalculates_on_mark_price_updates():
    symbol = "BTCUSDT"
    market = _build_market(symbol=symbol)

    market.apply_market_data_message(
        OpenInterestMarketDataMessage(
            provider="test",
            symbol=symbol,
            event_ts_ms=1_700_000_000_001,
            recv_ts_ms=1_700_000_000_001,
            source=MarketDataSource.REST,
            payload=OpenInterestPayload(
                symbol=symbol,
                open_interest=Decimal("10"),
                open_interest_value=None,
            ),
        )
    )
    market.apply_market_data_message(
        MarkFundingMarketDataMessage(
            provider="test",
            symbol=symbol,
            event_ts_ms=1_700_000_000_002,
            recv_ts_ms=1_700_000_000_002,
            source=MarketDataSource.WEBSOCKET,
            payload=MarkFundingPayload(
                symbol=symbol,
                mark_price=Decimal("100"),
            ),
        )
    )
    assert market.open_interest_value == Decimal("2000")

    market.apply_market_data_message(
        MarkFundingMarketDataMessage(
            provider="test",
            symbol=symbol,
            event_ts_ms=1_700_000_000_003,
            recv_ts_ms=1_700_000_000_003,
            source=MarketDataSource.WEBSOCKET,
            payload=MarkFundingPayload(
                symbol=symbol,
                mark_price=Decimal("101"),
            ),
        )
    )
    assert market.open_interest_value == Decimal("2020")
