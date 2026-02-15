from decimal import Decimal

from pytrader.exchange.aster.aster_exchange import AsterExchange


def test_get_recent_klines_parses_list_payload_rows():
    exchange = object.__new__(AsterExchange)

    class _DummyClient:
        @staticmethod
        def get_klines(symbol: str, interval: str, limit: int):
            assert symbol == "BTCUSDT"
            assert interval == "1m"
            assert limit == 2
            return [
                [1_700_000_000_000, "50000", "50100", "49900", "50050", "12.3", 1_700_000_059_999],
                [1_700_000_060_000, "50050", "50200", "50000", "50180", "13.1", 1_700_000_119_999],
            ]

    exchange._client = _DummyClient()

    rows = exchange.get_recent_klines(symbol="btcusdt", interval="1m", limit=2)

    assert len(rows) == 2
    assert rows[0]["open_time_ms"] == 1_700_000_000_000
    assert rows[0]["close_time_ms"] == 1_700_000_059_999
    assert rows[0]["open"] == Decimal("50000")
    assert rows[0]["high"] == Decimal("50100")
    assert rows[0]["low"] == Decimal("49900")
    assert rows[0]["close"] == Decimal("50050")


def test_get_recent_klines_parses_dict_payload_rows_and_filters_invalid_entries():
    exchange = object.__new__(AsterExchange)

    class _DummyClient:
        @staticmethod
        def get_klines(symbol: str, interval: str, limit: int):
            assert symbol == "ETHUSDT"
            assert interval == "5m"
            assert limit == 3
            return [
                {
                    "openTime": 1_700_000_000_000,
                    "closeTime": 1_700_000_299_999,
                    "o": "2500",
                    "h": "2520",
                    "l": "2490",
                    "c": "2510",
                },
                {
                    "openTime": 1_700_000_300_000,
                    "closeTime": 1_700_000_599_999,
                    "open": "2510",
                    "high": "2530",
                    "low": "2505",
                    "close": "2522",
                },
                {
                    "openTime": 1_700_000_600_000,
                    "closeTime": 1_700_000_899_999,
                    "open": None,
                    "high": "2535",
                    "low": "2510",
                    "close": "2528",
                },
            ]

    exchange._client = _DummyClient()

    rows = exchange.get_recent_klines(symbol="ETHUSDT", interval="5m", limit=3)

    assert len(rows) == 2
    assert rows[0]["open"] == Decimal("2500")
    assert rows[1]["close"] == Decimal("2522")
