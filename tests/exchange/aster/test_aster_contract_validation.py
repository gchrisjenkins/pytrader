import logging
from types import SimpleNamespace

import pytest

from pytrader.exchange.aster.aster_exchange import AsterExchange


def _symbol_info(
    symbol: str,
    *,
    status: str = "TRADING",
    contract_type: str | None = "PERPETUAL",
    quote_asset: str = "USDT",
    margin_asset: str = "USDT",
) -> dict:
    info = {
        "symbol": symbol,
        "status": status,
        "quoteAsset": quote_asset,
        "marginAsset": margin_asset,
        "baseAsset": symbol.replace("USDT", "") or symbol,
        "filters": [
            {"filterType": "PRICE_FILTER", "tickSize": "0.1"},
            {"filterType": "LOT_SIZE", "minQty": "0.001", "maxQty": "1000", "stepSize": "0.001"},
            {"filterType": "MIN_NOTIONAL", "notional": "5"},
        ],
    }
    if contract_type is not None:
        info["contractType"] = contract_type
    return info


class _DummyClient:
    def __init__(self, exchange_info: dict):
        self._exchange_info = exchange_info
        self.commission_requests: list[str] = []

    def get_exchange_info(self) -> dict:
        return self._exchange_info

    def get_commission_rate(self, symbol: str) -> dict:
        self.commission_requests.append(symbol)
        return {
            "makerCommissionRate": "0.0002",
            "takerCommissionRate": "0.0004",
        }


def _build_exchange(markets: list[str], exchange_info: dict) -> tuple[AsterExchange, _DummyClient]:
    exchange = object.__new__(AsterExchange)
    client = _DummyClient(exchange_info)

    exchange._config = SimpleNamespace(markets=markets)
    exchange._client = client
    exchange._markets = {}
    exchange._logger = logging.getLogger("tests.exchange.aster.test_aster_contract_validation")
    exchange._supported_linear_trading_symbols = set()
    exchange._unsupported_trading_symbols = {}

    return exchange, client


def test_init_markets_accepts_linear_trading_symbol_and_tracks_supported_symbols():
    exchange, client = _build_exchange(
        markets=["btcusdt"],
        exchange_info={
            "symbols": [
                _symbol_info("BTCUSDT"),
                _symbol_info("MBLUSDT", status="PENDING_TRADING", contract_type=None),
                _symbol_info("ETHBTC", quote_asset="BTC", margin_asset="BTC"),
            ]
        },
    )

    exchange._init_markets()

    assert set(exchange._markets.keys()) == {"BTCUSDT"}
    assert client.commission_requests == ["BTCUSDT"]
    assert exchange._supported_linear_trading_symbols == {"BTCUSDT"}
    assert "ETHBTC" in exchange._unsupported_trading_symbols
    assert "MBLUSDT" not in exchange._supported_linear_trading_symbols


def test_init_markets_rejects_configured_unsupported_contract():
    exchange, _ = _build_exchange(
        markets=["ETHBTC"],
        exchange_info={"symbols": [_symbol_info("ETHBTC", quote_asset="BTC", margin_asset="BTC")]},
    )

    with pytest.raises(ValueError, match="unsupported by the Aster linear adapter"):
        exchange._init_markets()


def test_init_markets_allows_blank_contract_type_for_trading_usdt_symbol():
    exchange, _ = _build_exchange(
        markets=["BTCUSDT"],
        exchange_info={"symbols": [_symbol_info("BTCUSDT", contract_type="")]},
    )

    exchange._init_markets()
    assert "BTCUSDT" in exchange._markets
