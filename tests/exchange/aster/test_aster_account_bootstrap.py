import logging
from decimal import Decimal

from pytrader.exchange import Position
from pytrader.exchange.aster.aster_exchange import AsterExchange


def _build_account_payload(*, include_positions: bool) -> dict:
    payload = {
        "totalWalletBalance": "1000",
        "totalUnrealizedProfit": "0",
        "totalInitialMargin": "0",
        "totalMaintMargin": "0",
        "totalPositionInitialMargin": "0",
        "totalOpenOrderInitialMargin": "0",
        "totalCrossWalletBalance": "1000",
        "totalCrossUnPnl": "0",
        "availableBalance": "1000",
        "maxWithdrawAmount": "1000",
        "canTrade": True,
        "canDeposit": True,
        "canWithdraw": True,
        "updateTime": 0,
        "assets": [
            {
                "asset": "USDT",
                "walletBalance": "1000",
                "unrealizedProfit": "0",
                "marginBalance": "1000",
                "initialMargin": "0",
                "maintMargin": "0",
                "positionInitialMargin": "0",
                "openOrderInitialMargin": "0",
                "crossWalletBalance": "1000",
                "crossUnPnl": "0",
                "availableBalance": "1000",
                "maxWithdrawAmount": "1000",
                "marginAvailable": True,
                "updateTime": 0,
            }
        ],
    }
    if include_positions:
        payload["positions"] = [
            {
                "symbol": "BTCUSDT",
                "positionSide": "BOTH",
                "positionAmt": "0.005",
                "entryPrice": "67000",
            }
        ]
    return payload


def test_init_account_seeds_positions_from_account_snapshot():
    exchange = object.__new__(AsterExchange)
    exchange._logger = logging.getLogger("tests.exchange.aster.test_aster_account_bootstrap")

    class _DummyClient:
        @staticmethod
        def get_account() -> dict:
            return _build_account_payload(include_positions=True)

        @staticmethod
        def get_is_multi_asset_mode() -> bool:
            return False

        @staticmethod
        def get_positions() -> list[dict]:
            raise AssertionError("positionRisk fallback should not be used when account snapshot has positions")

    exchange._client = _DummyClient()

    exchange._init_account()

    positions = exchange._account._get_positions("BTCUSDT")
    assert len(positions) == 1
    assert positions[0].mode == Position.Mode.NET
    assert positions[0].side is None
    assert positions[0].quantity == Decimal("0.005")
    assert positions[0].entry_price == Decimal("67000")


def test_init_account_falls_back_to_position_risk_when_account_snapshot_has_no_positions():
    exchange = object.__new__(AsterExchange)
    exchange._logger = logging.getLogger("tests.exchange.aster.test_aster_account_bootstrap")

    class _DummyClient:
        @staticmethod
        def get_account() -> dict:
            return _build_account_payload(include_positions=False)

        @staticmethod
        def get_is_multi_asset_mode() -> bool:
            return False

        @staticmethod
        def get_positions() -> list[dict]:
            return [
                {
                    "symbol": "BTCUSDT",
                    "positionSide": "LONG",
                    "positionAmt": "0.010",
                    "entryPrice": "68000",
                }
            ]

    exchange._client = _DummyClient()

    exchange._init_account()

    positions = exchange._account._get_positions("BTCUSDT")
    assert len(positions) == 1
    assert positions[0].mode == Position.Mode.HEDGE
    assert positions[0].side == Position.Side.LONG
    assert positions[0].quantity == Decimal("0.010")
    assert positions[0].entry_price == Decimal("68000")


def test_init_account_seeds_open_orders_from_open_orders_endpoint():
    exchange = object.__new__(AsterExchange)
    exchange._logger = logging.getLogger("tests.exchange.aster.test_aster_account_bootstrap")

    class _DummyClient:
        @staticmethod
        def get_account() -> dict:
            return _build_account_payload(include_positions=False)

        @staticmethod
        def get_is_multi_asset_mode() -> bool:
            return False

        @staticmethod
        def get_positions() -> list[dict]:
            return []

        @staticmethod
        def get_open_orders() -> list[dict]:
            return [
                {
                    "symbol": "BTCUSDT",
                    "orderId": 123456,
                    "clientOrderId": "bootstrap-order",
                    "side": "BUY",
                    "type": "LIMIT",
                    "timeInForce": "GTC",
                    "status": "NEW",
                    "origQty": "0.020",
                    "executedQty": "0",
                    "price": "66000",
                    "time": 1_700_000_010_000,
                    "avgPrice": "0",
                }
            ]

    exchange._client = _DummyClient()

    exchange._init_account()

    orders = exchange._account.get_orders("BTCUSDT", None, None)
    assert len(orders) == 1
    order = orders[0]
    assert order.id == "123456"
    assert order.client_id == "bootstrap-order"
    assert order.side.name == "BUY"
    assert order.status.name == "NEW"
    assert order.price == Decimal("66000")
    assert order.quantity == Decimal("0.020")
