from decimal import Decimal

from pytrader.exchange import Order, Position
from pytrader.exchange.aster.aster_exchange import AsterExchange, AsterAccount


def _build_account() -> AsterAccount:
    account = AsterAccount(
        margin_asset="USDT",
        quote_asset="USDT",
        is_multi_asset_mode=False,
        balance=Decimal("1000"),
        unrealized_pnl=Decimal("0"),
        initial_margin_requirement=Decimal("0"),
        maintenance_margin_requirement=Decimal("0"),
        position_initial_margin_requirement=Decimal("0"),
        open_order_initial_margin_requirement=Decimal("0"),
        cross_margin_balance=Decimal("1000"),
        cross_margin_unrealized_pnl=Decimal("0"),
        available_margin_balance=Decimal("1000"),
        max_withdraw_amount=Decimal("1000"),
        can_trade=True,
        can_deposit=True,
        can_withdraw=True,
        update_timestamp=0,
    )
    account.add_wallet(
        AsterAccount.Wallet(
            asset="USDT",
            balance=Decimal("1000"),
            unrealized_pnl=Decimal("0"),
            equity=Decimal("1000"),
            initial_margin_requirement=Decimal("0"),
            maintenance_margin_requirement=Decimal("0"),
            position_initial_margin_requirement=Decimal("0"),
            open_order_initial_margin_requirement=Decimal("0"),
            cross_margin_balance=Decimal("1000"),
            cross_margin_unrealized_pnl=Decimal("0"),
            available_margin_balance=Decimal("1000"),
            max_withdraw_amount=Decimal("1000"),
            is_marginable=True,
            update_timestamp=0,
        )
    )
    return account


def _build_exchange_for_mapping() -> AsterExchange:
    exchange = object.__new__(AsterExchange)
    exchange._account = _build_account()
    return exchange


def test_apply_account_update_handles_hedge_mode_positions():
    exchange = _build_exchange_for_mapping()

    exchange._apply_account_update(
        {
            "B": [
                {"a": "USDT", "wb": "1100", "cw": "1080"},
            ],
            "P": [
                {"s": "BTCUSDT", "ps": "LONG", "pa": "0.6", "ep": "42000"},
                {"s": "BTCUSDT", "ps": "SHORT", "pa": "-0.2", "ep": "43000"},
            ],
        }
    )

    positions = exchange._account._get_positions("BTCUSDT")
    assert len(positions) == 2

    long_position = next(
        position for position in positions
        if position.mode == Position.Mode.HEDGE and position.side == Position.Side.LONG
    )
    short_position = next(
        position for position in positions
        if position.mode == Position.Mode.HEDGE and position.side == Position.Side.SHORT
    )

    assert long_position.quantity == Decimal("0.6")
    assert long_position.entry_price == Decimal("42000")
    assert short_position.quantity == Decimal("-0.2")
    assert short_position.entry_price == Decimal("43000")

    wallet = exchange._account.get_wallet("USDT")
    assert wallet is not None
    assert wallet.balance == Decimal("1100")
    assert wallet.cross_margin_balance == Decimal("1080")

    # Close only LONG leg and ensure SHORT remains.
    exchange._apply_account_update(
        {
            "P": [
                {"s": "BTCUSDT", "ps": "LONG", "pa": "0", "ep": "0"},
            ]
        }
    )
    positions = exchange._account._get_positions("BTCUSDT")
    assert len(positions) == 1
    assert positions[0].mode == Position.Mode.HEDGE
    assert positions[0].side == Position.Side.SHORT
    assert positions[0].quantity == Decimal("-0.2")


def test_upsert_order_handles_partial_fill_transition_and_missing_side():
    exchange = _build_exchange_for_mapping()

    new_payload = {
        "s": "BTCUSDT",
        "i": 12345,
        "c": "alpha-1",
        "S": "BUY",
        "o": "LIMIT",
        "f": "GTC",
        "X": "NEW",
        "q": "1.0",
        "p": "50000",
        "T": 1_700_000_000_000,
        "z": "0",
    }
    partial_payload = {
        "s": "BTCUSDT",
        "i": 12345,
        "c": "alpha-1",
        "S": "BUY",
        "o": "LIMIT",
        "f": "GTC",
        "X": "PARTIALLY_FILLED",
        "q": "1.0",
        "p": "50000",
        "T": 1_700_000_000_500,
        "z": "0.4",
        "ap": "50010",
    }
    # Side intentionally omitted to exercise fallback to existing side.
    filled_payload = {
        "s": "BTCUSDT",
        "i": 12345,
        "c": "alpha-1",
        "o": "LIMIT",
        "f": "GTC",
        "X": "FILLED",
        "q": "1.0",
        "p": "50000",
        "T": 1_700_000_001_000,
        "z": "1.0",
        "ap": "50020",
    }

    order_1 = exchange._upsert_order_from_exchange_payload(new_payload, event_ts_ms=1_700_000_000_000)
    order_2 = exchange._upsert_order_from_exchange_payload(partial_payload, event_ts_ms=1_700_000_000_500)
    order_3 = exchange._upsert_order_from_exchange_payload(filled_payload, event_ts_ms=1_700_000_001_000)

    assert order_1 is not None
    assert order_2 is not None
    assert order_3 is not None

    orders = exchange._account.get_orders("BTCUSDT", None, None)
    assert len(orders) == 1
    order = orders[0]

    assert order.id == "12345"
    assert order.side == Order.Side.BUY
    assert order.status == Order.Status.FILLED
    assert order.filled_quantity == Decimal("1.0")
    assert order.average_fill_price == Decimal("50020")


def test_upsert_order_accepts_rest_style_payload_keys():
    exchange = _build_exchange_for_mapping()

    rest_payload = {
        "symbol": "BTCUSDT",
        "orderId": 98765,
        "clientOrderId": "rest-1",
        "side": "SELL",
        "type": "LIMIT",
        "timeInForce": "GTC",
        "status": "NEW",
        "origQty": "0.25",
        "price": "69000",
        "time": 1_700_000_002_000,
        "executedQty": "0",
        "avgPrice": "0",
    }

    order = exchange._upsert_order_from_exchange_payload(rest_payload, event_ts_ms=1_700_000_002_500)
    assert order is not None

    orders = exchange._account.get_orders("BTCUSDT", None, None)
    assert len(orders) == 1
    stored = orders[0]

    assert stored.id == "98765"
    assert stored.client_id == "rest-1"
    assert stored.symbol == "BTCUSDT"
    assert stored.side == Order.Side.SELL
    assert stored.type == Order.Type.LIMIT
    assert stored.time_in_force == Order.TimeInForce.GTC
    assert stored.status == Order.Status.NEW
    assert stored.quantity == Decimal("0.25")
    assert stored.price == Decimal("69000")
    assert stored.timestamp == 1_700_000_002_000
