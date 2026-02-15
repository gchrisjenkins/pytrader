from decimal import Decimal
from types import SimpleNamespace

from pytrader.exchange import Order, Position
from pytrader.exchange.aster.aster_exchange import AsterAccount
from pytrader.viewer.live_view_feed import LiveTraderViewFeed


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


def _build_feed(account: AsterAccount) -> LiveTraderViewFeed:
    exchange = SimpleNamespace(get_account=lambda _currency: account)
    strategy = SimpleNamespace(symbols=["BTCUSDT"], indicators={}, currency="USDT")
    trader = SimpleNamespace(exchange=exchange, strategy=strategy, state={}, action={})
    return LiveTraderViewFeed(trader)


def test_live_view_feed_includes_off_strategy_positions():
    account = _build_account()
    account.add_position(
        Position(
            symbol="ETHUSDT",
            mode=Position.Mode.NET,
            quantity=Decimal("0.5"),
            entry_price=Decimal("2800"),
        )
    )
    feed = _build_feed(account)

    rows = feed._build_position_rows(account=account)

    assert any(row.symbol == "ETHUSDT" for row in rows)


def test_live_view_feed_includes_off_strategy_orders():
    account = _build_account()
    account.add_order(
        Order(
            id="123",
            client_id="test-order",
            symbol="ETHUSDT",
            type=Order.Type.LIMIT,
            time_in_force=Order.TimeInForce.GTC,
            side=Order.Side.BUY,
            price=Decimal("2700"),
            quantity=Decimal("0.4"),
            timestamp=1_700_000_000_000,
            status=Order.Status.NEW,
            filled_quantity=Decimal("0"),
            average_fill_price=None,
        )
    )
    feed = _build_feed(account)

    rows = feed._build_order_rows(account=account)

    assert any(row.symbol == "ETHUSDT" for row in rows)
