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
    account.add_position(
        Position(
            symbol="BTCUSDT",
            mode=Position.Mode.NET,
            quantity=Decimal("0.1"),
            entry_price=Decimal("60000"),
        )
    )
    account.add_order(
        Order(
            id="1",
            client_id="seed",
            symbol="BTCUSDT",
            type=Order.Type.LIMIT,
            time_in_force=Order.TimeInForce.GTC,
            side=Order.Side.BUY,
            price=Decimal("59000"),
            quantity=Decimal("0.1"),
            timestamp=1_700_000_000_000,
            status=Order.Status.NEW,
        )
    )
    return account


def _build_feed() -> LiveTraderViewFeed:
    account = _build_account()
    exchange = SimpleNamespace(
        get_account=lambda _currency: account,
        get_market_state=lambda _symbol: None,
    )
    strategy = SimpleNamespace(symbols=["BTCUSDT"], indicators={}, currency="USDT")
    trader = SimpleNamespace(exchange=exchange, strategy=strategy, state={}, action={})
    return LiveTraderViewFeed(trader)


def test_build_seed_updates_resets_history_and_populates_candles():
    feed = _build_feed()
    candles = [
        {
            "open_time_ms": 1_700_000_000_000,
            "close_time_ms": 1_700_000_059_999,
            "open": Decimal("50000"),
            "high": Decimal("50100"),
            "low": Decimal("49900"),
            "close": Decimal("50050"),
        },
        {
            "open_time_ms": 1_700_000_060_000,
            "close_time_ms": 1_700_000_119_999,
            "open": Decimal("50050"),
            "high": Decimal("50200"),
            "low": Decimal("50000"),
            "close": Decimal("50180"),
        },
    ]

    updates = feed.build_seed_updates(symbol="btcusdt", candles=candles)

    assert updates[0].reset_history is True
    assert len(updates) == 1
    assert updates[0].frame is not None
    assert updates[0].frame.ohlc is not None
    assert updates[0].frame.ohlc.close == Decimal("50180")
    assert len(updates[0].seed_points) == 2


def test_set_candle_interval_clears_cached_symbol_candles():
    feed = _build_feed()
    seeded = feed.build_seed_updates(
        symbol="BTCUSDT",
        candles=[
            {
                "open_time_ms": 1_700_000_000_000,
                "close_time_ms": 1_700_000_059_999,
                "open": Decimal("50000"),
                "high": Decimal("50100"),
                "low": Decimal("49900"),
                "close": Decimal("50050"),
            }
        ],
    )
    assert seeded
    assert "BTCUSDT" in feed._candle_by_symbol

    changed = feed.set_candle_interval("5m")

    assert changed is True
    assert feed.candle_interval == "5m"
    assert feed._candle_by_symbol == {}
