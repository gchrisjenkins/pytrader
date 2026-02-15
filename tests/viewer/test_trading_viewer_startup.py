from datetime import timedelta
from decimal import Decimal

from dash import no_update

from pytrader.exchange import Duration, TimeUnit
from pytrader.viewer.models import (
    AccountViewState,
    MarketViewState,
    Ohlc,
    StrategyViewState,
    TraderViewDataUpdate,
    TraderViewFrame,
)
from pytrader.viewer.trading_viewer import TradingViewer, TraderViewConfiguration


def _build_viewer() -> TradingViewer:
    config = TraderViewConfiguration(
        strategy_name="StartupPrimeTest",
        symbols=["BTCUSDT"],
        currency="USDT",
        max_duration=Duration(value=1, time_unit=TimeUnit.DAY),
        time_unit=TimeUnit.MINUTE,
        max_candle_window_size=600,
    )
    return TradingViewer(listener=None, config=config)


def _build_startup_update() -> TraderViewDataUpdate:
    ohlc = Ohlc(
        time=28_333_333,
        open=Decimal("70000"),
        high=Decimal("70100"),
        low=Decimal("69900"),
        close=Decimal("70050"),
    )
    frame = TraderViewFrame(
        symbol="BTCUSDT",
        time=timedelta(milliseconds=1_700_000_000_000),
        ohlc=ohlc,
        market=MarketViewState(symbol="BTCUSDT", last_price=Decimal("70050")),
        account=AccountViewState(
            quote_asset="USDT",
            margin_asset="USDT",
            balance=Decimal("1000"),
            equity=Decimal("1000"),
            unrealized_pnl=Decimal("0"),
            available_margin_balance=Decimal("1000"),
            initial_margin_requirement=Decimal("0"),
            maintenance_margin_requirement=Decimal("0"),
        ),
        strategy=StrategyViewState(state={}, action={}),
    )
    update = TraderViewDataUpdate.from_frame(frame)
    update.reset_history = True
    update.seed_points = [(1_700_000_000_000, ohlc)]
    return update


def _find_component_by_id(component: object, component_id: str) -> object | None:
    current_id = getattr(component, "id", None)
    if current_id == component_id:
        return component

    children = getattr(component, "children", None)
    if children is None:
        return None

    if isinstance(children, (list, tuple)):
        child_items = children
    else:
        child_items = [children]

    for child in child_items:
        found = _find_component_by_id(child, component_id)
        if found is not None:
            return found

    return None


def test_trading_viewer_apply_startup_updates_forces_first_render():
    viewer = _build_viewer()
    update = _build_startup_update()

    applied_count = viewer.apply_startup_updates([update])
    first_result = viewer._update_view(0, "BTCUSDT", "1m", "orderbook", "open_orders")
    second_result = viewer._update_view(1, "BTCUSDT", "1m", "orderbook", "open_orders")

    assert applied_count == 1
    assert first_result[0] != no_update
    assert second_result[0] == no_update


def test_trading_viewer_first_render_occurs_when_startup_state_exists_without_force_flag():
    viewer = _build_viewer()
    update = _build_startup_update()
    viewer.apply_startup_updates([update])

    # Simulate a pre-startup callback having already consumed the force-render flag.
    viewer._force_full_render_next_tick = False

    result = viewer._update_view(0, "BTCUSDT", "1m", "orderbook", "open_orders")

    assert result[0] != no_update


def test_trading_viewer_skips_callback_when_previous_callback_is_running():
    viewer = _build_viewer()
    assert viewer._update_callback_lock.acquire(blocking=False)
    try:
        result = viewer._update_view(0, "BTCUSDT", "1m", "orderbook", "open_orders")
    finally:
        viewer._update_callback_lock.release()

    assert result[0] == no_update


def test_trading_viewer_figure_uses_datetime_xaxis_and_no_indicator_subplot():
    viewer = _build_viewer()
    update = _build_startup_update()
    viewer.apply_startup_updates([update])

    result = viewer._update_view(0, "BTCUSDT", "1m", "orderbook", "open_orders")
    figure = result[1]

    assert isinstance(figure, dict)
    assert figure["layout"]["xaxis"]["type"] == "date"
    assert figure["layout"]["xaxis"]["tickformat"] == "%H:%M\n%b"
    assert figure["layout"]["uirevision"] == "BTCUSDT:1m"
    assert len(figure["data"]) == 1


def test_trading_viewer_graph_enables_scroll_zoom():
    viewer = _build_viewer()
    layout = viewer._build_layout()
    graph = _find_component_by_id(layout, "viewer-candlestick-graph")

    assert graph is not None
    assert graph.config["scrollZoom"] is True


def test_trading_viewer_layout_includes_market_and_bottom_tabs():
    viewer = _build_viewer()
    layout = viewer._build_layout()

    assert _find_component_by_id(layout, "viewer-market-tabs") is not None
    assert _find_component_by_id(layout, "viewer-bottom-tabs") is not None
    assert _find_component_by_id(layout, "viewer-market-tab-content") is not None
    assert _find_component_by_id(layout, "viewer-bottom-tab-content") is not None


def test_trading_viewer_bottom_tab_switch_returns_trader_panel():
    viewer = _build_viewer()
    update = _build_startup_update()
    viewer.apply_startup_updates([update])

    result = viewer._update_view(0, "BTCUSDT", "1m", "orderbook", "trader")
    trader_panel = result[6]

    assert trader_panel is not None
