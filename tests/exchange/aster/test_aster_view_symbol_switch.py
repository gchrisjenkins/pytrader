import logging
import queue
import threading

from pytrader.exchange.aster.aster_exchange import AsterExchange


def _build_exchange_for_view_symbol_tests() -> AsterExchange:
    exchange = object.__new__(AsterExchange)
    exchange._logger = logging.getLogger("tests.exchange.aster.test_aster_view_symbol_switch")
    exchange._core_market_symbols = {"BTCUSDT"}
    exchange._viewer_stream_symbol = None
    exchange._market_stream_update_queue = queue.Queue()
    exchange._is_running = threading.Event()
    exchange._is_running.set()
    exchange._is_stopping = threading.Event()
    exchange._all_linear_trading_market_info = {}
    exchange._markets = {}
    return exchange


def test_set_view_symbol_enqueues_subscribe_for_non_core_symbol():
    exchange = _build_exchange_for_view_symbol_tests()
    primed_symbols: list[str] = []

    exchange._ensure_market = lambda _symbol: object()
    exchange._build_symbol_market_streams = lambda symbol: [f"{symbol.lower()}@bookTicker"]
    exchange._prime_symbol_market_data_from_rest = lambda symbol: primed_symbols.append(symbol)

    assert exchange.set_view_symbol("ethusdt") is True

    action, streams = exchange._market_stream_update_queue.get_nowait()
    assert action == "subscribe"
    assert streams == ["ethusdt@bookTicker"]
    assert exchange._viewer_stream_symbol == "ETHUSDT"
    assert primed_symbols == ["ETHUSDT"]


def test_set_view_symbol_switch_unsubscribes_previous_non_core_symbol():
    exchange = _build_exchange_for_view_symbol_tests()
    exchange._viewer_stream_symbol = "ETHUSDT"
    primed_symbols: list[str] = []

    exchange._ensure_market = lambda _symbol: object()
    exchange._build_symbol_market_streams = lambda symbol: [f"{symbol.lower()}@bookTicker"]
    exchange._prime_symbol_market_data_from_rest = lambda symbol: primed_symbols.append(symbol)

    assert exchange.set_view_symbol("solusdt") is True

    first_action, first_streams = exchange._market_stream_update_queue.get_nowait()
    second_action, second_streams = exchange._market_stream_update_queue.get_nowait()
    assert (first_action, first_streams) == ("unsubscribe", ["ethusdt@bookTicker"])
    assert (second_action, second_streams) == ("subscribe", ["solusdt@bookTicker"])
    assert exchange._viewer_stream_symbol == "SOLUSDT"
    assert primed_symbols == ["SOLUSDT"]


def test_set_view_symbol_switch_to_core_symbol_only_unsubscribes_previous_non_core_symbol():
    exchange = _build_exchange_for_view_symbol_tests()
    exchange._viewer_stream_symbol = "ETHUSDT"
    primed_symbols: list[str] = []

    exchange._ensure_market = lambda _symbol: object()
    exchange._build_symbol_market_streams = lambda symbol: [f"{symbol.lower()}@bookTicker"]
    exchange._prime_symbol_market_data_from_rest = lambda symbol: primed_symbols.append(symbol)

    assert exchange.set_view_symbol("BTCUSDT") is True

    action, streams = exchange._market_stream_update_queue.get_nowait()
    assert (action, streams) == ("unsubscribe", ["ethusdt@bookTicker"])
    assert exchange._market_stream_update_queue.empty()
    assert exchange._viewer_stream_symbol == "BTCUSDT"
    assert primed_symbols == ["BTCUSDT"]
