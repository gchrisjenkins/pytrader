import queue
import threading

from pytrader.exchange import MarketDataType
from pytrader.exchange.aster.aster_exchange_client import AsterExchangeClient


def _build_client() -> AsterExchangeClient:
    return AsterExchangeClient(
        {
            "api_key": "test-key",
            "secret": "test-secret",
            "provider_name": "aster",
            "base_url": "https://example.invalid",
            "ws_base_url": "wss://example.invalid",
        }
    )


def test_start_market_data_stream_uses_command_mode():
    client = _build_client()
    captured_modes: list[str | None] = []
    captured_has_update_queue: list[bool] = []

    async def _fake_stream(streams, *, max_messages=None, stop_event=None, mode=None, subscription_updates=None):
        _ = streams
        _ = max_messages
        _ = stop_event
        captured_modes.append(mode)
        captured_has_update_queue.append(subscription_updates is not None)
        yield "btcusdt@bookTicker", {
            "e": "bookTicker",
            "s": "BTCUSDT",
            "u": 123,
            "b": "100.0",
            "B": "1.5",
            "a": "100.1",
            "A": "2.0",
        }

    client.ws.stream = _fake_stream

    out_queue: queue.Queue = queue.Queue(maxsize=10)
    stop_event = threading.Event()
    thread = client.start_market_data_stream(
        ["btcusdt@bookTicker"],
        out_queue=out_queue,
        stop_event=stop_event,
        provider="aster",
        thread_name="TestMarketStreamThread",
    )
    thread.join(timeout=1.0)

    assert thread.is_alive() is False
    assert captured_modes == ["commands"]
    assert captured_has_update_queue == [False]
    message = out_queue.get_nowait()
    assert message.type == MarketDataType.BOOK_TOP
    assert message.symbol == "BTCUSDT"

    client.close()


def test_start_user_data_stream_keeps_combined_mode():
    client = _build_client()
    captured_modes: list[str | None] = []

    async def _fake_stream(streams, *, max_messages=None, stop_event=None, mode=None, subscription_updates=None):
        _ = streams
        _ = max_messages
        _ = stop_event
        _ = subscription_updates
        captured_modes.append(mode)
        yield None, {
            "e": "ACCOUNT_UPDATE",
            "E": 1_700_000_000_000,
            "a": {"B": [], "P": []},
        }

    client.ws.stream = _fake_stream

    out_queue: queue.Queue = queue.Queue(maxsize=10)
    stop_event = threading.Event()
    thread = client.start_user_data_stream(
        listen_key="listen-key",
        out_queue=out_queue,
        stop_event=stop_event,
        thread_name="TestUserStreamThread",
    )
    thread.join(timeout=1.0)

    assert thread.is_alive() is False
    assert captured_modes == ["combined"]
    event = out_queue.get_nowait()
    assert event["type"] == "account_update"

    client.close()
