import hashlib
import hmac
import json
import time
import asyncio
import logging
import queue
import threading
from typing import Any, Optional
from urllib.parse import urlencode

import requests

from pytrader.exchange.exchange import (
    Order,
    convert_to_decimal,
    MarketDataType,
    MarketDataSource,
    MarketDataMessage,
    NormalizedMarketDataMessage,
    MarketTradingStatus,
    BookTopPayload,
    BookTopMarketDataMessage,
    BookLevelPayload,
    BookSnapshotPayload,
    BookSnapshotMarketDataMessage,
    BookDeltaPayload,
    BookDeltaMarketDataMessage,
    TradePayload,
    TradeMarketDataMessage,
    MarkFundingPayload,
    MarkFundingMarketDataMessage,
    MarketStatusPayload,
    MarketStatusMarketDataMessage,
    OpenInterestPayload,
    OpenInterestMarketDataMessage,
    LiquidationPayload,
    LiquidationMarketDataMessage,
)


class AsterExchangeClient:
    """
    A client for interacting with the Aster Perpetuals API.
    Closely follows the Binance Futures API structure (REST endpoints, signing, etc.).

    Base URL: https://fapi.asterdex.com
    Supports public market data and private trading/account endpoints.

    WebSocket support:
      - Market streams base: wss://fstream.asterdex.com
      - Single stream: /ws/<streamName>
      - Combined streams: /stream?streams=<s1>/<s2>/...

    Stream names must be lowercase.
    """

    BASE_URL = "https://fapi.asterdex.com"

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the client.

        config: dict with keys:
            - 'api_key': your API key
            - 'secret': your API secret
            - optional 'recv_window': int (default 5000 ms)
            - optional 'timeout': float/int (default 10 seconds)
            - optional 'base_url': str

        WebSocket config keys:
            - optional 'ws_base_url': str (default wss://fstream.asterdex.com)
            - optional 'ws_mode': "combined" or "commands" (default "combined")
            - optional 'ws_reconnect': bool (default True)
            - optional 'ws_backoff_seconds': list[int|float] or number (default [1,2,5,10,20])
            - optional 'ws_ping_interval': int seconds (default 300)
            - optional 'ws_ping_timeout': int seconds (default 900)
        """
        self.api_key = config["api_key"]
        self.secret = config["secret"].encode("utf-8")
        self.recv_window = config.get("recv_window", 5000)
        self.timeout = config.get("timeout", 10)
        self.base_url = config.get("base_url", self.BASE_URL)
        self.provider_name = config.get("provider_name", "aster")

        cls = type(self)
        self._logger = logging.getLogger(f"{cls.__module__}.{cls.__name__}")

        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/x-www-form-urlencoded",
            "X-MBX-APIKEY": self.api_key
        })

        # WebSocket helper (inner class instance)
        self.ws = AsterExchangeClient.WebSocketClient(config)

    def close(self) -> None:
        self.session.close()

    # ---------------------- Param encoding + requests ----------------------

    @staticmethod
    def _encode_params(params: dict[str, Any]) -> str:
        normalized: dict[str, Any] = {}
        for k, v in (params or {}).items():
            if v is None:
                continue
            if isinstance(v, (list, dict)):
                normalized[k] = json.dumps(v, separators=(",", ":"))
            elif isinstance(v, bool):
                normalized[k] = "true" if v else "false"
            else:
                normalized[k] = v
        return urlencode(normalized)

    def _public_request(self, method: str, path: str, params: dict = None) -> Any:
        url = self.base_url + path
        if params:
            url += "?" + self._encode_params(params)
        response = self.session.request(method, url, timeout=self.timeout)
        return self._handle_response(response)

    def _api_key_request(self, method: str, path: str, params: dict = None) -> Any:
        url = self.base_url + path
        if params:
            url += "?" + self._encode_params(params)
        response = self.session.request(method, url, timeout=self.timeout)
        return self._handle_response(response)

    def _private_request(self, method: str, path: str, params: dict = None) -> Any:
        params = params or {}
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = self.recv_window

        query_string = self._encode_params(params)
        signature = hmac.new(self.secret, query_string.encode("utf-8"), hashlib.sha256).hexdigest()

        url = self.base_url + path + "?" + query_string + "&signature=" + signature
        response = self.session.request(method, url, timeout=self.timeout)
        return self._handle_response(response)

    @staticmethod
    def _handle_response(response: requests.Response) -> Any:
        data: dict[str, Any] = {}
        try:
            data = response.json()
        except ValueError:
            response.raise_for_status()

        if isinstance(data, dict) and "code" in data and "msg" in data and data["code"] < 0:
            raise Exception(f"API Error {data['code']}: {data['msg']}")

        if response.status_code != 200:
            msg = data.get("msg", response.reason) if isinstance(data, dict) else response.text
            raise Exception(f"HTTP {response.status_code} - {msg}")

        return data

    # ---------------------- Public Market Data ----------------------

    def ping(self) -> dict:
        return self._public_request("GET", "/fapi/v1/ping")

    def get_server_time(self) -> dict:
        return self._public_request("GET", "/fapi/v1/time")

    def get_exchange_info(self) -> dict:
        return self._public_request("GET", "/fapi/v1/exchangeInfo")

    def get_asset_index(self, symbol: str) -> dict:
        params: dict[str, Any] = {"symbol": symbol}
        return self._public_request("GET", "/fapi/v1/assetIndex", params)

    def get_orderbook(self, symbol: str, limit: int = None) -> dict:
        params: dict[str, Any] = {"symbol": symbol}
        if limit:
            params["limit"] = limit
        return self._public_request("GET", "/fapi/v1/depth", params)

    def get_recent_trades(self, symbol: str, limit: int = None) -> list[dict[str, Any]] | dict[str, Any]:
        params: dict[str, Any] = {"symbol": symbol}
        if limit:
            params["limit"] = limit
        return self._public_request("GET", "/fapi/v1/trades", params)

    def get_klines(
        self, symbol: str, interval: str, start_time: int = None, end_time: int = None, limit: int = None
    ) -> list:
        params: dict[str, Any] = {"symbol": symbol, "interval": interval}
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        if limit:
            params["limit"] = limit
        return self._public_request("GET", "/fapi/v1/klines", params)

    def get_book_ticker(self, symbol: str = None) -> dict:
        params = {}
        if symbol:
            params["symbol"] = symbol
        return self._public_request("GET", "/fapi/v1/ticker/bookTicker", params)

    def get_ticker_price(self, symbol: str = None) -> dict:
        params = {}
        if symbol:
            params["symbol"] = symbol
        return self._public_request("GET", "/fapi/v1/ticker/price", params)

    def get_mark_price(self, symbol: str = None) -> dict[str, Any] | list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol.upper()
        return self._public_request("GET", "/fapi/v1/premiumIndex", params)

    def get_open_interest(self, symbol: str) -> dict:
        params: dict[str, Any] = {"symbol": symbol.upper()}
        return self._public_request("GET", "/fapi/v1/openInterest", params)

    # ---------------------- Private Trading & Account ----------------------
    def get_is_multi_asset_mode(self) -> bool:

        try:
            data = self._private_request("GET", "/fapi/v1/multiAssetsMargin")
            return data.get("multiAssetsMargin", False)
        except Exception as e:
            raise RuntimeError("Failed to fetch multiAssetsMargin flag from Aster exchange") from e

    def get_account(self) -> dict:

        try:
            return self._private_request("GET", "/fapi/v4/account")
        except Exception as e:
            raise RuntimeError("Failed to fetch account information from Aster exchange") from e

    def get_commission_rate(self, symbol: str) -> dict:

        params = {"symbol": symbol.upper()}

        return self._private_request("GET", "/fapi/v1/commissionRate", params)

    def get_balance(self) -> list:
        return self._private_request("GET", "/fapi/v2/balance")

    def get_positions(self, symbol: str = None) -> list:
        params = {}
        if symbol:
            params["symbol"] = symbol
        return self._private_request("GET", "/fapi/v2/positionRisk", params)

    def get_position_mode(self) -> bool:
        data = self._private_request("GET", "/fapi/v1/positionSide/dual")
        return data["dualSidePosition"]

    def set_position_mode(self, hedge: bool = True) -> dict:
        return self._private_request(
            "POST",
            "/fapi/v1/positionSide/dual",
            {"dualSidePosition": "true" if hedge else "false"}
        )

    def create_order(self, position_side: str = None, **params) -> dict:
        if position_side is not None:
            params["positionSide"] = position_side.upper()
        return self._private_request("POST", "/fapi/v1/order", params)

    def create_batch_orders(self, batch_orders: list) -> dict:
        return self._private_request("POST", "/fapi/v1/batchOrders", {"batchOrders": batch_orders})

    def cancel_order(self, symbol: str, order_id: int = None, orig_client_order_id: str = None) -> dict:
        params: dict[str, Any] = {"symbol": symbol}
        if order_id:
            params["orderId"] = order_id
        elif orig_client_order_id:
            params["origClientOrderId"] = orig_client_order_id
        else:
            raise ValueError("Either order_id or orig_client_order_id must be provided")
        return self._private_request("DELETE", "/fapi/v1/order", params)

    def cancel_all_open_orders(self, symbol: str) -> dict:
        return self._private_request("DELETE", "/fapi/v1/allOpenOrders", {"symbol": symbol})

    def get_open_orders(self, symbol: str = None) -> list:
        params = {}
        if symbol:
            params["symbol"] = symbol
        return self._private_request("GET", "/fapi/v1/openOrders", params)

    def get_order(self, symbol: str, order_id: int = None, orig_client_order_id: str = None) -> dict:
        params: dict[str, Any] = {"symbol": symbol}
        if order_id:
            params["orderId"] = order_id
        elif orig_client_order_id:
            params["origClientOrderId"] = orig_client_order_id
        return self._private_request("GET", "/fapi/v1/order", params)

    def create_listen_key(self) -> dict:
        return self._api_key_request("POST", "/fapi/v1/listenKey")

    def keepalive_listen_key(self) -> dict:
        return self._api_key_request("PUT", "/fapi/v1/listenKey")

    def close_listen_key(self) -> dict:
        return self._api_key_request("DELETE", "/fapi/v1/listenKey")

    # ---------------------- WebSocket stream name helpers ----------------------

    @staticmethod
    def ws_stream_ticker(symbol: str) -> str:
        # 24hr rolling ticker: includes last price (c) and last qty (Q)
        return f"{symbol.lower()}@ticker"

    @staticmethod
    def ws_stream_book_ticker(symbol: str) -> str:
        # best bid/ask updates
        return f"{symbol.lower()}@bookTicker"

    @staticmethod
    def ws_stream_mark_price(symbol: str, speed: str | None = None) -> str:
        # mark/index/funding: <symbol>@markPrice or <symbol>@markPrice@1s
        if speed:
            return f"{symbol.lower()}@markPrice@{speed}"
        return f"{symbol.lower()}@markPrice"

    @staticmethod
    def ws_stream_agg_trade(symbol: str) -> str:
        # last trade-like feed (price p, qty q)
        return f"{symbol.lower()}@aggTrade"

    @staticmethod
    def ws_stream_depth(symbol: str, speed: str | None = "100ms") -> str:
        # incremental order-book updates
        if speed:
            return f"{symbol.lower()}@depth@{speed}"
        return f"{symbol.lower()}@depth"

    def build_market_streams(
        self,
        symbols: list[str],
        *,
        include_book_ticker: bool = True,
        include_trades: bool = True,
        include_mark_price: bool = True,
        include_book_delta: bool = False,
        depth_speed: str = "100ms",
    ) -> list[str]:
        streams: list[str] = []
        for symbol in symbols:
            if include_book_ticker:
                streams.append(self.ws_stream_book_ticker(symbol))
            if include_trades:
                streams.append(self.ws_stream_agg_trade(symbol))
            if include_mark_price:
                streams.append(self.ws_stream_mark_price(symbol, "1s"))
            if include_book_delta:
                streams.append(self.ws_stream_depth(symbol, speed=depth_speed))
        return streams

    @staticmethod
    def _parse_order_side(value: Any) -> Order.Side | None:
        if value is None:
            return None
        if isinstance(value, Order.Side):
            return value
        if isinstance(value, str):
            v = value.strip().upper()
            if v == "BUY":
                return Order.Side.BUY
            if v == "SELL":
                return Order.Side.SELL
        return None

    @staticmethod
    def _extract_symbol(payload: dict[str, Any], stream_name: str | None = None) -> str | None:
        symbol = payload.get("s")
        if not isinstance(symbol, str) or not symbol.strip():
            symbol = payload.get("symbol")
        if isinstance(symbol, str) and symbol.strip():
            return symbol.strip().upper()

        if stream_name and "@" in stream_name:
            symbol_from_stream = stream_name.split("@", 1)[0]
            if symbol_from_stream and not symbol_from_stream.startswith("!"):
                return symbol_from_stream.strip().upper()

        return None

    @staticmethod
    def _extract_event_timestamp(payload: dict[str, Any], recv_ts_ms: int) -> int:
        candidate = payload.get("E")
        if candidate is None:
            candidate = payload.get("T")
        if candidate is None:
            candidate = payload.get("u")
        if candidate is None:
            candidate = payload.get("time")

        try:
            return int(candidate)
        except (TypeError, ValueError):
            return recv_ts_ms

    @staticmethod
    def _parse_book_levels(levels: Any) -> list[BookLevelPayload]:
        parsed: list[BookLevelPayload] = []
        if not isinstance(levels, list):
            return parsed

        for level in levels:
            if isinstance(level, dict):
                price = level.get("price")
                quantity = level.get("quantity")
            elif isinstance(level, (list, tuple)) and len(level) >= 2:
                price = level[0]
                quantity = level[1]
            else:
                continue

            try:
                parsed.append(
                    BookLevelPayload(
                        price=convert_to_decimal(price),
                        quantity=convert_to_decimal(quantity),
                    )
                )
            except (TypeError, ValueError):
                continue

        return parsed

    def normalize_market_data_message(
        self,
        payload: dict[str, Any],
        *,
        stream_name: str | None = None,
        provider: str | None = None,
        recv_ts_ms: int | None = None,
    ) -> NormalizedMarketDataMessage | None:
        if not isinstance(payload, dict):
            return None

        provider_name = provider or self.provider_name
        recv_ms = recv_ts_ms if recv_ts_ms is not None else int(time.time() * 1000)
        event_type = payload.get("e")
        event_type = event_type if isinstance(event_type, str) else ""
        symbol = self._extract_symbol(payload, stream_name=stream_name)
        event_ts_ms = self._extract_event_timestamp(payload, recv_ms)
        raw_payload = dict(payload)

        if (
            event_type == "bookTicker"
            or (stream_name and "@bookTicker" in stream_name)
        ):
            if symbol is None:
                return None
            return BookTopMarketDataMessage(
                provider=provider_name,
                type=MarketDataType.BOOK_TOP,
                symbol=symbol,
                event_ts_ms=event_ts_ms,
                recv_ts_ms=recv_ms,
                sequence=payload.get("u"),
                source=MarketDataSource.WEBSOCKET,
                payload=BookTopPayload(
                    symbol=symbol,
                    best_bid=convert_to_decimal(payload.get("b")),
                    best_bid_quantity=convert_to_decimal(payload.get("B")),
                    best_ask=convert_to_decimal(payload.get("a")),
                    best_ask_quantity=convert_to_decimal(payload.get("A")),
                ),
                raw=raw_payload,
            )

        if (
            event_type == "depthUpdate"
            or (stream_name and "@depth" in stream_name)
        ):
            if symbol is None:
                return None
            return BookDeltaMarketDataMessage(
                provider=provider_name,
                type=MarketDataType.BOOK_DELTA,
                symbol=symbol,
                event_ts_ms=event_ts_ms,
                recv_ts_ms=recv_ms,
                sequence=payload.get("u"),
                source=MarketDataSource.WEBSOCKET,
                payload=BookDeltaPayload(
                    symbol=symbol,
                    bids=self._parse_book_levels(payload.get("b", [])),
                    asks=self._parse_book_levels(payload.get("a", [])),
                ),
                raw=raw_payload,
            )

        if "lastUpdateId" in payload and ("bids" in payload or "asks" in payload):
            if symbol is None:
                return None
            return BookSnapshotMarketDataMessage(
                provider=provider_name,
                type=MarketDataType.BOOK_SNAPSHOT,
                symbol=symbol,
                event_ts_ms=event_ts_ms,
                recv_ts_ms=recv_ms,
                sequence=payload.get("lastUpdateId"),
                source=MarketDataSource.REST,
                payload=BookSnapshotPayload(
                    symbol=symbol,
                    bids=self._parse_book_levels(payload.get("bids", [])),
                    asks=self._parse_book_levels(payload.get("asks", [])),
                ),
                raw=raw_payload,
            )

        if (
            event_type == "aggTrade"
            or (stream_name and "@aggTrade" in stream_name)
        ):
            if symbol is None:
                return None
            buyer_is_maker = payload.get("m")
            aggressor_side = None
            if isinstance(buyer_is_maker, bool):
                aggressor_side = Order.Side.SELL if buyer_is_maker else Order.Side.BUY

            return TradeMarketDataMessage(
                provider=provider_name,
                type=MarketDataType.TRADE,
                symbol=symbol,
                event_ts_ms=event_ts_ms,
                recv_ts_ms=recv_ms,
                sequence=payload.get("a"),
                source=MarketDataSource.WEBSOCKET,
                payload=TradePayload(
                    symbol=symbol,
                    price=convert_to_decimal(payload.get("p")),
                    quantity=convert_to_decimal(payload.get("q")),
                    aggressor_side=aggressor_side,
                ),
                raw=raw_payload,
            )

        if (
            event_type == "24hrTicker"
            or (stream_name and "@ticker" in stream_name)
        ):
            if symbol is None:
                return None
            return TradeMarketDataMessage(
                provider=provider_name,
                type=MarketDataType.TRADE,
                symbol=symbol,
                event_ts_ms=event_ts_ms,
                recv_ts_ms=recv_ms,
                sequence=payload.get("E"),
                source=MarketDataSource.WEBSOCKET,
                payload=TradePayload(
                    symbol=symbol,
                    price=convert_to_decimal(payload.get("c")),
                    quantity=convert_to_decimal(payload.get("Q", "0")),
                    aggressor_side=None,
                ),
                raw=raw_payload,
            )

        if (
            event_type == "markPriceUpdate"
            or (stream_name and "@markPrice" in stream_name)
        ):
            if symbol is None:
                return None
            return MarkFundingMarketDataMessage(
                provider=provider_name,
                type=MarketDataType.MARK_FUNDING,
                symbol=symbol,
                event_ts_ms=event_ts_ms,
                recv_ts_ms=recv_ms,
                sequence=payload.get("E"),
                source=MarketDataSource.WEBSOCKET,
                payload=MarkFundingPayload(
                    symbol=symbol,
                    mark_price=convert_to_decimal(payload.get("p")),
                    index_price=convert_to_decimal(payload["i"]) if payload.get("i") is not None else None,
                    funding_rate=convert_to_decimal(payload["r"]) if payload.get("r") is not None else None,
                    next_funding_time=int(payload["T"]) if payload.get("T") is not None else None,
                ),
                raw=raw_payload,
            )

        if event_type == "forceOrder":
            order_payload = payload.get("o", {})
            if not isinstance(order_payload, dict):
                return None
            symbol = self._extract_symbol(order_payload, stream_name=stream_name)
            if symbol is None:
                return None

            side = self._parse_order_side(order_payload.get("S"))
            if side is None:
                return None

            liquidation_event_ts = self._extract_event_timestamp(order_payload, event_ts_ms)
            return LiquidationMarketDataMessage(
                provider=provider_name,
                type=MarketDataType.LIQUIDATION,
                symbol=symbol,
                event_ts_ms=liquidation_event_ts,
                recv_ts_ms=recv_ms,
                sequence=order_payload.get("t"),
                source=MarketDataSource.WEBSOCKET,
                payload=LiquidationPayload(
                    symbol=symbol,
                    side=side,
                    price=convert_to_decimal(order_payload.get("p", order_payload.get("ap"))),
                    quantity=convert_to_decimal(order_payload.get("q", order_payload.get("l"))),
                ),
                raw=raw_payload,
            )

        if "openInterest" in payload and symbol is not None:
            return OpenInterestMarketDataMessage(
                provider=provider_name,
                type=MarketDataType.OPEN_INTEREST,
                symbol=symbol,
                event_ts_ms=event_ts_ms,
                recv_ts_ms=recv_ms,
                sequence=payload.get("time"),
                source=MarketDataSource.REST,
                payload=OpenInterestPayload(
                    symbol=symbol,
                    open_interest=convert_to_decimal(payload.get("openInterest")),
                    open_interest_value=(
                        convert_to_decimal(payload.get("openInterestValue"))
                        if payload.get("openInterestValue") is not None
                        else None
                    ),
                ),
                raw=raw_payload,
            )

        if "status" in payload and symbol is not None:
            status_value = payload.get("status")
            if isinstance(status_value, str):
                normalized = status_value.strip().lower()
                if normalized in {"trading", "halted", "close_only", "post_only", "settlement", "delisted"}:
                    status_enum = MarketTradingStatus(normalized)
                    return MarketStatusMarketDataMessage(
                        provider=provider_name,
                        type=MarketDataType.MARKET_STATUS,
                        symbol=symbol,
                        event_ts_ms=event_ts_ms,
                        recv_ts_ms=recv_ms,
                        sequence=payload.get("u"),
                        source=MarketDataSource.REST,
                        payload=MarketStatusPayload(symbol=symbol, status=status_enum, reason=None),
                        raw=raw_payload,
                    )

        return None

    def start_market_data_stream(
        self,
        streams: str | list[str],
        *,
        out_queue: queue.Queue[MarketDataMessage | NormalizedMarketDataMessage],
        stop_event: threading.Event,
        subscription_updates: queue.Queue[tuple[str, list[str]]] | None = None,
        provider: str | None = None,
        thread_name: str | None = None,
    ) -> threading.Thread:
        if out_queue is None:
            raise ValueError("'out_queue' is required")
        if stop_event is None:
            raise ValueError("'stop_event' is required")

        provider_name = provider or self.provider_name

        async def _consume():
            # Market-data streams run in websocket command mode so subscriptions
            # can be changed dynamically without reconnecting.
            async for stream_name, payload in self.ws.stream(
                streams,
                stop_event=stop_event,
                mode="commands",
                subscription_updates=subscription_updates,
            ):
                if stop_event.is_set():
                    return
                if not isinstance(payload, dict):
                    continue

                recv_ts_ms = int(time.time() * 1000)
                message = self.normalize_market_data_message(
                    payload,
                    stream_name=stream_name,
                    provider=provider_name,
                    recv_ts_ms=recv_ts_ms,
                )
                if message is not None:
                    while not stop_event.is_set():
                        try:
                            out_queue.put(message, timeout=0.25)
                            break
                        except queue.Full:
                            continue

        def _run():
            try:
                asyncio.run(_consume())
            except Exception:
                if not stop_event.is_set():
                    self._logger.exception("Aster market-data stream worker failed")

        thread = threading.Thread(
            target=_run,
            name=thread_name or "AsterWsMarketDataThread",
            daemon=True,
        )
        thread.start()
        return thread

    @staticmethod
    def normalize_user_data_event(payload: dict[str, Any], *, recv_ts_ms: int | None = None) -> dict[str, Any] | None:
        if not isinstance(payload, dict):
            return None

        event_type = payload.get("e")
        if not isinstance(event_type, str):
            return None

        recv_ms = recv_ts_ms if recv_ts_ms is not None else int(time.time() * 1000)
        event_ts_ms = AsterExchangeClient._extract_event_timestamp(payload, recv_ms)

        if event_type == "ACCOUNT_UPDATE":
            return {
                "type": "account_update",
                "event_ts_ms": event_ts_ms,
                "payload": payload.get("a", {}),
                "raw": payload,
            }

        if event_type == "ORDER_TRADE_UPDATE":
            order_payload = payload.get("o", {})
            symbol = order_payload.get("s") if isinstance(order_payload, dict) else None
            return {
                "type": "order_trade_update",
                "symbol": symbol.upper() if isinstance(symbol, str) else None,
                "event_ts_ms": event_ts_ms,
                "payload": order_payload if isinstance(order_payload, dict) else {},
                "raw": payload,
            }

        if event_type == "listenKeyExpired":
            return {
                "type": "listen_key_expired",
                "event_ts_ms": event_ts_ms,
                "payload": {},
                "raw": payload,
            }

        if event_type == "MARGIN_CALL":
            return {
                "type": "margin_call",
                "event_ts_ms": event_ts_ms,
                "payload": payload.get("p", []),
                "raw": payload,
            }

        return None

    def start_user_data_stream(
        self,
        *,
        listen_key: str,
        out_queue: queue.Queue[dict[str, Any]],
        stop_event: threading.Event,
        thread_name: str | None = None,
    ) -> threading.Thread:
        if not listen_key:
            raise ValueError("'listen_key' is required")
        if out_queue is None:
            raise ValueError("'out_queue' is required")
        if stop_event is None:
            raise ValueError("'stop_event' is required")

        async def _consume():
            # Keep user-data stream on the direct listen-key path.
            async for _stream_name, payload in self.ws.stream(
                listen_key,
                stop_event=stop_event,
                mode="combined",
            ):
                if stop_event.is_set():
                    return
                if not isinstance(payload, dict):
                    continue

                event = self.normalize_user_data_event(payload, recv_ts_ms=int(time.time() * 1000))
                if event is not None:
                    while not stop_event.is_set():
                        try:
                            out_queue.put(event, timeout=0.25)
                            break
                        except queue.Full:
                            continue

        def _run():
            try:
                asyncio.run(_consume())
            except Exception:
                if not stop_event.is_set():
                    self._logger.exception("Aster user-data stream worker failed")

        thread = threading.Thread(
            target=_run,
            name=thread_name or "AsterWsUserDataThread",
            daemon=True,
        )
        thread.start()
        return thread

    class WebSocketClient:
        """
        Minimal WebSocket helper for market-data streams.

        Modes:
          - ws_mode = "combined" (default):
              Connects to /stream?streams=... and yields messages.
              Reconnects by rebuilding the combined URL from the stream list.
              (No live SUBSCRIBE/UNSUBSCRIBE needed.)

          - ws_mode = "commands":
              Connects to /ws and uses SUBSCRIBE/UNSUBSCRIBE messages.
              Keeps an internal subscription set and resubscribes on reconnect.

        Note: websockets package required:
          pip install websockets
        """

        def __init__(self, config: dict[str, Any]):
            self.ws_base_url = config.get("ws_base_url", "wss://fstream.asterdex.com").rstrip("/")
            self.ws_mode = config.get("ws_mode", "combined")  # "combined" or "commands"
            self.ws_reconnect = config.get("ws_reconnect", True)
            self.ws_backoff_seconds = config.get("ws_backoff_seconds", [1, 2, 5, 10, 20])
            self.ws_ping_interval = config.get("ws_ping_interval", 300)  # seconds
            self.ws_ping_timeout = config.get("ws_ping_timeout", 900)    # seconds

            # Only used in "commands" mode
            self._subscriptions: set[str] = set()
            self._next_id = 1

        @staticmethod
        def _normalize_stream_name(stream: str) -> str:
            """
            Lowercase ONLY the symbol portion (before '@') and preserve the suffix casing
            (e.g. bookTicker, markPrice, aggTrade).
            """
            s = (stream or "").strip()
            if not s:
                return s

            # All-market streams like "!bookTicker", "!markPrice@arr@1s"
            if s.startswith("!"):
                return s

            if "@" in s:
                sym, rest = s.split("@", 1)
                return f"{sym.lower()}@{rest}"

            # Fallback: if no '@', just lowercase
            return s.lower()

        def build_url(self, streams: str | list[str]) -> str:
            if isinstance(streams, str):
                stream = self._normalize_stream_name(streams)
                return f"{self.ws_base_url}/ws/{stream}"

            streams_norm = [self._normalize_stream_name(s) for s in streams]
            joined = "/".join(streams_norm)
            return f"{self.ws_base_url}/stream?streams={joined}"

        def build_command_url(self) -> str:
            # In "commands" mode we connect to /ws and then send SUBSCRIBE messages.
            return f"{self.ws_base_url}/ws"

        @staticmethod
        def normalize_message(msg: dict[str, Any]) -> tuple[Optional[str], dict[str, Any]]:
            """
            Combined streams deliver: {"stream": "...", "data": {...}}
            Single streams often deliver just the payload dict.

            Returns (stream_name_or_none, payload_dict).
            """
            if isinstance(msg, dict) and "stream" in msg and "data" in msg and isinstance(msg["data"], dict):
                return msg.get("stream"), msg["data"]
            return None, msg

        def _make_cmd(self, method: str, params: list[str]) -> dict[str, Any]:
            cmd = {"method": method, "params": [self._normalize_stream_name(p) for p in params], "id": self._next_id}
            self._next_id += 1
            return cmd

        async def subscribe(self, websocket, streams: list[str]) -> None:
            """
            Only meaningful in ws_mode="commands".
            """
            streams_norm = [self._normalize_stream_name(s) for s in streams]
            self._subscriptions.update(streams_norm)
            cmd = self._make_cmd("SUBSCRIBE", streams_norm)
            await websocket.send(json.dumps(cmd, separators=(",", ":")))

        async def unsubscribe(self, websocket, streams: list[str]) -> None:
            """
            Only meaningful in ws_mode="commands".
            """
            streams_norm = [self._normalize_stream_name(s) for s in streams]
            for s in streams_norm:
                self._subscriptions.discard(s)
            cmd = self._make_cmd("UNSUBSCRIBE", streams_norm)
            await websocket.send(json.dumps(cmd, separators=(",", ":")))

        async def _resubscribe_all(self, websocket) -> None:
            if not self._subscriptions:
                return
            cmd = self._make_cmd("SUBSCRIBE", sorted(self._subscriptions))
            await websocket.send(json.dumps(cmd, separators=(",", ":")))

        async def stream(
            self,
            streams: str | list[str],
            *,
            max_messages: Optional[int] = None,
            stop_event: threading.Event | None = None,
            mode: str | None = None,
            subscription_updates: queue.Queue[tuple[str, list[str]]] | None = None,
        ):
            """
            Async generator yielding (stream_name_or_none, payload_dict).

            - In "combined" mode, connect URL encodes all streams; on reconnect, rebuild URL.
            - In "commands" mode, connect to /ws and SUBSCRIBE; on reconnect, resubscribe.
            """
            import websockets  # lazy import

            messages_seen = 0
            ws_mode = (mode or self.ws_mode).strip().lower()
            if ws_mode not in {"combined", "commands"}:
                raise ValueError(f"Unsupported websocket mode: {ws_mode}")
            backoffs: list[float]
            if isinstance(self.ws_backoff_seconds, (int, float)):
                backoffs = [float(self.ws_backoff_seconds)]
            else:
                backoffs = [float(x) for x in self.ws_backoff_seconds] or [1.0]

            # Normalize input stream list for combined mode (lowercase symbol only; preserve suffix)
            combined_streams: list[str]
            if isinstance(streams, str):
                combined_streams = [self._normalize_stream_name(streams)]
            else:
                combined_streams = [self._normalize_stream_name(s) for s in streams]

            attempt = 0
            while True:
                if stop_event is not None and stop_event.is_set():
                    return
                try:
                    if ws_mode == "commands":
                        url = self.build_command_url()
                        async with websockets.connect(
                            url,
                            ping_interval=self.ws_ping_interval,
                            ping_timeout=self.ws_ping_timeout,
                            close_timeout=1.0,
                        ) as ws:
                            # initial subscribe set comes from `streams`
                            if not self._subscriptions:
                                self._subscriptions = set(combined_streams)
                            else:
                                self._subscriptions.update(combined_streams)
                            await self._resubscribe_all(ws)

                            while True:
                                if stop_event is not None and stop_event.is_set():
                                    return
                                if subscription_updates is not None:
                                    while True:
                                        try:
                                            action, update_streams = subscription_updates.get_nowait()
                                        except queue.Empty:
                                            break

                                        streams_payload = [str(s) for s in update_streams]
                                        if action == "subscribe":
                                            await self.subscribe(ws, streams_payload)
                                        elif action == "unsubscribe":
                                            await self.unsubscribe(ws, streams_payload)
                                        else:
                                            self._logger.warning("Ignoring unsupported stream action: %s", action)
                                        subscription_updates.task_done()
                                try:
                                    raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                                except asyncio.TimeoutError:
                                    continue
                                msg = json.loads(raw)
                                stream_name, payload = self.normalize_message(msg)
                                yield stream_name, payload

                                messages_seen += 1
                                if max_messages is not None and messages_seen >= max_messages:
                                    return

                    else:
                        # combined mode
                        url = self.build_url(combined_streams if len(combined_streams) > 1 else combined_streams[0])
                        async with websockets.connect(
                            url,
                            ping_interval=self.ws_ping_interval,
                            ping_timeout=self.ws_ping_timeout,
                            close_timeout=1.0,
                        ) as ws:
                            while True:
                                if stop_event is not None and stop_event.is_set():
                                    return
                                try:
                                    raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                                except asyncio.TimeoutError:
                                    continue
                                msg = json.loads(raw)
                                stream_name, payload = self.normalize_message(msg)
                                yield stream_name, payload

                                messages_seen += 1
                                if max_messages is not None and messages_seen >= max_messages:
                                    return

                    # Clean exit if loop ends naturally
                    return

                except asyncio.CancelledError:
                    raise
                except Exception:
                    if stop_event is not None and stop_event.is_set():
                        return
                    if not self.ws_reconnect:
                        raise

                    # backoff + retry
                    b = backoffs[min(attempt, len(backoffs) - 1)]
                    attempt += 1
                    await asyncio.sleep(b)
                    continue
