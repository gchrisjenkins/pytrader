import copy
import queue
import threading
import time
from decimal import Decimal
from typing import Literal, Any

from pydantic import Field, SecretStr, field_validator
from pydantic_core.core_schema import ValidationInfo
from pydantic_settings import SettingsConfigDict

from pytrader.exchange import (
    Exchange,
    Order,
    Position,
    Market,
    Account,
    convert_to_decimal,
    ContractType,
    MarketDataSource,
    MarketTradingStatus,
    InstrumentPayload,
    InstrumentMarketDataMessage,
    MarketStatusPayload,
    MarketStatusMarketDataMessage,
    MarketDataType,
    MarketDataMessage,
    NormalizedMarketDataMessage,
    BookTopPayload,
    BookTopMarketDataMessage,
    TradePayload,
    TradeMarketDataMessage,
    MarkFundingPayload,
    MarkFundingMarketDataMessage,
    OpenInterestPayload,
    OpenInterestMarketDataMessage,
)
from pytrader.exchange.aster import AsterExchangeClient


class AsterMarket(Market["AsterMarket.Settings"]):

    class Settings(Market.Settings):

        min_quantity: Decimal
        max_quantity: Decimal
        min_notional: Decimal
        open_interest_value_multiplier: Decimal = Decimal("1")

        @field_validator("min_quantity", "max_quantity", "min_notional", mode="before")
        @classmethod
        def _transform_required_decimal(cls, value: Any, info: ValidationInfo) -> Decimal:
            return Market.Settings._transform_required_decimal(value, info)

        @field_validator("min_quantity", "max_quantity", "min_notional", mode="after")
        @classmethod
        def _validate_positive_decimal(cls, value: Decimal, info: ValidationInfo) -> Decimal:
            return Market.Settings._validate_positive_decimal(value, info)

        @field_validator("open_interest_value_multiplier", mode="before")
        @classmethod
        def _transform_open_interest_value_multiplier(cls, value: Any) -> Decimal:
            return convert_to_decimal(value)

        @field_validator("open_interest_value_multiplier", mode="after")
        @classmethod
        def _validate_open_interest_value_multiplier(cls, value: Decimal) -> Decimal:
            if value <= 0:
                raise ValueError("'open_interest_value_multiplier' must be positive")
            return value

    def apply_market_data_message(self, message: MarketDataMessage | NormalizedMarketDataMessage):
        super().apply_market_data_message(message)

        # Keep open-interest notional value aligned with latest mark price between OI polls.
        if message.type in {MarketDataType.MARK_FUNDING, MarketDataType.OPEN_INTEREST}:
            if self.open_interest is not None and self.mark_price is not None:
                self._open_interest_value = (
                    self.open_interest
                    * self.mark_price
                    * self.settings.open_interest_value_multiplier
                )


class AsterAccount(Account["AsterAccount.Wallet"]):

    is_multi_asset_mode: bool
    max_withdraw_amount: Decimal
    update_timestamp: int

    can_trade: bool
    can_deposit: bool
    can_withdraw: bool

    class Wallet(Account.Wallet):

        max_withdraw_amount: Decimal
        is_marginable: bool
        update_timestamp: int


@Exchange.register_provider("aster")
class AsterExchange(Exchange["AsterExchange.Configuration", "AsterExchange.Credentials"]):

    def __init__(self, config: "AsterExchange.Configuration", credentials: "AsterExchange.Credentials"):
        super().__init__(config, credentials)

        self._client: AsterExchangeClient = self._create_exchange_client()
        self._supported_linear_trading_symbols: set[str] = set()
        self._unsupported_trading_symbols: dict[str, str] = {}
        self._all_linear_trading_market_info: dict[str, dict[str, Any]] = {}
        self._core_market_symbols: set[str] = set()
        self._viewer_stream_symbol: str | None = None
        self._market_stream_update_queue: queue.Queue[tuple[str, list[str]]] | None = None
        self._router_stop_event = threading.Event()
        self._router_thread: threading.Thread | None = None
        self._ws_stop_event = threading.Event()
        self._ws_market_thread: threading.Thread | None = None
        self._open_interest_poll_stop_event = threading.Event()
        self._open_interest_poll_thread: threading.Thread | None = None
        self._user_data_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._user_router_stop_event = threading.Event()
        self._user_router_thread: threading.Thread | None = None
        self._user_ws_stop_event = threading.Event()
        self._user_ws_thread: threading.Thread | None = None
        self._listen_key_keepalive_stop_event = threading.Event()
        self._listen_key_keepalive_thread: threading.Thread | None = None
        self._listen_key: str | None = None

        self._init_markets()
        self._init_account()

    def _create_exchange_client(self) -> AsterExchangeClient:

        config = {
            "api_key": self._credentials.api_key.get_secret_value(),
            "secret": self._credentials.api_secret.get_secret_value(),
            "provider_name": self._config.provider,
            "base_url": self._config.rest_url,
            "ws_base_url": self._config.ws_url,
            "recv_window": self._config.recv_window_ms,
            "timeout": self._config.request_timeout_sec,
            "ws_ping_interval": self._config.ws_ping_interval_sec,
            # The client accepts a single number or list for backoff; a single value works fine for now
            "ws_backoff_seconds": self._config.ws_reconnect_delay_sec,
        }
        client = AsterExchangeClient(config)
        del config

        return client

    @staticmethod
    def _normalize_required_symbol(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip().upper()
        return normalized if normalized else None

    @staticmethod
    def _normalize_optional_enum_like(value: Any) -> str:
        if not isinstance(value, str):
            return ""
        return value.strip().upper()

    @classmethod
    def _get_contract_rejection_reason(cls, market_info: dict[str, Any]) -> str | None:
        quote_asset = cls._normalize_required_symbol(market_info.get("quoteAsset"))
        if quote_asset is None:
            return "missing quoteAsset"

        margin_asset = cls._normalize_required_symbol(market_info.get("marginAsset"))
        if margin_asset is None:
            return "missing marginAsset"

        if quote_asset != "USDT" or margin_asset != "USDT":
            return f"unsupported quote/margin assets (quoteAsset={quote_asset}, marginAsset={margin_asset})"

        contract_type = cls._normalize_optional_enum_like(market_info.get("contractType"))
        if contract_type and contract_type != "PERPETUAL":
            return f"unsupported contractType '{contract_type}'"

        return None

    @classmethod
    def _classify_trading_contracts(
        cls,
        exchange_info: dict[str, Any],
    ) -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[str, dict[str, Any]]]:
        symbols_payload = exchange_info.get("symbols")
        if not isinstance(symbols_payload, list):
            raise RuntimeError("Invalid exchangeInfo payload: expected 'symbols' list")

        all_symbols: dict[str, dict[str, Any]] = {}
        supported_linear_trading: dict[str, dict[str, Any]] = {}
        unsupported_trading: dict[str, str] = {}

        for market_info in symbols_payload:
            if not isinstance(market_info, dict):
                continue

            symbol = cls._normalize_required_symbol(market_info.get("symbol"))
            if symbol is None:
                continue
            all_symbols[symbol] = market_info

            status = cls._normalize_optional_enum_like(market_info.get("status"))
            if status != "TRADING":
                continue

            rejection_reason = cls._get_contract_rejection_reason(market_info)
            if rejection_reason is not None:
                unsupported_trading[symbol] = rejection_reason
                continue
            supported_linear_trading[symbol] = market_info

        return all_symbols, supported_linear_trading, unsupported_trading

    def _init_markets(self) -> None:
        """
        Fetch exchangeInfo once and create a Market instance (with populated Settings) for every
        symbol listed in self._config.markets.
        """
        try:
            exchange_info = self._client.get_exchange_info()
        except Exception as e:
            raise RuntimeError("Failed to fetch exchangeInfo from Aster exchange") from e

        all_market_info_map, market_info_map, unsupported_trading = self._classify_trading_contracts(exchange_info)
        self._supported_linear_trading_symbols = set(market_info_map.keys())
        self._unsupported_trading_symbols = dict(unsupported_trading)
        self._all_linear_trading_market_info = dict(market_info_map)
        self._core_market_symbols = set()

        unsupported_count = len(self._unsupported_trading_symbols)
        if unsupported_count:
            examples = ", ".join(
                f"{symbol}: {reason}"
                for symbol, reason in list(sorted(self._unsupported_trading_symbols.items()))[:3]
            )
            if unsupported_count > 3:
                examples = f"{examples}, ... (+{unsupported_count - 3} more)"
            self._logger.warning(
                "Aster exchangeInfo has %d unsupported TRADING symbols for this adapter: %s",
                unsupported_count,
                examples,
            )

        for configured_symbol in self._config.markets:
            symbol = self._normalize_required_symbol(configured_symbol)
            if symbol is None:
                raise ValueError(f"Invalid configured symbol value: {configured_symbol!r}")

            info = market_info_map.get(symbol)
            if info is None:
                rejection_reason = self._unsupported_trading_symbols.get(symbol)
                if rejection_reason is not None:
                    raise ValueError(
                        f"Configured symbol '{symbol}' is unsupported by the Aster linear adapter: {rejection_reason}"
                    )
                status = self._normalize_optional_enum_like(all_market_info_map.get(symbol, {}).get("status"))
                if status:
                    raise ValueError(f"Configured symbol '{symbol}' is not tradable (status={status})")
                raise ValueError(
                    f"Configured symbol '{symbol}' is not present in Aster exchangeInfo"
                )
            self._markets[symbol] = self._create_market(symbol=symbol, info=info)
            self._core_market_symbols.add(symbol)

    def _create_market(self, *, symbol: str, info: dict[str, Any]) -> AsterMarket:
        base_asset = info["baseAsset"]
        quote_asset = info["quoteAsset"]
        margin_asset = info["marginAsset"]
        contract_type = self._normalize_optional_enum_like(info.get("contractType"))
        if not contract_type:
            self._logger.warning(
                "Aster exchangeInfo omitted contractType for symbol '%s'; treating as PERPETUAL",
                symbol,
            )

        filters = {f["filterType"]: f for f in info.get("filters", [])}

        price_filter = filters.get("PRICE_FILTER")
        if price_filter is None:
            raise ValueError(f"Missing PRICE_FILTER for symbol {symbol}")
        tick_size = convert_to_decimal(price_filter["tickSize"])

        lot_size_filter = filters.get("LOT_SIZE")
        if lot_size_filter is None:
            raise ValueError(f"Missing LOT_SIZE filter for symbol {symbol}")
        order_increment = convert_to_decimal(lot_size_filter["stepSize"])

        min_quantity = convert_to_decimal(lot_size_filter["minQty"])
        max_quantity = convert_to_decimal(lot_size_filter["maxQty"])

        notional_filter = filters.get("MIN_NOTIONAL")
        min_notional = convert_to_decimal(notional_filter["notional"]) if notional_filter else Decimal("0")

        try:
            commission_rate = self._client.get_commission_rate(symbol=symbol)
        except Exception as e:
            raise RuntimeError("Failed to fetch commission rate from Aster exchange") from e

        open_interest_value_multiplier = getattr(
            self._config,
            "open_interest_value_multiplier",
            Decimal("2"),
        )

        settings = AsterMarket.Settings(
            symbol=symbol,
            base_asset=base_asset,
            quote_asset=quote_asset,
            margin_asset=margin_asset,
            tick_size=tick_size,
            order_increment=order_increment,
            min_quantity=min_quantity,
            max_quantity=max_quantity,
            min_notional=min_notional,
            open_interest_value_multiplier=open_interest_value_multiplier,
            maker_fee=Decimal(commission_rate["makerCommissionRate"]),
            taker_fee=Decimal(commission_rate["takerCommissionRate"]),
        )
        return AsterMarket(settings)

    def _init_account(self):

        data = self._client.get_account()

        is_multi_asset_mode = self._client.get_is_multi_asset_mode()

        account = AsterAccount(
            margin_asset="USDT",
            quote_asset="USDT",
            is_multi_asset_mode=is_multi_asset_mode,

            balance=convert_to_decimal(data.get("totalWalletBalance", "0")),
            unrealized_pnl=convert_to_decimal(data.get("totalUnrealizedProfit", "0")),

            initial_margin_requirement=convert_to_decimal(data.get("totalInitialMargin", "0")),
            maintenance_margin_requirement=convert_to_decimal(data.get("totalMaintMargin", "0")),
            position_initial_margin_requirement=convert_to_decimal(data.get("totalPositionInitialMargin", "0")),
            open_order_initial_margin_requirement=convert_to_decimal(data.get("totalOpenOrderInitialMargin", "0")),

            cross_margin_balance=convert_to_decimal(data.get("totalCrossWalletBalance", "0")),
            cross_margin_unrealized_pnl=convert_to_decimal(data.get("totalCrossUnPnl", "0")),

            available_margin_balance=convert_to_decimal(data.get("availableBalance", "0")),
            max_withdraw_amount=convert_to_decimal(data.get("maxWithdrawAmount", "0")),

            can_trade=bool(data.get("canTrade", True)),
            can_deposit=bool(data.get("canDeposit", True)),
            can_withdraw=bool(data.get("canWithdraw", True)),

            update_timestamp=int(data.get("updateTime", 0)),
        )

        for asset_data in data.get("assets", []):
            wallet = AsterAccount.Wallet(
                asset=asset_data["asset"],

                balance=convert_to_decimal(asset_data["walletBalance"]),
                unrealized_pnl=convert_to_decimal(asset_data["unrealizedProfit"]),

                equity=convert_to_decimal(asset_data["marginBalance"]),  # Direct rename

                initial_margin_requirement=convert_to_decimal(asset_data["initialMargin"]),
                maintenance_margin_requirement=convert_to_decimal(asset_data["maintMargin"]),
                position_initial_margin_requirement=convert_to_decimal(asset_data["positionInitialMargin"]),
                open_order_initial_margin_requirement=convert_to_decimal(asset_data["openOrderInitialMargin"]),

                cross_margin_balance=convert_to_decimal(asset_data["crossWalletBalance"]),
                cross_margin_unrealized_pnl=convert_to_decimal(asset_data["crossUnPnl"]),

                available_margin_balance=convert_to_decimal(asset_data["availableBalance"]),
                max_withdraw_amount=convert_to_decimal(asset_data["maxWithdrawAmount"]),

                is_marginable=bool(asset_data.get("marginAvailable", True)),
                update_timestamp=int(asset_data["updateTime"]),
            )
            account.add_wallet(wallet)
            # account._wallets[wallet.asset] = wallet

        self._account = account

        # Seed positions from account snapshot when available so the viewer
        # reflects already-open positions immediately after startup.
        positions_seeded = self._upsert_positions_from_snapshot(data.get("positions", []))
        if not positions_seeded:
            try:
                position_risk_rows = self._client.get_positions()
            except Exception:
                self._logger.warning(
                    "Failed to fetch startup positions from positionRisk endpoint",
                    exc_info=True,
                )
            else:
                self._upsert_positions_from_snapshot(position_risk_rows)

        self._sync_open_orders_from_rest()

    def _upsert_positions_from_snapshot(self, snapshot: Any) -> bool:
        if self._account is None or not isinstance(snapshot, list):
            return False

        has_rows = False
        for row in snapshot:
            if not isinstance(row, dict):
                continue

            symbol_raw = row.get("symbol", row.get("s"))
            if not isinstance(symbol_raw, str) or not symbol_raw.strip():
                continue
            symbol = symbol_raw.strip().upper()

            side_value = row.get("positionSide", row.get("ps"))
            mode, side = self._map_position_side(side_value)
            quantity_raw = row.get("positionAmt", row.get("pa", "0"))
            entry_price_raw = row.get("entryPrice", row.get("ep", "0"))
            quantity = convert_to_decimal(quantity_raw)
            entry_price = convert_to_decimal(entry_price_raw)

            has_rows = True
            position = self._find_position(symbol, mode=mode, side=side)
            if quantity == 0:
                if position is not None:
                    self._account.delete_position(position)
                continue

            if position is None:
                self._account.add_position(Position(symbol=symbol, mode=mode, side=side))
                position = self._find_position(symbol, mode=mode, side=side)

            if position is not None:
                position.quantity = quantity
                position.entry_price = entry_price
                position.side = side

        return has_rows

    def _sync_open_orders_from_rest(self) -> None:
        if self._account is None:
            return

        try:
            payload = self._client.get_open_orders()
        except Exception:
            self._logger.warning("Failed to fetch startup open orders from openOrders endpoint", exc_info=True)
            return

        if not isinstance(payload, list):
            return

        now_ms = int(time.time() * 1000)
        for order_payload in payload:
            if not isinstance(order_payload, dict):
                continue
            self._upsert_order_from_exchange_payload(order_payload, event_ts_ms=now_ms)

    @staticmethod
    def _safe_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _extract_event_ts_ms(cls, payload: dict[str, Any], fallback_ts_ms: int) -> int:
        for key in ("time", "T", "E", "u"):
            ts = cls._safe_int(payload.get(key))
            if ts is not None and ts >= 0:
                return ts
        return fallback_ts_ms

    @staticmethod
    def _find_symbol_payload(payload: Any, symbol: str) -> dict[str, Any] | None:
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    continue
                item_symbol = item.get("symbol", item.get("s"))
                if isinstance(item_symbol, str) and item_symbol.strip().upper() == symbol:
                    return item
        return None

    def _prime_symbol_market_data_from_rest(self, symbol: str) -> None:
        recv_ts_ms = int(time.time() * 1000)
        mark_price_hint: Decimal | None = None

        try:
            raw_book_top = self._client.get_book_ticker(symbol=symbol)
            book_top = self._find_symbol_payload(raw_book_top, symbol)
            if book_top is not None:
                bid_price = book_top.get("bidPrice", book_top.get("b"))
                bid_quantity = book_top.get("bidQty", book_top.get("B"))
                ask_price = book_top.get("askPrice", book_top.get("a"))
                ask_quantity = book_top.get("askQty", book_top.get("A"))
                if (
                    bid_price is not None
                    and bid_quantity is not None
                    and ask_price is not None
                    and ask_quantity is not None
                ):
                    self.publish_market_data_message(
                        BookTopMarketDataMessage(
                            provider=self._config.provider,
                            symbol=symbol,
                            event_ts_ms=self._extract_event_ts_ms(book_top, recv_ts_ms),
                            recv_ts_ms=recv_ts_ms,
                            sequence=self._safe_int(book_top.get("u", book_top.get("lastUpdateId"))),
                            source=MarketDataSource.REST,
                            payload=BookTopPayload(
                                symbol=symbol,
                                best_bid=convert_to_decimal(bid_price),
                                best_bid_quantity=convert_to_decimal(bid_quantity),
                                best_ask=convert_to_decimal(ask_price),
                                best_ask_quantity=convert_to_decimal(ask_quantity),
                            ),
                            raw=dict(book_top),
                        )
                    )
        except Exception:
            self._logger.warning("Failed to prime REST book-top data for symbol '%s'", symbol, exc_info=True)

        try:
            raw_trades = self._client.get_recent_trades(symbol=symbol, limit=1)
            trade_payload: dict[str, Any] | None = None
            if isinstance(raw_trades, list) and raw_trades:
                candidate = raw_trades[-1]
                if isinstance(candidate, dict):
                    trade_payload = candidate
            elif isinstance(raw_trades, dict):
                trade_payload = raw_trades

            if trade_payload is not None:
                price = trade_payload.get("price", trade_payload.get("p"))
                quantity = trade_payload.get("qty", trade_payload.get("q"))
                if price is not None and quantity is not None:
                    is_buyer_maker = trade_payload.get("isBuyerMaker", trade_payload.get("m"))
                    aggressor_side = None
                    if isinstance(is_buyer_maker, bool):
                        aggressor_side = Order.Side.SELL if is_buyer_maker else Order.Side.BUY

                    self.publish_market_data_message(
                        TradeMarketDataMessage(
                            provider=self._config.provider,
                            symbol=symbol,
                            event_ts_ms=self._extract_event_ts_ms(trade_payload, recv_ts_ms),
                            recv_ts_ms=recv_ts_ms,
                            sequence=self._safe_int(trade_payload.get("id", trade_payload.get("a"))),
                            source=MarketDataSource.REST,
                            payload=TradePayload(
                                symbol=symbol,
                                price=convert_to_decimal(price),
                                quantity=convert_to_decimal(quantity),
                                aggressor_side=aggressor_side,
                            ),
                            raw=dict(trade_payload),
                        )
                    )
        except Exception:
            self._logger.warning("Failed to prime REST trade data for symbol '%s'", symbol, exc_info=True)

        try:
            raw_mark_payload = self._client.get_mark_price(symbol=symbol)
            mark_payload = self._find_symbol_payload(raw_mark_payload, symbol)
            if mark_payload is not None:
                mark_price_raw = mark_payload.get("markPrice", mark_payload.get("p"))
                if mark_price_raw is not None:
                    mark_price_hint = convert_to_decimal(mark_price_raw)
                    self.publish_market_data_message(
                        MarkFundingMarketDataMessage(
                            provider=self._config.provider,
                            symbol=symbol,
                            event_ts_ms=self._extract_event_ts_ms(mark_payload, recv_ts_ms),
                            recv_ts_ms=recv_ts_ms,
                            sequence=self._safe_int(mark_payload.get("time", mark_payload.get("E"))),
                            source=MarketDataSource.REST,
                            payload=MarkFundingPayload(
                                symbol=symbol,
                                mark_price=mark_price_hint,
                                index_price=(
                                    convert_to_decimal(mark_payload.get("indexPrice", mark_payload.get("i")))
                                    if mark_payload.get("indexPrice", mark_payload.get("i")) is not None
                                    else None
                                ),
                                funding_rate=(
                                    convert_to_decimal(mark_payload.get("lastFundingRate", mark_payload.get("r")))
                                    if mark_payload.get("lastFundingRate", mark_payload.get("r")) is not None
                                    else None
                                ),
                                next_funding_time=self._safe_int(
                                    mark_payload.get("nextFundingTime", mark_payload.get("T"))
                                ),
                            ),
                            raw=dict(mark_payload),
                        )
                    )
        except Exception:
            self._logger.warning("Failed to prime REST mark/funding data for symbol '%s'", symbol, exc_info=True)

        try:
            open_interest_payload = self._client.get_open_interest(symbol)
            if isinstance(open_interest_payload, dict):
                open_interest_raw = open_interest_payload.get("openInterest")
                if open_interest_raw is not None:
                    open_interest = convert_to_decimal(open_interest_raw)
                    open_interest_value_raw = open_interest_payload.get("openInterestValue")
                    open_interest_value = (
                        convert_to_decimal(open_interest_value_raw)
                        if open_interest_value_raw is not None
                        else None
                    )
                    if open_interest_value is None:
                        market = self._markets.get(symbol)
                        reference_mark_price = mark_price_hint or (market.mark_price if market is not None else None)
                        if reference_mark_price is not None:
                            open_interest_value = (
                                open_interest
                                * reference_mark_price
                                * self._config.open_interest_value_multiplier
                            )

                    self.publish_market_data_message(
                        OpenInterestMarketDataMessage(
                            provider=self._config.provider,
                            symbol=symbol,
                            event_ts_ms=self._extract_event_ts_ms(open_interest_payload, recv_ts_ms),
                            recv_ts_ms=recv_ts_ms,
                            sequence=self._safe_int(open_interest_payload.get("time")),
                            source=MarketDataSource.REST,
                            payload=OpenInterestPayload(
                                symbol=symbol,
                                open_interest=open_interest,
                                open_interest_value=open_interest_value,
                            ),
                            raw=dict(open_interest_payload),
                        )
                    )
        except Exception:
            self._logger.warning("Failed to prime REST open-interest data for symbol '%s'", symbol, exc_info=True)

    def _prime_market_data_from_rest(self) -> None:
        for symbol in self._markets.keys():
            self._prime_symbol_market_data_from_rest(symbol)

    @staticmethod
    def _get_missing_market_data_fields(market: Market) -> list[str]:
        missing: list[str] = []
        if market.last_price is None:
            missing.append("last_price")
        if market.last_quantity is None:
            missing.append("last_quantity")
        if market.best_bid is None:
            missing.append("best_bid")
        if market.best_bid_quantity is None:
            missing.append("best_bid_quantity")
        if market.best_ask is None:
            missing.append("best_ask")
        if market.best_ask_quantity is None:
            missing.append("best_ask_quantity")
        if market.mark_price is None:
            missing.append("mark_price")
        return missing

    def _wait_for_market_data_initialization(self) -> None:
        deadline = time.monotonic() + self._config.market_data_init_timeout_sec

        while True:
            if self._is_stopping.is_set() or not self._is_running.is_set():
                return

            missing_by_symbol: dict[str, list[str]] = {}
            for symbol, market in self._markets.items():
                missing = self._get_missing_market_data_fields(market)
                if missing:
                    missing_by_symbol[symbol] = missing

            if not missing_by_symbol:
                return

            if time.monotonic() >= deadline:
                sample = ", ".join(
                    f"{symbol}: {','.join(fields)}"
                    for symbol, fields in list(sorted(missing_by_symbol.items()))[:3]
                )
                if len(missing_by_symbol) > 3:
                    sample = f"{sample}, ... (+{len(missing_by_symbol) - 3} more)"
                raise RuntimeError(
                    "Timed out waiting for initial market data. Missing fields -> "
                    f"{sample}"
                )

            if self._is_stopping.wait(timeout=self._config.market_data_init_poll_interval_sec):
                return

    def _startup(self):
        self._logger.info("Starting Aster exchange market-data services")

        self._router_stop_event.clear()
        self._ws_stop_event.clear()
        self._open_interest_poll_stop_event.clear()
        self._user_router_stop_event.clear()
        self._user_ws_stop_event.clear()
        self._listen_key_keepalive_stop_event.clear()

        self.configure_market_data_queue(
            maxsize=self._config.exchange_market_data_queue_maxsize,
            drop_oldest_on_full=self._config.drop_oldest_on_full,
            critical_types=self._get_critical_market_data_types(),
        )
        if self._user_data_queue.qsize() > 0 and self._config.user_data_queue_maxsize != self._user_data_queue.maxsize:
            raise RuntimeError("Cannot resize user-data queue while it is non-empty")
        if self._config.user_data_queue_maxsize != self._user_data_queue.maxsize:
            self._user_data_queue = queue.Queue(maxsize=self._config.user_data_queue_maxsize)
        self._market_stream_update_queue = queue.Queue()
        self._viewer_stream_symbol = None

        self._start_market_workers()

        self._router_thread = threading.Thread(
            target=self._market_data_router_loop,
            name=f"{type(self).__name__}RouterThread",
            daemon=True,
        )
        self._router_thread.start()

        streams = self._client.build_market_streams(
            symbols=sorted(self._core_market_symbols),
            include_book_ticker=True,
            include_trades=True,
            include_mark_price=True,
            include_book_delta=self._config.subscribe_book_delta_streams,
            depth_speed=self._config.depth_stream_speed,
        )
        self._ws_market_thread = self._client.start_market_data_stream(
            streams=streams,
            out_queue=self._market_data_queue,
            stop_event=self._ws_stop_event,
            subscription_updates=self._market_stream_update_queue,
            provider=self._config.provider,
            thread_name=f"{type(self).__name__}WsThread",
        )

        self._publish_bootstrap_market_messages()
        self._prime_market_data_from_rest()
        self._start_open_interest_polling()
        self._start_user_data_services()
        if self._config.wait_for_market_data_on_startup:
            self._wait_for_market_data_initialization()
        if self._is_stopping.is_set() or not self._is_running.is_set():
            return
        self._is_ready.set()

    def _shutdown(self):
        self._logger.info("Stopping Aster exchange market-data services")
        self._is_ready.clear()
        self._ws_stop_event.set()
        self._open_interest_poll_stop_event.set()
        self._stop_user_data_services()

        join_timeout = self._config.thread_join_timeout_sec

        if self._open_interest_poll_thread and self._open_interest_poll_thread.is_alive():
            self._open_interest_poll_thread.join(timeout=join_timeout)
            if self._open_interest_poll_thread.is_alive():
                self._logger.warning("Open-interest poll thread did not stop within %.2f sec", join_timeout)
        self._open_interest_poll_thread = None

        if self._ws_market_thread and self._ws_market_thread.is_alive():
            self._ws_market_thread.join(timeout=join_timeout)
            if self._ws_market_thread.is_alive():
                self._logger.warning("Market-data websocket thread did not stop within %.2f sec", join_timeout)
        self._ws_market_thread = None
        self._market_stream_update_queue = None

        deadline = time.monotonic() + join_timeout
        while self._market_data_queue.qsize() > 0 and time.monotonic() < deadline:
            time.sleep(0.05)

        self._router_stop_event.set()

        if self._router_thread and self._router_thread.is_alive():
            self._router_thread.join(timeout=join_timeout)
            if self._router_thread.is_alive():
                self._logger.warning("Market-data router thread did not stop within %.2f sec", join_timeout)
        self._router_thread = None

        self._stop_market_workers()
        self._viewer_stream_symbol = None

        self._client.close()

    def _publish_bootstrap_market_messages(self):
        now_ms = int(time.time() * 1000)

        for symbol, market in self._markets.items():
            self._publish_bootstrap_market_messages_for_symbol(symbol=symbol, market=market, now_ms=now_ms)

    def _publish_bootstrap_market_messages_for_symbol(
        self,
        *,
        symbol: str,
        market: AsterMarket,
        now_ms: int | None = None,
    ) -> None:
        timestamp_ms = int(time.time() * 1000) if now_ms is None else now_ms

        instrument_message = InstrumentMarketDataMessage(
            provider=self._config.provider,
            symbol=symbol,
            event_ts_ms=timestamp_ms,
            recv_ts_ms=timestamp_ms,
            source=MarketDataSource.REST,
            payload=InstrumentPayload(
                symbol=symbol,
                base_asset=market.settings.base_asset,
                quote_asset=market.settings.quote_asset,
                margin_asset=market.settings.margin_asset,
                contract_type=ContractType.PERPETUAL,
                tick_size=market.settings.tick_size,
                quantity_step=market.settings.order_increment,
                min_quantity=market.settings.min_quantity if isinstance(market.settings, AsterMarket.Settings) else None,
                max_quantity=market.settings.max_quantity if isinstance(market.settings, AsterMarket.Settings) else None,
                min_notional=market.settings.min_notional if isinstance(market.settings, AsterMarket.Settings) else None,
                status=MarketTradingStatus.TRADING,
            ),
            raw=None,
        )
        self.publish_market_data_message(instrument_message)

        status_message = MarketStatusMarketDataMessage(
            provider=self._config.provider,
            symbol=symbol,
            event_ts_ms=timestamp_ms,
            recv_ts_ms=timestamp_ms,
            source=MarketDataSource.REST,
            payload=MarketStatusPayload(
                symbol=symbol,
                status=MarketTradingStatus.TRADING,
                reason="bootstrap",
            ),
            raw=None,
        )
        self.publish_market_data_message(status_message)

    def _market_data_router_loop(self):
        while not self._router_stop_event.is_set():
            message = self.poll_market_data_message(timeout_sec=self._config.router_poll_timeout_sec)
            if message is None:
                continue

            try:
                routed = self.route_market_data_message(message)
                if not routed:
                    self._logger.debug("Dropped market-data message for unknown symbol '%s'", message.symbol)
            except Exception:
                self._logger.exception("Failed to route market-data message for symbol '%s'", message.symbol)
            finally:
                self._market_data_queue.task_done()

    def _start_market_workers(self):
        for symbol, market in self._markets.items():
            self._start_market_worker_for_symbol(symbol=symbol, market=market)

    def _start_market_worker_for_symbol(self, *, symbol: str, market: AsterMarket) -> None:
        market.configure_market_data_queue(
            maxsize=self._config.market_data_queue_maxsize,
            drop_oldest_on_full=self._config.drop_oldest_on_full,
            critical_types=self._get_critical_market_data_types(),
        )
        market.start_market_data_worker(
            poll_timeout_sec=self._config.market_worker_poll_timeout_sec,
            thread_name=f"{type(self).__name__}:{symbol}:MarketWorker",
        )

    def _stop_market_workers(self):
        for market in self._markets.values():
            market.stop_market_data_worker(timeout_sec=self._config.market_worker_join_timeout_sec)

    def _build_symbol_market_streams(self, symbol: str) -> list[str]:
        return self._client.build_market_streams(
            symbols=[symbol],
            include_book_ticker=True,
            include_trades=True,
            include_mark_price=True,
            include_book_delta=self._config.subscribe_book_delta_streams,
            depth_speed=self._config.depth_stream_speed,
        )

    def _queue_market_stream_subscription_update(self, *, action: str, symbol: str) -> None:
        if self._market_stream_update_queue is None:
            return
        self._market_stream_update_queue.put((action, self._build_symbol_market_streams(symbol)))

    def _ensure_market(self, symbol: str) -> AsterMarket | None:
        market = self._markets.get(symbol)
        if market is not None:
            return market

        info = self._all_linear_trading_market_info.get(symbol)
        if info is None:
            self._logger.warning("Cannot watch symbol '%s': not supported by the Aster linear adapter", symbol)
            return None

        try:
            market = self._create_market(symbol=symbol, info=info)
        except Exception:
            self._logger.warning("Failed to initialize market for symbol '%s'", symbol, exc_info=True)
            return None

        self._markets[symbol] = market

        if self._is_running.is_set() and not self._is_stopping.is_set():
            self._start_market_worker_for_symbol(symbol=symbol, market=market)
            self._publish_bootstrap_market_messages_for_symbol(symbol=symbol, market=market)

        return market

    def handoff_market_data_routing_to_runner(self) -> bool:
        """
        Stop the internal router so an external runner can consume and route
        market-data messages deterministically.
        """
        router_thread = self._router_thread
        if router_thread is None:
            return False

        if not router_thread.is_alive():
            self._router_thread = None
            return False

        self._router_stop_event.set()
        join_timeout = self._config.thread_join_timeout_sec
        router_thread.join(timeout=join_timeout)
        if router_thread.is_alive():
            self._logger.warning(
                "Market-data router thread did not stop within %.2f sec during runner handoff",
                join_timeout,
            )
            return False

        self._router_thread = None
        return True

    def set_view_symbol(self, symbol: str) -> bool:
        normalized = self._normalize_required_symbol(symbol)
        if normalized is None:
            return False
        if normalized == self._viewer_stream_symbol:
            return True

        market = self._ensure_market(normalized)
        if market is None:
            return False

        previous_symbol = self._viewer_stream_symbol
        self._viewer_stream_symbol = normalized

        if self._is_running.is_set() and not self._is_stopping.is_set():
            if previous_symbol and previous_symbol != normalized and previous_symbol not in self._core_market_symbols:
                self._queue_market_stream_subscription_update(action="unsubscribe", symbol=previous_symbol)
            if normalized not in self._core_market_symbols:
                self._queue_market_stream_subscription_update(action="subscribe", symbol=normalized)

        # Prime from REST so the viewer can render quickly before websocket
        # deltas arrive for the newly selected symbol.
        try:
            self._prime_symbol_market_data_from_rest(normalized)
        except Exception:
            self._logger.warning("Failed to prime selected view symbol '%s' from REST", normalized, exc_info=True)

        return True

    def _start_open_interest_polling(self):
        if not self._config.enable_open_interest_polling:
            return

        self._open_interest_poll_thread = threading.Thread(
            target=self._open_interest_poll_loop,
            name=f"{type(self).__name__}OpenInterestThread",
            daemon=True,
        )
        self._open_interest_poll_thread.start()

    def _open_interest_poll_loop(self):
        while not self._open_interest_poll_stop_event.is_set():
            for symbol in tuple(self._markets.keys()):
                if self._open_interest_poll_stop_event.is_set():
                    return
                try:
                    payload = self._client.get_open_interest(symbol)
                    if (
                        isinstance(payload, dict)
                        and payload.get("openInterestValue") is None
                    ):
                        market = self._markets.get(symbol)
                        if market is not None and market.mark_price is not None:
                            open_interest_raw = payload.get("openInterest")
                            if open_interest_raw is not None:
                                open_interest = convert_to_decimal(open_interest_raw)
                                open_interest_value = (
                                    open_interest
                                    * market.mark_price
                                    * self._config.open_interest_value_multiplier
                                )
                                payload = dict(payload)
                                payload["openInterestValue"] = str(open_interest_value)

                    message = self._client.normalize_market_data_message(
                        payload,
                        provider=self._config.provider,
                        recv_ts_ms=int(time.time() * 1000),
                    )
                    if message is None or message.type != MarketDataType.OPEN_INTEREST:
                        continue
                    self.publish_market_data_message(message)
                except Exception:
                    self._logger.warning(
                        "Failed to poll open interest for symbol '%s'",
                        symbol,
                        exc_info=True,
                    )
            if self._open_interest_poll_stop_event.wait(self._config.open_interest_poll_interval_sec):
                return

    def _start_user_data_services(self):
        if not self._config.enable_user_data_stream:
            self._logger.info("User-data stream disabled by configuration")
            return

        try:
            listen_key_resp = self._client.create_listen_key()
            self._listen_key = self._resolve_listen_key(listen_key_resp)
        except Exception:
            self._logger.exception("Failed to create Aster listen key")
            self._listen_key = None
            return

        if self._listen_key is None:
            self._logger.error("Could not resolve listen key from response")
            return

        self._user_router_thread = threading.Thread(
            target=self._user_data_router_loop,
            name=f"{type(self).__name__}UserRouterThread",
            daemon=True,
        )
        self._user_router_thread.start()

        self._user_ws_thread = self._client.start_user_data_stream(
            listen_key=self._listen_key,
            out_queue=self._user_data_queue,
            stop_event=self._user_ws_stop_event,
            thread_name=f"{type(self).__name__}UserWsThread",
        )

        self._listen_key_keepalive_thread = threading.Thread(
            target=self._listen_key_keepalive_loop,
            name=f"{type(self).__name__}ListenKeyKeepaliveThread",
            daemon=True,
        )
        self._listen_key_keepalive_thread.start()

    def _stop_user_data_services(self):
        self._user_ws_stop_event.set()
        self._user_router_stop_event.set()
        self._listen_key_keepalive_stop_event.set()

        join_timeout = self._config.thread_join_timeout_sec

        if self._user_ws_thread and self._user_ws_thread.is_alive():
            self._user_ws_thread.join(timeout=join_timeout)
            if self._user_ws_thread.is_alive():
                self._logger.warning("User-data websocket thread did not stop within %.2f sec", join_timeout)
        self._user_ws_thread = None

        if self._listen_key_keepalive_thread and self._listen_key_keepalive_thread.is_alive():
            self._listen_key_keepalive_thread.join(timeout=join_timeout)
            if self._listen_key_keepalive_thread.is_alive():
                self._logger.warning("Listen-key keepalive thread did not stop within %.2f sec", join_timeout)
        self._listen_key_keepalive_thread = None

        if self._user_router_thread and self._user_router_thread.is_alive():
            self._user_router_thread.join(timeout=join_timeout)
            if self._user_router_thread.is_alive():
                self._logger.warning("User-data router thread did not stop within %.2f sec", join_timeout)
        self._user_router_thread = None

        if self._listen_key is not None:
            try:
                self._client.close_listen_key()
            except Exception:
                self._logger.warning("Failed to close listen key cleanly", exc_info=True)
            self._listen_key = None

    @staticmethod
    def _resolve_listen_key(value: Any) -> str | None:
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            listen_key = value.get("listenKey")
            if isinstance(listen_key, str) and listen_key.strip():
                return listen_key.strip()
        return None

    @staticmethod
    def _map_order_side(value: Any) -> Order.Side:
        if isinstance(value, str):
            v = value.strip().upper()
            if v == "BUY":
                return Order.Side.BUY
            if v == "SELL":
                return Order.Side.SELL
        raise ValueError(f"Unsupported order side: {value}")

    @staticmethod
    def _map_order_type(value: Any) -> Order.Type:
        mapping = {
            "MARKET": Order.Type.MARKET,
            "LIMIT": Order.Type.LIMIT,
            "STOP": Order.Type.STOP,
            "STOP_MARKET": Order.Type.STOP_MARKET,
            "TAKE_PROFIT": Order.Type.TAKE_PROFIT,
            "TAKE_PROFIT_MARKET": Order.Type.TAKE_PROFIT_MARKET,
            "TRAILING_STOP_MARKET": Order.Type.TRAILING_STOP_MARKET,
        }
        if isinstance(value, str):
            return mapping.get(value.strip().upper(), Order.Type.LIMIT)
        return Order.Type.LIMIT

    @staticmethod
    def _map_order_status(value: Any) -> Order.Status:
        mapping = {
            "NEW": Order.Status.NEW,
            "FILLED": Order.Status.FILLED,
            "PARTIALLY_FILLED": Order.Status.PARTIALLY_FILLED,
            "CANCELED": Order.Status.CANCELED,
            "REJECTED": Order.Status.REJECTED,
            "EXPIRED": Order.Status.EXPIRED,
        }
        if isinstance(value, str):
            return mapping.get(value.strip().upper(), Order.Status.NEW)
        return Order.Status.NEW

    @staticmethod
    def _map_time_in_force(value: Any) -> Order.TimeInForce:
        mapping = {
            "GTC": Order.TimeInForce.GTC,
            "IOC": Order.TimeInForce.IOC,
            "FOK": Order.TimeInForce.FOK,
            "GTX": Order.TimeInForce.GTX,
            "RPI": Order.TimeInForce.RPI,
        }
        if isinstance(value, str):
            return mapping.get(value.strip().upper(), Order.TimeInForce.GTC)
        return Order.TimeInForce.GTC

    @staticmethod
    def _map_position_side(value: Any) -> tuple[Position.Mode, Position.Side | None]:
        if not isinstance(value, str):
            return Position.Mode.NET, None
        v = value.strip().upper()
        if v == "LONG":
            return Position.Mode.HEDGE, Position.Side.LONG
        if v == "SHORT":
            return Position.Mode.HEDGE, Position.Side.SHORT
        return Position.Mode.NET, None

    def _get_critical_market_data_types(self) -> set[MarketDataType]:
        if not self._config.preserve_critical_messages_on_backpressure:
            return set()
        return {
            MarketDataType.INSTRUMENT,
            MarketDataType.MARKET_STATUS,
            MarketDataType.LIQUIDATION,
            MarketDataType.MARK_FUNDING,
        }

    def _listen_key_keepalive_loop(self):
        while not self._listen_key_keepalive_stop_event.wait(self._config.listen_key_keepalive_interval_sec):
            if self._listen_key_keepalive_stop_event.is_set():
                return
            try:
                self._client.keepalive_listen_key()
            except Exception:
                self._logger.warning("Failed to keepalive listen key", exc_info=True)

    def _user_data_router_loop(self):
        while not self._user_router_stop_event.is_set():
            try:
                event = self._user_data_queue.get(timeout=self._config.user_router_poll_timeout_sec)
            except queue.Empty:
                continue

            try:
                self._handle_user_data_event(event)
            except Exception:
                self._logger.exception("Failed to process user-data event")
            finally:
                self._user_data_queue.task_done()

    def _handle_user_data_event(self, event: dict[str, Any]):
        event_type = event.get("type")
        payload = event.get("payload", {})
        event_ts_ms = int(event.get("event_ts_ms", int(time.time() * 1000)))

        if event_type == "account_update":
            self._apply_account_update(payload)
            return

        if event_type == "order_trade_update":
            self._upsert_order_from_exchange_payload(payload, event_ts_ms=event_ts_ms)
            return

        if event_type == "listen_key_expired":
            self._logger.warning("Aster listen key expired event received")
            return

        if event_type == "margin_call":
            self._logger.warning("Aster margin call event received")
            return

    def _apply_account_update(self, payload: dict[str, Any]):
        if self._account is None:
            return

        if not isinstance(payload, dict):
            return

        balances = payload.get("B", [])
        if isinstance(balances, list):
            for b in balances:
                if not isinstance(b, dict):
                    continue
                asset = b.get("a")
                if not isinstance(asset, str) or not asset.strip():
                    continue
                asset = asset.strip().upper()

                wallet_balance = convert_to_decimal(b.get("wb", "0"))
                cross_wallet_balance = convert_to_decimal(b.get("cw", b.get("wb", "0")))

                wallet = self._account.get_wallet(asset)
                if wallet is None:
                    wallet = AsterAccount.Wallet(
                        asset=asset,
                        balance=wallet_balance,
                        unrealized_pnl=Decimal("0"),
                        equity=wallet_balance,
                        initial_margin_requirement=Decimal("0"),
                        maintenance_margin_requirement=Decimal("0"),
                        position_initial_margin_requirement=Decimal("0"),
                        open_order_initial_margin_requirement=Decimal("0"),
                        cross_margin_balance=cross_wallet_balance,
                        cross_margin_unrealized_pnl=Decimal("0"),
                        available_margin_balance=cross_wallet_balance,
                        max_withdraw_amount=cross_wallet_balance,
                        is_marginable=True,
                        update_timestamp=int(time.time() * 1000),
                    )
                else:
                    wallet.balance = wallet_balance
                    wallet.cross_margin_balance = cross_wallet_balance
                    wallet.equity = wallet_balance + wallet.unrealized_pnl
                    wallet.available_margin_balance = cross_wallet_balance
                    wallet.max_withdraw_amount = cross_wallet_balance
                    wallet.update_timestamp = int(time.time() * 1000)
                self._account.add_wallet(wallet)

                if asset == self._account.margin_asset:
                    self._account.balance = wallet_balance
                    self._account.cross_margin_balance = cross_wallet_balance
                    self._account.available_margin_balance = cross_wallet_balance

        positions = payload.get("P", [])
        if isinstance(positions, list):
            for p in positions:
                if not isinstance(p, dict):
                    continue

                symbol = p.get("s")
                if not isinstance(symbol, str) or not symbol.strip():
                    continue
                symbol = symbol.strip().upper()

                mode, side = self._map_position_side(p.get("ps"))
                quantity = convert_to_decimal(p.get("pa", "0"))
                entry_price = convert_to_decimal(p.get("ep", "0"))

                position = self._find_position(symbol, mode=mode, side=side)
                if quantity == 0:
                    if position is not None:
                        self._account.delete_position(position)
                    continue

                if position is None:
                    position = Position(symbol=symbol, mode=mode, side=side)
                    self._account.add_position(position)
                    position = self._find_position(symbol, mode=mode, side=side)

                if position is not None:
                    position.quantity = quantity
                    position.entry_price = entry_price
                    position.side = side

    def _find_position(self, symbol: str, *, mode: Position.Mode, side: Position.Side | None) -> Position | None:
        for position in self._account._get_positions(symbol):
            if position.mode != mode:
                continue
            if mode == Position.Mode.HEDGE and position.side != side:
                continue
            return position
        return None

    def _upsert_order_from_exchange_payload(self, payload: dict[str, Any], *, event_ts_ms: int) -> Order | None:
        if self._account is None:
            return None
        if not isinstance(payload, dict):
            return None

        symbol = payload.get("s", payload.get("symbol"))
        if not isinstance(symbol, str) or not symbol.strip():
            return None
        symbol = symbol.strip().upper()

        order_id_value = payload.get("i", payload.get("orderId"))
        if order_id_value is None:
            return None
        order_id = str(order_id_value)

        orders = self._account._get_orders(symbol)
        existing = next((o for o in orders if o.id == order_id), None)

        client_id = payload.get("c", payload.get("clientOrderId"))
        client_id = client_id if isinstance(client_id, str) and client_id.strip() else None

        try:
            side = self._map_order_side(payload.get("S", payload.get("side")))
        except ValueError:
            side = existing.side if existing is not None else Order.Side.BUY
        order_type = self._map_order_type(payload.get("o", payload.get("type")))
        time_in_force = self._map_time_in_force(payload.get("f", payload.get("timeInForce")))
        status = self._map_order_status(payload.get("X", payload.get("status")))

        quantity = convert_to_decimal(payload.get("q", payload.get("origQty", "0")))
        price = convert_to_decimal(payload.get("p", payload.get("price", "0")))
        timestamp = int(payload.get("T", payload.get("updateTime", payload.get("time", event_ts_ms))))
        filled_quantity = convert_to_decimal(payload.get("z", payload.get("executedQty", "0")))

        avg_price_raw = payload.get("ap", payload.get("avgPrice"))
        average_fill_price = None
        if avg_price_raw is not None:
            avg_price = convert_to_decimal(avg_price_raw)
            if avg_price > 0:
                average_fill_price = avg_price

        if existing is None:
            order = Order(
                id=order_id,
                client_id=client_id,
                symbol=symbol,
                type=order_type,
                time_in_force=time_in_force,
                side=side,
                price=price,
                quantity=quantity,
                timestamp=timestamp,
                status=status,
                filled_quantity=filled_quantity,
                average_fill_price=average_fill_price,
            )
            self._account.add_order(order)
            return order

        existing.client_id = client_id
        existing.type = order_type
        existing.time_in_force = time_in_force
        existing.side = side
        existing.price = price
        existing.quantity = quantity
        existing.timestamp = timestamp
        existing.status = status
        existing.filled_quantity = filled_quantity
        existing.average_fill_price = average_fill_price
        return existing

    def _get_market(self, symbol: str) -> AsterMarket:
        if symbol is None:
            raise ValueError("'symbol' is required")

        normalized = symbol.strip().upper()
        market = self._markets.get(normalized)
        if market is None:
            raise ValueError(f"Unknown market symbol: {normalized}")
        return market

    def get_account(self, quote_currency: str) -> Account:
        if self._account is None:
            raise RuntimeError("Account data has not been initialized")
        return copy.deepcopy(self._account)

    def get_market_settings(self, symbol: str) -> Market.Settings | None:
        market = self._markets.get(symbol)
        return market.settings if market else None

    def get_recent_klines(
        self,
        *,
        symbol: str,
        interval: str,
        limit: int = 240,
    ) -> list[dict[str, Any]]:
        normalized_symbol = self._normalize_required_symbol(symbol)
        if normalized_symbol is None:
            raise ValueError("'symbol' is required")
        if not isinstance(interval, str) or not interval.strip():
            raise ValueError("'interval' is required")
        if limit < 1:
            raise ValueError("'limit' must be >= 1")

        normalized_interval = interval.strip().lower()
        raw_rows = self._client.get_klines(
            symbol=normalized_symbol,
            interval=normalized_interval,
            limit=limit,
        )

        if not isinstance(raw_rows, list):
            return []

        parsed_rows: list[dict[str, Any]] = []
        for row in raw_rows:
            try:
                if isinstance(row, (list, tuple)) and len(row) >= 5:
                    open_time_ms = self._safe_int(row[0])
                    close_time_ms = self._safe_int(row[6]) if len(row) > 6 else open_time_ms
                    open_price = convert_to_decimal(row[1])
                    high_price = convert_to_decimal(row[2])
                    low_price = convert_to_decimal(row[3])
                    close_price = convert_to_decimal(row[4])
                elif isinstance(row, dict):
                    open_time_ms = self._safe_int(row.get("open_time_ms", row.get("openTime", row.get("t"))))
                    close_time_ms = self._safe_int(
                        row.get("close_time_ms", row.get("closeTime", row.get("T")))
                    )
                    open_price = convert_to_decimal(row.get("open", row.get("o")))
                    high_price = convert_to_decimal(row.get("high", row.get("h")))
                    low_price = convert_to_decimal(row.get("low", row.get("l")))
                    close_price = convert_to_decimal(row.get("close", row.get("c")))
                else:
                    continue
            except (TypeError, ValueError):
                continue

            if open_time_ms is None:
                continue
            if close_time_ms is None or close_time_ms < open_time_ms:
                close_time_ms = open_time_ms

            parsed_rows.append(
                {
                    "open_time_ms": open_time_ms,
                    "close_time_ms": close_time_ms,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                }
            )

        parsed_rows.sort(key=lambda item: (item["open_time_ms"], item["close_time_ms"]))
        return parsed_rows

    def create_market_order(
        self, symbol: str, side: Order.Side, quantity: Decimal, position_side: Position.Side | None = None
    ) -> Order:
        market = self._get_market(symbol)
        rounded_quantity = market.round_quantity(quantity)
        if rounded_quantity <= 0:
            raise ValueError("Order quantity must be positive after rounding")

        response = self._client.create_order(
            symbol=market.settings.symbol,
            side=side.name,
            type="MARKET",
            quantity=str(rounded_quantity),
            newOrderRespType="RESULT",
            position_side=position_side.name if position_side else None,
        )
        order = self._upsert_order_from_exchange_payload(response, event_ts_ms=int(time.time() * 1000))
        if order is None:
            raise RuntimeError("Failed to parse market-order response from exchange")
        return copy.deepcopy(order)

    def create_limit_order(
        self, symbol: str, side: Order.Side, price: Decimal, quantity: Decimal,
        position_side: Position.Side | None = None
    ) -> Order:
        market = self._get_market(symbol)
        rounded_price = market.round_price(price)
        rounded_quantity = market.round_quantity(quantity)
        if rounded_price <= 0:
            raise ValueError("Order price must be positive after rounding")
        if rounded_quantity <= 0:
            raise ValueError("Order quantity must be positive after rounding")

        response = self._client.create_order(
            symbol=market.settings.symbol,
            side=side.name,
            type="LIMIT",
            timeInForce="GTC",
            price=str(rounded_price),
            quantity=str(rounded_quantity),
            newOrderRespType="RESULT",
            position_side=position_side.name if position_side else None,
        )
        order = self._upsert_order_from_exchange_payload(response, event_ts_ms=int(time.time() * 1000))
        if order is None:
            raise RuntimeError("Failed to parse limit-order response from exchange")
        return copy.deepcopy(order)

    def cancel_order(self, order: Order) -> Order:
        if order is None:
            raise ValueError("'order' is required")

        symbol = order.symbol.strip().upper()
        order_id: int | None = None
        if order.id and str(order.id).strip().isdigit():
            order_id = int(str(order.id).strip())

        response = self._client.cancel_order(
            symbol=symbol,
            order_id=order_id,
            orig_client_order_id=order.client_id if order_id is None else None,
        )
        updated = self._upsert_order_from_exchange_payload(response, event_ts_ms=int(time.time() * 1000))
        if updated is None:
            order.status = Order.Status.CANCELED
            return copy.deepcopy(order)
        return copy.deepcopy(updated)

    class Configuration(Exchange.Configuration):
        provider: Literal["aster"]

        # endpoints
        rest_url: str
        ws_url: str

        # properties we know the client will likely need
        recv_window_ms: int = Field(5000, ge=1)
        request_timeout_sec: float = Field(10.0, gt=0)
        ws_ping_interval_sec: float = Field(15.0, gt=0)
        ws_reconnect_delay_sec: float = Field(3.0, gt=0)
        subscribe_book_delta_streams: bool = False
        depth_stream_speed: str = "100ms"
        router_poll_timeout_sec: float = Field(0.25, gt=0)
        thread_join_timeout_sec: float = Field(5.0, gt=0)
        market_worker_poll_timeout_sec: float = Field(0.25, gt=0)
        market_worker_join_timeout_sec: float = Field(5.0, gt=0)
        exchange_market_data_queue_maxsize: int = Field(20000, ge=0)
        market_data_queue_maxsize: int = Field(5000, ge=0)
        drop_oldest_on_full: bool = True
        preserve_critical_messages_on_backpressure: bool = True
        enable_open_interest_polling: bool = True
        open_interest_poll_interval_sec: float = Field(5.0, gt=0)
        wait_for_market_data_on_startup: bool = True
        market_data_init_timeout_sec: float = Field(30.0, gt=0)
        market_data_init_poll_interval_sec: float = Field(0.05, gt=0)
        open_interest_value_multiplier: Decimal = Field(Decimal("2"), gt=0)
        enable_user_data_stream: bool = True
        user_data_queue_maxsize: int = Field(10000, ge=0)
        user_router_poll_timeout_sec: float = Field(0.25, gt=0)
        listen_key_keepalive_interval_sec: float = Field(1800.0, gt=0)

        @field_validator("open_interest_value_multiplier", mode="before")
        @classmethod
        def _transform_open_interest_value_multiplier(cls, value: Any) -> Decimal:
            return convert_to_decimal(value)

        # optional environment selection
        use_testnet: bool = False

    class Credentials(Exchange.Credentials):
        model_config = SettingsConfigDict(env_prefix="aster_")

        api_key: SecretStr
        api_secret: SecretStr

    class EnvOverrides(Exchange.EnvOverrides):
        model_config = SettingsConfigDict(env_prefix="aster_override_")

        use_testnet: bool | None = None
