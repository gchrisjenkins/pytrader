import logging
from decimal import Decimal
from typing import Any

from pytrader import OrderSide, OrderType, OrderStatus
from pytrader.simulation import CorrelatedRandomWalkExchange


def main(config: dict[str, Any]):

    # Initialize the exchange with the loaded configuration
    exchange = CorrelatedRandomWalkExchange(config)

    # Set the maximum number of steps for the simulation
    max_steps = 1000
    step = 0

    while step < max_steps:

        # Advance the simulation by one step
        exchange.step()
        step += 1

        account = exchange.get_account('USDT')

        if step == 1:
            # Step 0: Display the current state of the exchange
            btc_ticker = exchange.get_market_snapshot('BTC/USDT')
            eth_ticker = exchange.get_market_snapshot('ETH/USDT')
            print(
                f"Step {step}: BTC price = {btc_ticker.last_price}, ETH price = {eth_ticker.last_price}, "
                f"Cash balance = {account.cash_balance}"
            )

        elif step == 2:
            btc_ticker = exchange.get_market_snapshot('BTC/USDT')
            eth_ticker = exchange.get_market_snapshot('ETH/USDT')
            print(
                f"Step {step}: BTC price = {btc_ticker.last_price}, ETH price = {eth_ticker.last_price}, "
                f"Cash balance = {account.cash_balance}"
            )
            # Step 1: Create market orders
            btc_buy_order = exchange.create_market_order('BTC/USDT', OrderSide.BUY, Decimal('1.0'))
            eth_buy_order = exchange.create_market_order('ETH/USDT', OrderSide.BUY, Decimal('10.0'))
            print(f"Market orders created: BTC buy @ {btc_buy_order.price}, ETH buy @ {eth_buy_order.price}")

        elif step == 3:
            btc_ticker = exchange.get_market_snapshot('BTC/USDT')
            eth_ticker = exchange.get_market_snapshot('ETH/USDT')
            print(
                f"Step {step}: BTC price = {btc_ticker.last_price}, ETH price = {eth_ticker.last_price}, "
                f"Cash balance = {account.cash_balance}"
            )
            # Step 2: Create limit orders around the current price
            btc_price = btc_ticker.last_price
            eth_price = eth_ticker.last_price
            btc_buy_limit = exchange.create_limit_order(
                'BTC/USDT', OrderSide.BUY, btc_price * Decimal('0.997'), Decimal('0.5')
            )
            btc_sell_limit = exchange.create_limit_order(
                'BTC/USDT', OrderSide.SELL, btc_price * Decimal('1.003'), Decimal('0.5')
            )
            eth_buy_limit = exchange.create_limit_order(
                'ETH/USDT', OrderSide.BUY, eth_price * Decimal('0.997'), Decimal('10.0')
            )
            eth_sell_limit = exchange.create_limit_order(
                'ETH/USDT', OrderSide.SELL, eth_price * Decimal('1.003'), Decimal('10.0')
            )
            print(f"Limit orders created for BTC and ETH around current prices.")

        # Check if any limit orders have been filled
        # account = exchange.get_account('USDT')
        current_orders = account.orders
        filled_limit_orders = [
            order for symbol_orders in account.orders.values() for order in symbol_orders
            if order.type == OrderType.LIMIT and order.status == OrderStatus.FILLED
        ]
        if filled_limit_orders:
            print(f"Limit order filled at step {step}:")
            for order in filled_limit_orders:
                print(f"  {order.symbol} {order.side} {order.quantity} @ {order.price}")
            break

        # Optional: Print price updates every 100 steps
        if step % 100 == 0 and step > 0:
            btc_ticker = exchange.get_market_snapshot('BTC/USDT')
            eth_ticker = exchange.get_market_snapshot('ETH/USDT')
            print(f"Step {step}: BTC price = {btc_ticker.last_price}, ETH price = {eth_ticker.last_price}")
            print(f"  Cash balance = {account.cash_balance}")
            print(f"  Unrealized P&L = {account.unrealized_pnl}")
            print(f"  Equity = {account.equity}")
            print(f"  Realized P&L = {account.realized_pnl}")
            print(f"  Available margin = {account.available_margin}")
            print(f"  Initial margin requirement = {account.initial_margin_requirement}")
            print(f"  Maintenance margin requirement = {account.maintenance_margin_requirement}")
            print(f"  Reserved margin requirement = {account.reserved_margin_requirement}")

    else:
        print("No limit orders filled within the maximum steps.")

    # Advance the simulation by one step
    exchange.step()
    step += 1

    account = exchange.get_account('USDT')

    # Display final account state
    print("Final account state:")
    btc_ticker = exchange.get_market_snapshot('BTC/USDT')
    eth_ticker = exchange.get_market_snapshot('ETH/USDT')
    print(f"Step {step}: BTC price = {btc_ticker.last_price}, ETH price = {eth_ticker.last_price}")
    print(f"  Cash balance = {account.cash_balance}")
    print(f"  Unrealized P&L = {account.unrealized_pnl}")
    print(f"  Equity = {account.equity}")
    print(f"  Realized P&L = {account.realized_pnl}")
    print(f"  Available margin = {account.available_margin}")
    print(f"  Initial margin requirement = {account.initial_margin_requirement}")
    print(f"  Maintenance margin requirement = {account.maintenance_margin_requirement}")
    print(f"  Reserved margin requirement = {account.reserved_margin_requirement}")


if __name__ == "__main__":

    import json

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(threadName)s | %(levelname)s | %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    with open("config.json", 'r') as f:
        config = json.load(f)

    main(config)
