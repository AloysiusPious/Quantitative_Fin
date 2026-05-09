# execution/paper_broker.py

from .broker_base import BrokerBase


class PaperBroker(BrokerBase):

    def place_amo_buy(self, symbol, qty):
        order_id = f"PAPER_AMO_BUY_{symbol}"
        print(f"[PAPER AMO BUY] {symbol} | Qty={qty}")
        return order_id

    def place_target_sell(self, symbol, qty, price):
        print(f"[PAPER TARGET SELL] {symbol} | Qty={qty} | Price={price}")

    def cancel_all_open_orders(self):
        print("[PAPER] Cancel all open orders")
