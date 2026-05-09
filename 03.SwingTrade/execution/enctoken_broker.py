# execution/enctoken_broker.py

import requests
from .broker_base import BrokerBase


class EnctokenBroker(BrokerBase):

    def __init__(self, enctoken):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"enctoken {enctoken}"
        })

    def place_amo_buy(self, symbol, qty):
        """
        Place AMO BUY order.
        """
        print(f"[AMO BUY] {symbol} | Qty={qty}")

        # NOTE:
        # Real Zerodha AMO API call goes here.
        # Kept as print for safety.

        return f"AMO_BUY_{symbol}"

    def place_target_sell(self, symbol, qty, price):
        """
        Place LIMIT TARGET SELL order.
        """
        print(
            f"[TARGET SELL] {symbol} | Qty={qty} | Price={price}"
        )

        # Real LIMIT sell API call goes here

    def cancel_all_open_orders(self):
        """
        Cancel all pending / unexecuted orders.
        """
        print("[LIVE] Cancel all open orders")

        # Real cancel API call goes here
