# execution/kite_api_broker.py

from .broker_base import BrokerBase

try:
    from kiteconnect import KiteConnect
except ImportError:
    KiteConnect = None


class KiteApiBroker(BrokerBase):

    def __init__(self, api_key, access_token):
        if KiteConnect is None:
            raise ImportError("kiteconnect package not installed")

        self.kite = KiteConnect(api_key=api_key)
        self.kite.set_access_token(access_token)

    def place_amo_buy(self, symbol, qty):
        print(f"[API AMO BUY] {symbol} | Qty={qty}")

        return self.kite.place_order(
            variety="amo",
            exchange="NSE",
            tradingsymbol=symbol,
            transaction_type="BUY",
            quantity=qty,
            order_type="MARKET",
            product="CNC"
        )

    def place_target_sell(self, symbol, qty, price):
        print(f"[API TARGET SELL] {symbol} | Qty={qty} | Price={price}")

        return self.kite.place_order(
            variety="regular",
            exchange="NSE",
            tradingsymbol=symbol,
            transaction_type="SELL",
            quantity=qty,
            order_type="LIMIT",
            price=price,
            product="CNC"
        )

    def cancel_all_open_orders(self):
        print("[API] Cancel all open orders")

        orders = self.kite.orders()
        for o in orders:
            if o["status"] in ("OPEN", "TRIGGER PENDING"):
                self.kite.cancel_order(
                    variety=o["variety"],
                    order_id=o["order_id"]
                )
