import requests
import time
from Zerodha.broker_base import BrokerBase

class EnctokenBroker(BrokerBase):
    def __init__(self, config):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"enctoken {config.enctoken}"
        })

        if not self._is_alive():
            raise RuntimeError("ENCTOKEN INVALID / EXPIRED")

    def _is_alive(self):
        r = self.session.get(
            "https://kite.zerodha.com/oms/user/profile"
        )
        return r.status_code == 200

    def place_buy(self, symbol, qty):
        payload = {
            "exchange": "NSE",
            "tradingsymbol": symbol,
            "transaction_type": "BUY",
            "quantity": qty,
            "order_type": "MARKET",
            "product": "CNC"
        }

        r = self.session.post(
            "https://kite.zerodha.com/oms/orders/regular",
            data=payload
        )

        if r.status_code != 200:
            raise RuntimeError("ORDER FAILED")

        order_id = r.json()["data"]["order_id"]
        time.sleep(1)
        return order_id

    def place_sell(self, symbol, qty):
        # Same as buy with SELL
        pass

    def fetch_positions(self):
        r = self.session.get(
            "https://kite.zerodha.com/oms/portfolio/positions"
        )
        return r.json()["data"]["net"]
