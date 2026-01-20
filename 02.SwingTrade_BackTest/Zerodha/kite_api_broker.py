from kiteconnect import KiteConnect

class KiteApiBroker(BrokerBase):
    def __init__(self, config):
        self.kite = KiteConnect(api_key=config.api_key)
        self.kite.set_access_token(config.access_token)

    def place_buy(self, symbol, qty):
        return self.kite.place_order(
            variety="regular",
            exchange="NSE",
            tradingsymbol=symbol,
            transaction_type="BUY",
            quantity=qty,
            order_type="MARKET",
            product="CNC"
        )

    def place_sell(self, symbol, qty):
        return self.kite.place_order(
            variety="regular",
            exchange="NSE",
            tradingsymbol=symbol,
            transaction_type="SELL",
            quantity=qty,
            order_type="MARKET",
            product="CNC"
        )

    def fetch_positions(self):
        return self.kite.positions()["net"]
