class BrokerBase:
    def place_buy(self, symbol, qty):
        raise NotImplementedError

    def place_sell(self, symbol, qty):
        raise NotImplementedError

    def fetch_positions(self):
        raise NotImplementedError
