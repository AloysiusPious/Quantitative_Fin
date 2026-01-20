class PaperBroker(BrokerBase):
    def place_buy(self, symbol, qty):
        print(f"[PAPER BUY] {symbol} {qty}")
        return f"PAPER_{symbol}"

    def place_sell(self, symbol, qty):
        print(f"[PAPER SELL] {symbol} {qty}")
        return f"PAPER_EXIT_{symbol}"

    def fetch_positions(self):
        return []
