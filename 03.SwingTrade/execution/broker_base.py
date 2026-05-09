# execution/broker_base.py

class BrokerBase:
    """
    Abstract broker interface.
    """

    def place_amo_buy(self, symbol, qty):
        raise NotImplementedError

    def place_target_sell(self, symbol, qty, price):
        raise NotImplementedError

    def cancel_all_open_orders(self):
        raise NotImplementedError
