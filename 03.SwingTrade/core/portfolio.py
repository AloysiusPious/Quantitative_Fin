# core/portfolio.py

class Portfolio:
    def __init__(self, initial_capital, max_positions):
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.position_size = initial_capital / max_positions
        self.active_symbols = set()

    def can_enter(self):
        return len(self.active_symbols) < self.max_positions

    def enter_trade(self, symbol):
        self.active_symbols.add(symbol)

    def exit_trade(self, symbol):
        self.active_symbols.discard(symbol)
