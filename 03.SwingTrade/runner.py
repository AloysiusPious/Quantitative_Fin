
from utils.config_loader import load_config
from core.scanner import scan_signals
from core.portfolio import Portfolio
from core.entry_gate import entry_gate
from execution.execution_factory import get_broker

def main():
    config = load_config("config.cfg")
    portfolio = Portfolio(config)
    broker = get_broker(config)

    signals = scan_signals(config)

    for s in signals:
        ok, reason = entry_gate(portfolio)
        if not ok:
            print(f"[REJECTED] {s['symbol']} {reason}")
            continue

        qty = int(portfolio.position_size / s["entry_price"])
        broker.buy(s["symbol"], qty)
        broker.sell(s["symbol"], qty)
        portfolio.add_position(s)

if __name__ == "__main__":
    main()
