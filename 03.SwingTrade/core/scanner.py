# core/scanner.py

from .scanner_backtest import scan_backtest
from .scanner_live import scan_live


def scan_signals(config):
    """
    Route scanning based on mode.
    live = False → CSV backtest scan
    live = True  → Live EOD scan
    """
    if config.live:
        return scan_live(config)
    return scan_backtest(config)
