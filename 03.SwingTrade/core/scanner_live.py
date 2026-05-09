# core/scanner_live.py

import requests
from .signal_engine import generate_signal


def fetch_close_price(symbol, enctoken):
    url = f"https://kite.zerodha.com/oms/quote/ltp?i=NSE:{symbol}"
    headers = {"Authorization": f"enctoken {enctoken}"}

    r = requests.get(url, headers=headers, timeout=10)
    r.raise_for_status()

    return r.json()["data"][f"NSE:{symbol}"]["last_price"]


def scan_live(config):
    """
    Live EOD scan.
    Assumes indicators are precomputed offline.
    """

    signals = []
    symbols = open("symbols.txt").read().splitlines()

    for symbol in symbols:
        try:
            close = fetch_close_price(symbol, config.enctoken)
        except Exception as e:
            print(f"[LIVE SCAN ERROR] {symbol}: {e}")
            continue

        # Minimal row (indicators should come from stored data in real use)
        row = {
            "Close": close,
            "EMA20": close * 1.01,
            "EMA200": close * 1.05,
            "Pct_Below_20EMA": -3
        }

        # VIX hardcoded placeholder for live
        vix_close = 18

        signal = generate_signal(row, vix_close)
        if signal:
            signals.append({
                "symbol": symbol,
                **signal
            })

    return signals
