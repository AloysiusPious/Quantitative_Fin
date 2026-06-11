"""
Zerodha Historical Data Downloader
====================================
Downloads data into:  data/START_END/  (e.g. data/2021-01-01_2024-01-01/)
Always downloads:     Nifty 50 index (token 256265) + India VIX (token 264969)

WARMUP PERIOD:
  Data download starts from (start_date - warmup_days) so that 20 EMA,
  50 EMA, ATR, and volume average are fully accurate from the first
  trading day. Warmup candles are downloaded but the backtest engine
  ignores them for signal generation.

USAGE:
    python zerodha_downloader.py                    # all stocks + indices
    python zerodha_downloader.py --symbol RELIANCE  # single stock
    python zerodha_downloader.py --force            # re-download existing
"""

import requests
import pandas as pd
import json
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta


def load_config(path="config_ltcg.json") -> dict:
    with open(path) as f:
        return json.load(f)

def load_watchlist(cfg: dict) -> list:
    with open(cfg["stock_selection"]["watchlist_file"]) as f:
        return json.load(f)["stocks"]

def get_headers(enctoken: str) -> dict:
    return {
        "Authorization": f"enctoken {enctoken}",
        "User-Agent": "Mozilla/5.0",
    }

def get_data_dir(cfg: dict) -> Path:
    """Returns data/START_END/ and creates it."""
    start = cfg["backtest"]["start_date"]
    end   = cfg["backtest"]["end_date"]
    d     = Path(cfg["output"]["data_dir"]) / f"{start}_{end}"
    d.mkdir(parents=True, exist_ok=True)
    return d

def get_warmup_start(cfg: dict) -> str:
    """
    Actual download start = start_date - warmup_days.
    e.g. start=2021-01-01, warmup=100 → download from 2020-09-23
    Gives ~70 trading candles before first signal day so all indicators
    (20 EMA, 50 EMA, ATR-14, Vol-20) are accurate from day 1.
    """
    start  = datetime.strptime(cfg["backtest"]["start_date"], "%Y-%m-%d")
    warmup = cfg["backtest"].get("warmup_days", 100)
    return (start - timedelta(days=warmup)).strftime("%Y-%m-%d")

def get_instrument_token(symbol: str, headers: dict, exchange: str = "NSE") -> int | None:
    try:
        resp  = requests.get("https://api.kite.trade/instruments", headers=headers, timeout=20)
        df    = pd.read_csv(pd.io.common.StringIO(resp.text))
        match = df[(df["tradingsymbol"] == symbol) & (df["exchange"] == exchange)]
        if not match.empty:
            return int(match.iloc[0]["instrument_token"])
        print(f"    ⚠ Token not found for {symbol} on {exchange}")
    except Exception as e:
        print(f"    ⚠ Instrument lookup error: {e}")
    return None

def fetch_historical(token: int, from_date: str, to_date: str,
                     interval: str, headers: dict, delay: float) -> pd.DataFrame:
    """
    Fetch OHLCV from Zerodha historical API.
    Automatically chunks into 1800-day windows to stay under the 2000-day API limit.
    Multiple chunks are merged and deduplicated.
    """
    from datetime import datetime, timedelta

    start_dt = datetime.strptime(from_date, "%Y-%m-%d")
    end_dt   = datetime.strptime(to_date,   "%Y-%m-%d")
    CHUNK    = 1800  # safe margin below 2000-day limit

    all_candles = []
    chunk_start = start_dt

    while chunk_start < end_dt:
        chunk_end = min(chunk_start + timedelta(days=CHUNK), end_dt)

        url    = f"https://kite.zerodha.com/oms/instruments/historical/{token}/{interval}"
        params = {
            "from": chunk_start.strftime("%Y-%m-%d"),
            "to":   chunk_end.strftime("%Y-%m-%d"),
            "oi":   1
        }
        try:
            time.sleep(delay)
            resp = requests.get(url, headers=headers, params=params, timeout=20)
            data = resp.json()
            if data.get("status") == "success":
                candles = data["data"]["candles"]
                all_candles.extend(candles)
            else:
                msg = data.get("message", "Unknown")
                print(f"\n    ⚠ API [{chunk_start.date()}→{chunk_end.date()}]: {msg}")
        except Exception as e:
            print(f"\n    ⚠ Fetch error [{chunk_start.date()}]: {e}")

        chunk_start = chunk_end + timedelta(days=1)

    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame(all_candles, columns=["date","open","high","low","close","volume","oi"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def download_index(name: str, token: int, cfg: dict,
                   from_date: str, to_date: str, force: bool = False) -> bool:
    """Download a fixed-token index using hardcoded token — no instrument lookup."""
    data_dir = get_data_dir(cfg)
    interval = cfg["backtest"]["interval"]
    out_path = data_dir / f"{name}_{interval}.csv"

    print(f"  ↓ {name:22s}", end=" ", flush=True)
    if out_path.exists() and not force:
        print("already exists")
        return True

    headers = get_headers(cfg["zerodha"]["enctoken"])
    delay   = cfg["zerodha"]["request_delay_sec"]
    df      = fetch_historical(token, from_date, to_date, interval, headers, delay)

    if df.empty:
        print("SKIP (no data)")
        return False
    df.to_csv(out_path, index=False)
    print(f"→ {len(df)} candles ({from_date} to {to_date}) → {out_path.name}")
    return True


def download_symbol(symbol: str, cfg: dict, from_date: str, to_date: str,
                    force: bool = False) -> bool:
    data_dir = get_data_dir(cfg)
    interval = cfg["backtest"]["interval"]
    out_path = data_dir / f"{symbol}_{interval}.csv"

    if out_path.exists() and not force:
        print(f"  ✓ {symbol:22s} already exists")
        return True

    enctoken = cfg["zerodha"]["enctoken"]
    if enctoken == "PASTE_YOUR_ENCTOKEN_HERE":
        print("❌ enctoken not set — run get_enctoken.py first")
        return False

    headers = get_headers(enctoken)
    delay   = cfg["zerodha"]["request_delay_sec"]

    print(f"  ↓ {symbol:22s}", end=" ", flush=True)
    token = get_instrument_token(symbol, headers)
    if not token:
        print("SKIP (no token)")
        return False

    df = fetch_historical(token, from_date, to_date, interval, headers, delay)
    if df.empty:
        print("SKIP (no data)")
        return False

    df.to_csv(out_path, index=False)
    print(f"→ {len(df)} candles ({from_date} to {to_date}) → {out_path.name}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, help="Download single symbol e.g. RELIANCE")
    parser.add_argument("--force",  action="store_true", help="Re-download even if file exists")
    parser.add_argument("--config", default="config_ltcg.json")
    args = parser.parse_args()

    cfg       = load_config(args.config)
    watchlist = load_watchlist(cfg)
    enctoken  = cfg["zerodha"]["enctoken"]
    data_dir  = get_data_dir(cfg)

    if enctoken == "PASTE_YOUR_ENCTOKEN_HERE":
        print("\n❌ enctoken not set — run: python get_enctoken.py\n")
        return

    # ── Date range: warmup start → backtest end ──
    warmup_start = get_warmup_start(cfg)
    trade_start  = cfg["backtest"]["start_date"]
    end_date     = cfg["backtest"]["end_date"]
    warmup_days  = cfg["backtest"].get("warmup_days", 100)

    print(f"\n📥 Zerodha Data Downloader")
    print(f"   Backtest period : {trade_start} → {end_date}")
    print(f"   Download from   : {warmup_start} (incl. {warmup_days}-day EMA warmup)")
    print(f"   Interval        : {cfg['backtest']['interval']}")
    print(f"   Data dir        : {data_dir}/")
    print(f"   Force           : {'Yes' if args.force else 'No'}\n")
    print(f"   ℹ  Warmup candles ({warmup_start} → {trade_start}) are downloaded")
    print(f"      so that 20 EMA, 50 EMA, ATR and volume avg are fully")
    print(f"      accurate from the very first signal on {trade_start}.\n")

    # ── Always download benchmark indices first ──
    print("── Benchmark Indices (with warmup) ──")
    download_index("NIFTY50",  256265, cfg, warmup_start, end_date, args.force)
    download_index("INDIAVIX", 264969, cfg, warmup_start, end_date, args.force)
    print()

    # ── Stocks ──
    if args.symbol:
        print("── Stock (with warmup) ──")
        download_symbol(args.symbol.upper(), cfg, warmup_start, end_date, args.force)
    else:
        blacklist = set(cfg["stock_selection"].get("blacklist", []))
        active    = [s for s in watchlist if s["symbol"] not in blacklist]
        skipped   = [s["symbol"] for s in watchlist if s["symbol"] in blacklist]

        print(f"── Nifty F&O Stocks ({len(active)} active, {len(skipped)} blacklisted) ──")
        if skipped:
            print(f"   Skipping blacklisted: {', '.join(skipped)}\n")

        ok = sum(download_symbol(s["symbol"], cfg, warmup_start, end_date, args.force)
                 for s in active)
        print(f"\n✅ {ok}/{len(active)} stocks + 2 indices downloaded → {data_dir}/")
        print(f"   Each file contains {warmup_days} days of warmup data before {trade_start}")


if __name__ == "__main__":
    main()
