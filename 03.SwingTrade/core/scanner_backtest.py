import os
import pandas as pd
from .signal_engine import compute_indicators


def scan_backtest(config):
    """
    Backtest scanner with indicator warm-up handling.

    - Loads extra historical data before from_date
    - Computes indicators on full data
    - Returns only rows within [from_date, to_date]
    """

    scanned_data = []

    # ---------- Load symbols ----------
    def load_symbols(path="symbols.txt"):
        with open(path) as f:
            return [
                line.strip().upper()
                for line in f
                if line.strip() and not line.lstrip().startswith("#")
            ]

    symbols = load_symbols()
    exclude_files = {"NIFTY_50", "VIX", "STOCK_DATE_REF"}

    # ---------- Warm-up window ----------
    WARMUP_DAYS = 300  # safe for EMA200

    warmup_start_date = config.from_date - pd.Timedelta(days=WARMUP_DAYS)

    for symbol in symbols:
        if symbol in exclude_files:
            continue

        csv_file = f"{symbol}.csv"
        csv_path = os.path.join(config.csv_dir, csv_file)

        if not os.path.exists(csv_path):
            print(f"[MISSING CSV] {csv_file}")
            continue

        print(f"Processing {csv_file}")

        df = pd.read_csv(csv_path)

        if "Date" not in df.columns:
            continue

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        # ---------- LOAD EXTRA HISTORY ----------
        df = df[
            (df["Date"] >= warmup_start_date) &
            (df["Date"] <= config.to_date)
        ]

        if df.empty:
            continue

        # ---------- COMPUTE INDICATORS ON FULL DATA ----------
        df = compute_indicators(df,symbol )

        # ---------- FINAL BACKTEST WINDOW ----------
        df = df[
            (df["Date"] >= config.from_date) &
            (df["Date"] <= config.to_date)
        ].reset_index(drop=True)

        if df.empty:
            continue

        scanned_data.append({
            "symbol": symbol,
            "df": df
        })

    return scanned_data
