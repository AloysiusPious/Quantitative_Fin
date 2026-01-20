# core/signal_engine.py

import pandas as pd
import talib
import os


# -------------------------------------------------
# INDICATORS
# -------------------------------------------------
def compute_indicators(df: pd.DataFrame, symbol) -> pd.DataFrame:
    """
    Computes indicators + calendar-based filters
    exactly matching your strategy logic.
    """

    df = df.copy()

    # ---------------- ENSURE DATE ----------------
    if "Date" not in df.columns:
        raise ValueError("DataFrame must contain 'Date' column")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    # ---------------- EMA ----------------
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA100"] = df["Close"].ewm(span=100, adjust=False).mean()
    df["EMA200"] = df["Close"].ewm(span=200, adjust=False).mean()
    df['Swing_Low_Causal'] = (df['Low'] == df['Low'].rolling(5).min())

    df["Pct_Below_20EMA"] = (
        (df["Close"] - df["EMA20"]) / df["EMA20"] * 100
    )

    # ---------------- RSI ----------------
    df["RSI14"] = talib.RSI(df["Close"], timeperiod=14)

    # ---------------- MACD ----------------
    macd, macdsignal, macdhist = talib.MACD(
        df["Close"],
        fastperiod=12,
        slowperiod=26,
        signalperiod=9
    )

    df["MACD"] = macd
    df["MACD_SIGNAL"] = macdsignal
    df["MACD_HIST"] = macdhist

    # ---------------- TRADING DAY OF MONTH ----------------
    df["YearMonth"] = df["Date"].dt.to_period("M")
    df["Trading_Day_of_Month"] = (
        df.groupby("YearMonth").cumcount() + 1
    )

    # ---------------- PREVIOUS MONTH WAS RED ----------------
    monthly_close = (
        df.groupby("YearMonth")["Close"].last()
    )

    prev_month_close = monthly_close.shift(1)

    df["Prev_Month_Close"] = (
        df["YearMonth"].map(prev_month_close)
    )

    df["Prev_Month_Was_Red"] = (
        df["Prev_Month_Close"] > df["Close"].shift(1)
    )

    # ---------------- BLOCK FIRST 5 DAYS ----------------
    df["Block_First_5"] = (
        df["Prev_Month_Was_Red"] &
        (df["Trading_Day_of_Month"] <= 5)
    )
    # ---------------- Swing Low (Causal) ----------------
    # ---------- Rolling structure ----------
    df["Low_20_Min"] = df["Low"].rolling(20, min_periods=20).min()
    df["Close_20_Max"] = df["Close"].rolling(20, min_periods=20).max()
    df["Low_5_Min"] = df["Close"].rolling(5, min_periods=5).min()
    df["Is_Swing_Low_10"] = (
            df["Low"] == df["Low"].rolling(10, min_periods=10).min()
    )

    # Cleanup helper columns (optional)
    df.drop(columns=["YearMonth", "Prev_Month_Close"], inplace=True)
    os.makedirs("computed", exist_ok=True)
    df.to_csv(f"computed/{symbol}_trades.csv", index=False)
    return df


# -------------------------------------------------
# SIGNAL GENERATION
# -------------------------------------------------

def generate_signal(row, vix_close=None):
    """
    Backtest-only: EMA50 pullback + swing-low strategy
    """

    # --------- Indicator warm-up guard ---------
    if pd.isna(row["EMA200"]):
        return None

    cond_buy = (
        (row["Is_Swing_Low_10"]) &
        (row["EMA50"] > row["EMA200"]) &
        (row["Low_20_Min"] < row["EMA50"]) &
        (row["Close_20_Max"] > row["EMA50"]) &
        (row['MACD_SIGNAL'] > row['MACD']) &
        (vix_close < 30)
    )

    if not cond_buy:
        return None

    entry_price = row["Close"]

    return {
        "entry_price": entry_price,
        "target_price": round(entry_price * 1.20, 2),
        "stop_price": round(entry_price * 0.90, 2),
        "signal_reason": "EMA50>EMA200 + Pullback + SwingLow"
    }
