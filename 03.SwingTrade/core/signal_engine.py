# core/signal_engine.py
######### Exit All
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
    df["Low_3M_Min"] = df["Low"].rolling(63).min()
    df["Low_25_Min"] = df["Low"].rolling(25).min()
    df["Low_5_Min"] = df["Low"].rolling(5).min()
    # --- ATR 14 (CORRECT implementation) ---
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()

    df["EMA20_prev"] = df["EMA20"].shift(1)
    df["EMA50_prev"] = df["EMA50"].shift(1)

    df["ATR14_20_MA"] = df["ATR14"].rolling(20).mean()
    df["Volume_20_MA"] = df["Volume"].rolling(20).mean()

    df["Pct_Below_20EMA"] = (
        (df["Close"] - df["EMA20"]) / df["EMA20"] * 100
    )
    df["VWAP"] = (
            (df["Close"] * df["Volume"]).cumsum() /
            df["Volume"].cumsum()
    )
    ###################################
    df["Open_1"] = df["Open"].shift(1)
    df["Close_1"] = df["Close"].shift(1)
    df["Low_1"] = df["Low"].shift(1)
    df["High_1"] = df["High"].shift(1)
    df["EMA200_1"] = df["EMA200"].shift(1)

    df["Low_20_Min"] = df["Low"].shift(1).rolling(20).min()
    df["Range"] = df["High"] - df["Low"]
    df["Range_10_Avg"] = df["Range"].rolling(10).mean()

    #--------- Leader Filter ----------
    # ---------------- ATR (EXIT EDGE) ----------------
    df["ATR14"] = talib.ATR(df["High"], df["Low"], df["Close"], 14)
    df["ATR14_20MA"] = df["ATR14"].rolling(20).mean()

    # ---------------- LEADER FILTER ----------------
    df["RS_52W"] = df["Close"] / df["Close"].rolling(252).max()

    # ---------------- TREND SLOPE ----------------
    df["EMA200_SLOPE"] = df["EMA200"].diff(20)

    # ---------------- PULLBACK STATE ----------------
    df["Pullback"] = df["Close"] < df["EMA20"]
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
    df["Inside_Bar"] = (
            (df["High"] < df["High"].shift(1)) &
            (df["Low"] > df["Low"].shift(1))
    )

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

def strategy_trend_breakout(row, vix_close):
    if pd.isna(row["Close_20_Max"]) or pd.isna(row["ATR14"]):
        return None

    cond = (
        (row["Close"] >= row["Close_20_Max"]) and
        (row["EMA20"] > row["EMA50"] > row["EMA200"]) and
        (row["RSI14"] > 60) and
        (row["RS_52W"] > 0.8) and
        (vix_close < 30)
    )

    if not cond:
        return None

    entry = row["Close"]

    return {
        "entry_price": entry,
        "target_price": round(entry + 5 * row["ATR14"], 2),
        "stop_price": round(entry - 2.5 * row["ATR14"], 2),
        "signal_reason": "Leader Trend Breakout"
    }

def strategy_post_correction_entry(row, vix_close):
    if pd.isna(row["ATR14"]) or pd.isna(row["EMA50"]):
        return None

    cond = (
        (row["RS_52W"] > 0.75) and
        (row["EMA50"] > row["EMA200"]) and
        (row["Low"] <= row["EMA50"]) and
        (row["Close"] > row["EMA50"]) and
        (row["RSI14"] > 45) and
        (vix_close < 28)
    )

    if not cond:
        return None

    entry = row["Close"]

    return {
        "entry_price": entry,
        "target_price": round(entry + 4 * row["ATR14"], 2),
        "stop_price": round(entry - 2 * row["ATR14"], 2),
        "signal_reason": "Post-Correction Re-entry"
    }
def strategy_leader_pullback(row, vix_close):
    if pd.isna(row["ATR14"]) or pd.isna(row["EMA200"]):
        return None

    cond = (
        # ---- Leader filter ----
        (row["RS_52W"] > 0.75) and

        # ---- Strong trend ----
        (row["EMA50"] > row["EMA200"]) and
        (row["EMA200_SLOPE"] > 0) and

        # ---- Controlled pullback ----
        (row["Close"] < row["EMA20"]) and
        (40 <= row["RSI14"] <= 55) and

        # ---- Risk filter ----
        (vix_close < 30)
    )

    if not cond:
        return None

    entry = row["Close"]

    return {
        "entry_price": entry,
        "target_price": round(entry + 4 * row["ATR14"], 2),
        "stop_price": round(entry - 2 * row["ATR14"], 2),
        "signal_reason": "Leader Pullback + ATR Exit"
    }

def strategy_rsi_mean_reversion(row, vix_close):
    if pd.isna(row["RSI14"]):
        return None

    cond = (
        row["RSI14"] < 30 and
        row["Close"] > row["EMA200"] and
        row["Low"] == row["Low_5_Min"] and
        (vix_close < 30)
    )

    if not cond:
        return None

    entry = row["Close"]
    return {
        "entry_price": entry,
        "target_price": round(entry * 1.12, 2),
        "stop_price": round(entry * 0.95, 2),
        "signal_reason": "RSI Mean Reversion"
    }
def strategy_ema_pullback(row, vix_close):
    if pd.isna(row["EMA200"]):
        return None

    cond = (
        row["Is_Swing_Low_10"] and
        row["EMA50"] > row["EMA200"] and
        row["Low_20_Min"] < row["EMA50"] and
        row["Close_20_Max"] > row["EMA50"] and
        row["MACD_SIGNAL"] > row["MACD"] and
        vix_close < 30
    )

    if not cond:
        return None

    entry = row["Close"]
    return {
        "entry_price": entry,
        "target_price": round(entry * 1.18, 2),
        "stop_price": round(entry * 0.92, 2),
        "signal_reason": "EMA Pullback + Swing Low"
    }

def strategy_trend_breakout(row, vix_close):
    if pd.isna(row["Close_20_Max"]):
        return None

    cond = (
        row["Close"] >= row["Close_20_Max"] and
        row["EMA20"] > row["EMA50"] > row["EMA200"] and
        row["RSI14"] > 55 and
        vix_close < 30
    )

    if not cond:
        return None

    entry = row["Close"]
    return {
        "entry_price": entry,
        "target_price": round(entry * 1.25, 2),
        "stop_price": round(entry * 0.90, 2),
        "signal_reason": "Trend Breakout"
    }
def strategy_vwap_reclaim(row , vix_close):
    if pd.isna(row["VWAP"]):
        return None

    cond = (
        row["Low"] < row["VWAP"] and
        row["Close"] > row["VWAP"] and
        row["RSI14"] > 40
    )

    if not cond:
        return None

    entry = row["Close"]
    return {
        "entry_price": entry,
        "target_price": round(entry * 1.15, 2),
        "stop_price": round(entry * 0.94, 2),
        "signal_reason": "VWAP Reclaim"
    }
def strategy_post_correction_entry(row, vix_close):
    if pd.isna(row["ATR14"]) or pd.isna(row["EMA50"]):
        return None

    cond = (
        (row["RS_52W"] > 0.75) and
        (row["EMA50"] > row["EMA200"]) and
        (row["Low"] <= row["EMA50"]) and
        (row["Close"] > row["EMA50"]) and
        (row["RSI14"] > 45) and
        (vix_close < 28)
    )

    if not cond:
        return None

    entry = row["Close"]

    return {
        "entry_price": entry,
        "target_price": round(entry + 4 * row["ATR14"], 2),
        "stop_price": round(entry - 2 * row["ATR14"], 2),
        "signal_reason": "Post-Correction Re-entry"
    }
def strategy_volatility_contraction(row, vix_close):
    if pd.isna(row["ATR14"]) or pd.isna(row["ATR14_20MA"]):
        return None

    cond = (
        (row["RS_52W"] > 0.8) and
        (row["ATR14"] < row["ATR14_20MA"]) and
        (row["Close"] > row["EMA50"]) and
        (row["EMA50"] > row["EMA200"]) and
        (vix_close < 25)
    )

    if not cond:
        return None

    entry = row["Close"]

    return {
        "entry_price": entry,
        "target_price": round(entry + 6 * row["ATR14"], 2),
        "stop_price": round(entry - 2 * row["ATR14"], 2),
        "signal_reason": "Volatility Contraction Breakout"
    }
def strategy_defensive_swing(row, vix_close):
    if pd.isna(row["ATR14"]):
        return None

    cond = (
        (row["RS_52W"] > 0.7) and
        (row["EMA20"] > row["EMA50"]) and
        (row["EMA50"] > row["EMA200"]) and
        (row["RSI14"] > 50) and
        (vix_close < 30)
    )

    if not cond:
        return None

    entry = row["Close"]

    return {
        "entry_price": entry,
        "target_price": round(entry + 3 * row["ATR14"], 2),
        "stop_price": round(entry - 1.5 * row["ATR14"], 2),
        "signal_reason": "Defensive Trend Swing"
    }

def swing_casual_low(row, vix_close):

    cond = (
        (row["Swing_Low_Causal"]) and
        (row['Pct_Below_20EMA']) and
        (row["EMA20"] > row["EMA50"]) and
        (row["EMA50"] > row["EMA200"]) and
        (vix_close < 30)
    )

    if not cond:
        return None

    entry = row["Close"]

    return {
        "entry_price": entry,
        "target_price": round(entry + 3 * row["ATR14"], 2),
        "stop_price": round(entry - 1.5 * row["ATR14"], 2),
        "signal_reason": "Defensive Trend Swing"
    }
def swing_3m_low_ema_crossover(row, vix_close):
    """
    Swing reversal strategy:
    - 3M lowest low compressed into 25D low
    - Fresh EMA20 / EMA50 bullish crossover
    - Structural 5D swing-low stop
    - Volume + volatility compression filters
    - No external market_row dependency
    """

    # ---- Guard clauses (hard stop for bad rows) ----
    required_cols = [
        "Low_3M_Min", "Low_25_Min", "Low_5_Min",
        "EMA20", "EMA50", "EMA200",
        "EMA20_prev", "EMA50_prev",
        "ATR14", "ATR14_20_MA",
        "Volume", "Volume_20_MA",
        "Close", "Low"
    ]

    for col in required_cols:
        if col not in row or pd.isna(row[col]):
            return None

    # ---- Core conditions ----
    cond = (
        # Structural base compression
        (row["Low_3M_Min"] <= row["Low_25_Min"]) and

        # Fresh bullish EMA crossover
        (row["EMA20_prev"] <= row["EMA50_prev"]) and
        (row["EMA20"] > row["EMA50"]) and

        # Trend alignment
        (row["EMA50"] > row["EMA200"]) and

        # Retest + hold above recent structure
        (row["Close"] > row["Low_5_Min"]) and
        (row["Low"] > row["Low_5_Min"]) and

        # Avoid extended entries
        ((row["Close"] - row["Low_5_Min"]) / row["Low_5_Min"] < 0.08) and

        # Volume confirmation
        (row["Volume"] > 1.3 * row["Volume_20_MA"]) and

        # Volatility compression
        (row["ATR14"] < row["ATR14_20_MA"]) and

        # Risk environment
        (vix_close < 30)
    )

    if not cond:
        return None

    # ---- Trade construction ----
    entry = row["Close"]
    stop  = row["Low_5_Min"]

    # Safety check
    if stop >= entry:
        return None

    risk = entry - stop
    target = entry + 3 * risk

    return {
        "entry_price": round(entry, 2),
        "stop_price": round(stop, 2),
        "target_price": round(target, 2),
        "signal_reason": (
            "3M Low + 25D Compression + EMA20/50 Cross | "
            "Retest Hold | Volume + Volatility Filter"
        )
    }

def strategy_accumulation_manipulation_breakout(row):
    if (
        pd.isna(row["EMA200"]) or
        pd.isna(row["Low_20_Min"]) or
        pd.isna(row["Range_10_Avg"])
    ):
        return None

    cond = (
        # ===== ACCUMULATION =====
        row["Range_10_Avg"] < row["ATR14"] * 0.8 and

        # ===== MANIPULATION =====
        row["Close_1"] < row["Open_1"] and
        row["Low_1"] < row["EMA200_1"] and
        row["Low_1"] == row["Low_20_Min"] and

        # ===== EXPANSION =====
        row["Close"] > row["Open"] and
        row["Close"] > row["EMA200"] and
        row["Close"] > row["High_1"]
    )

    if not cond:
        return None

    entry = row["Close"]
    stop  = round(row["Low_1"] * 0.99, 2)
    target = round(entry * 1.10, 2)

    return {
        "entry_price": entry,
        "target_price": target,
        "stop_price": stop,
        "signal_reason": "Accumulation → Manipulation → Expansion Breakout"
    }


# -------------------------------------------------
# SIGNAL GENERATION
# -------------------------------------------------

def generate_signal(row, vix_close=None):
    #return strategy_trend_breakout(row, vix_close)
    #return strategy_post_correction_entry(row, vix_close)
    #return strategy_leader_pullback(row, vix_close)
    #return strategy_ema200_reclaim(row, vix_close)
    #return swing_casual_low(row, vix_close)
    #return strategy_post_correction_entry(row, vix_close)
    return strategy_accumulation_manipulation_breakout(row)
    #return strategy_volatility_contraction(row, vix_close)
    #return strategy_defensive_swing(row, vix_close)
    #return swing_3m_low_ema_crossover(row, vix_close)