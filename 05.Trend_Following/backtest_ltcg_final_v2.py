"""
LTCG Positional Backtesting Engine
====================================
Strategy : Pullback to 20 EMA in uptrend → hold 12+ months for LTCG 10% tax
Entry    : LIMIT ORDER at signal day close price
           Evening scanner fires at 4:30 PM → signal found → place limit order
           at today's close price for tomorrow.
           Next day: stock must trade at or below that limit price to fill.
           If stock gaps UP and never comes back → order NOT filled → no trade.
           If stock gaps DOWN or trades at/near close → order fills at limit.
           This ensures you always buy at the price you saw in the scanner —
           never chasing a gap-up open.
Exit     : Stop Loss (swing low) / Max hold 400 days (LTCG) / Target (RR=5)
Tax      : LTCG 10% if held >= 366 days, else STCG 15%

Entry logic detail:
  Signal day (today)  : scanner fires, close = ₹233.50
  Limit order placed  : ₹233.50 (signal day close)
  Next day scenario A : opens at ₹240 (gap up), never comes to ₹233.50 → NO FILL
  Next day scenario B : opens at ₹225 (gap down), immediately fills at ₹233.50 → FILL
  Next day scenario C : opens at ₹234, dips to ₹232 during day → FILL at ₹233.50
  Next day scenario D : opens at ₹233 → FILL at ₹233.50 immediately

Directory structure (auto-created):
  ../data/START_END/          ← CSV files (stocks + indices)
  ../reports/LTCG/START_END/  ← backtest_report.html, trade_log.csv, daily_rankings.csv

USAGE:
    python backtest_ltcg_final.py
    python backtest_ltcg_final.py --symbol RELIANCE
    python backtest_ltcg_final.py --config config_ltcg.json
"""

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")


# ─── CONFIG & PATHS ───────────────────────────────────────────────────────────

def load_config(path="config_ltcg.json") -> dict:
    with open(path) as f:
        return json.load(f)

def load_watchlist(cfg: dict) -> dict:
    with open(cfg["stock_selection"]["watchlist_file"]) as f:
        return json.load(f)

def get_data_dir(cfg: dict) -> Path:
    start = cfg["backtest"]["start_date"]
    end   = cfg["backtest"]["end_date"]
    d = Path(cfg["output"]["data_dir"]) / f"{start}_{end}"
    d.mkdir(parents=True, exist_ok=True)
    return d

def get_report_dir(cfg: dict) -> Path:
    start = cfg["backtest"]["start_date"]
    end   = cfg["backtest"]["end_date"]
    d = Path(cfg["output"]["report_dir"]) / f"{start}_{end}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ─── INDICATORS ───────────────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Adds all technical indicators to stock OHLCV data.
    Includes warmup candles so EMAs are accurate from start_date.
    """
    s  = cfg["strategy"]
    df = df.copy()

    # EMAs
    df["ema_fast"] = df["close"].ewm(span=s["ema_fast"], adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=s["ema_slow"], adjust=False).mean()

    # Volume average
    df["vol_avg"]  = df["volume"].rolling(s["volume_avg_period"]).mean()

    # Swing low — for stop loss calculation
    df["swing_low"] = df["low"].rolling(cfg["exit_rules"]["swing_low_candles"]).min()

    # ATR — for trailing stop
    df["tr"]  = np.maximum(
                    df["high"] - df["low"],
                    np.maximum(
                        abs(df["high"] - df["close"].shift(1)),
                        abs(df["low"]  - df["close"].shift(1))
                    ))
    df["atr"] = df["tr"].rolling(14).mean()

    # Momentum — for ranking
    df["momentum"] = df["close"].pct_change(cfg["stock_selection"]["ranking"]["momentum_days"])

    # ── RSI (14) — for entry filter ──
    # Measures momentum recovery. RSI > 45 means stock is bouncing,
    # not still falling. Eliminates entries where pullback is actually
    # a downtrend continuation (responsible for most 0-30 day losses).
    rsi_period = s.get("rsi_period", 14)
    delta      = df["close"].diff()
    gain       = delta.clip(lower=0).rolling(rsi_period).mean()
    loss_      = (-delta.clip(upper=0)).rolling(rsi_period).mean()
    rs         = gain / loss_.replace(0, np.nan)
    df["rsi"]  = 100 - (100 / (1 + rs))

    # ── Entry Signal ──
    # All conditions must be true on the same candle:
    # 1. In uptrend    : close above both 20 EMA and 50 EMA
    # 2. Pullback      : low touched 20 EMA zone (within tolerance%)
    # 3. Bullish body  : close > open, body >= 50% of candle range
    # 4. Volume        : today's volume > 20-day avg × multiplier
    # 5. RSI recovery  : RSI(14) > rsi_min_entry (default 45)
    #                    Confirms bounce, not continued downtrend
    tol = s["pullback_tolerance_pct"] / 100
    rsi_min = s.get("rsi_min_entry", 45)

    df["in_uptrend"]     = (df["close"] > df["ema_fast"]) & (df["close"] > df["ema_slow"])
    df["pullback"]       = (df["low"] <= df["ema_fast"] * (1 + tol)) & (df["close"] > df["ema_fast"])
    body = abs(df["close"] - df["open"])
    rng  = df["high"] - df["low"]
    df["bullish_candle"] = (df["close"] > df["open"]) & (body >= s["bullish_body_pct"] / 100 * rng.replace(0, np.nan))
    df["vol_confirmed"]  = df["volume"] > df["vol_avg"] * s["volume_multiplier"]
    df["rsi_confirmed"]  = df["rsi"] > rsi_min

    df["signal"] = (
        df["in_uptrend"]    &
        df["pullback"]      &
        df["bullish_candle"] &
        df["vol_confirmed"] &
        df["rsi_confirmed"]
    )
    return df


# ─── MARKET FILTERS ───────────────────────────────────────────────────────────

def build_vix_map(vix_df: pd.DataFrame, cfg: dict) -> dict:
    """
    Returns {date: (vix_5day_avg, regime)}
    regime: 'normal' | 'caution' | 'no_entry'

    no_entry : VIX >= vix_no_entry_above (default 20) → block all new entries
    caution  : VIX >= vix_caution_above  (default 16) → only high-rank entries allowed
    normal   : VIX below both thresholds → trade freely
    """
    vcfg = cfg.get("volatility_filter", {})
    if not vcfg.get("enabled", False) or vix_df is None or vix_df.empty:
        return {}

    no_entry_thresh = vcfg.get("vix_no_entry_above", 20.0)
    caution_thresh  = vcfg.get("vix_caution_above",  16.0)
    lookback        = vcfg.get("vix_lookback_days",   5)

    vix_df = vix_df.sort_values("date").copy()
    vix_df["vix_avg"] = vix_df["close"].rolling(lookback, min_periods=1).mean()

    result = {}
    for _, row in vix_df.iterrows():
        v = row["vix_avg"]
        regime = "no_entry" if v >= no_entry_thresh else ("caution" if v >= caution_thresh else "normal")
        result[row["date"]] = (round(v, 2), regime)
    return result


def build_nifty_trend_map(nifty_df: pd.DataFrame, cfg: dict) -> dict:
    """
    Returns {date: info_dict} with trend filter results for each day.

    Three filters (all must pass for new entry to be allowed):
      1. nifty_trend_filter  : Nifty close > 50 EMA  (medium-term uptrend)
      2. nifty_200ema_filter : Nifty close > 200 EMA (long-term bull market)
      3. adx_filter          : Nifty ADX >= threshold (market is trending, not choppy)

    ADX caution zone (adx_caution_threshold to adx_min_threshold):
      Weak trend forming — only allow high-conviction entries (rank_score >= threshold)
    """
    vcfg = cfg.get("volatility_filter", {})
    if nifty_df is None or nifty_df.empty:
        return {}

    df = nifty_df.copy().sort_values("date").reset_index(drop=True)

    # EMAs on Nifty
    df["ema50"]  = df["close"].ewm(span=50,  adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    # ADX — Wilder's Average Directional Index
    # Measures trend STRENGTH (not direction). >25 = strong trend. <20 = choppy.
    adx_period = vcfg.get("adx_period", 14)
    high = df["high"]; low = df["low"]; close = df["close"]

    df["tr"]       = np.maximum(high - low,
                     np.maximum(abs(high - close.shift(1)),
                                abs(low  - close.shift(1))))
    df["dm_plus"]  = np.where((high - high.shift(1)) > (low.shift(1) - low),
                              np.maximum(high - high.shift(1), 0), 0)
    df["dm_minus"] = np.where((low.shift(1) - low) > (high - high.shift(1)),
                              np.maximum(low.shift(1) - low, 0), 0)

    def wilder_smooth(series, period):
        result = series.copy() * 0.0
        result.iloc[period] = series.iloc[1:period+1].sum()
        for i in range(period + 1, len(series)):
            result.iloc[i] = result.iloc[i-1] - (result.iloc[i-1] / period) + series.iloc[i]
        return result

    atr_w  = wilder_smooth(df["tr"],       adx_period)
    dmp_w  = wilder_smooth(df["dm_plus"],  adx_period)
    dmm_w  = wilder_smooth(df["dm_minus"], adx_period)
    df["di_plus"]  = (dmp_w / atr_w.replace(0, np.nan)) * 100
    df["di_minus"] = (dmm_w / atr_w.replace(0, np.nan)) * 100
    di_sum = df["di_plus"] + df["di_minus"]
    df["dx"]  = (abs(df["di_plus"] - df["di_minus"]) / di_sum.replace(0, np.nan)) * 100
    df["adx"] = wilder_smooth(df["dx"], adx_period)

    adx_min           = vcfg.get("adx_min_threshold",    25)
    adx_caution_level = vcfg.get("adx_caution_threshold", 20)
    adx_caution_score = vcfg.get("adx_caution_min_score", 0.65)

    result = {}
    for _, row in df.iterrows():
        checks   = []
        adx_val  = row.get("adx", np.nan)
        ema50_ok = bool(row["close"] > row["ema50"])
        ema200_ok= bool(row["close"] > row["ema200"])

        if vcfg.get("nifty_trend_filter",  False): checks.append(ema50_ok)
        if vcfg.get("nifty_200ema_filter", False): checks.append(ema200_ok)
        if vcfg.get("adx_filter", False) and not np.isnan(adx_val):
            checks.append(adx_val >= adx_min)

        result[row["date"]] = {
            "trend_ok":         all(checks) if checks else True,
            "ema50_ok":         ema50_ok,
            "ema200_ok":        ema200_ok,
            "adx":              round(float(adx_val), 2) if not np.isnan(adx_val) else 0,
            "adx_caution":      (vcfg.get("adx_filter", False) and
                                 not np.isnan(adx_val) and
                                 adx_caution_level <= adx_val < adx_min),
            "adx_caution_score": adx_caution_score,
        }
    return result


# ─── RANKING ──────────────────────────────────────────────────────────────────

def compute_rank_score(row: dict, nifty_return: float, cfg: dict) -> float:
    """
    Score = weighted sum of 5 factors (each normalised 0-1):
      trend_strength    : how far close is above 20 EMA (momentum confirmation)
      relative_strength : stock return vs Nifty over rs_lookback_days
      volume_surge      : today's volume vs 20-day average
      momentum          : price change over momentum_days
      volatility_score  : lower ATR% = more stable = higher score
    """
    w = cfg["stock_selection"]["ranking"]["weights"]
    s = {}
    s["trend_strength"]    = max(0, min(row.get("ema_fast_gap", 0), 0.1) / 0.1)
    rs = row.get("rs_return", 0) - nifty_return
    s["relative_strength"] = min(max((rs + 0.1) / 0.2, 0), 1)
    s["volume_surge"]      = min(row.get("vol_ratio", 1), 3) / 3
    s["momentum"]          = min(max((row.get("momentum", 0) + 0.05) / 0.1, 0), 1)
    s["volatility_score"]  = max(0, 1 - (row.get("atr_pct", 0.02) / 0.05))
    return round(sum(s[k] * w[k] for k in w), 4)


def rank_stocks_on_date(all_data: dict, target_date, nifty_df: pd.DataFrame,
                        cfg: dict, wl_meta: dict) -> pd.DataFrame:
    """
    Ranks all stocks on a given date.
    Returns DataFrame sorted by rank_score descending.
    Applies sector_limit to ensure diversification.
    """
    rk      = cfg["stock_selection"]["ranking"]
    rs_days = rk["rs_lookback_days"]

    # Nifty return over same period (for relative strength calculation)
    nifty_past   = nifty_df[nifty_df["date"] <= target_date].tail(rs_days + 1)
    nifty_return = 0.0
    if len(nifty_past) >= 2:
        nifty_return = ((nifty_past["close"].iloc[-1] - nifty_past["close"].iloc[0])
                        / nifty_past["close"].iloc[0])

    sym_sector = {s: sec for sec, syms in wl_meta.get("sector_groups", {}).items() for s in syms}
    records    = []

    for symbol, df in all_data.items():
        dft = df[df["date"] <= target_date]
        if len(dft) < rs_days + 5:
            continue
        latest    = dft.iloc[-1]
        past      = dft.tail(rs_days + 1)
        ema_gap   = (latest["close"] - latest["ema_fast"]) / latest["ema_fast"] if latest["ema_fast"] > 0 else 0
        rs_return = ((past["close"].iloc[-1] - past["close"].iloc[0])
                     / past["close"].iloc[0]) if past["close"].iloc[0] > 0 else 0
        vol_ratio = latest["volume"] / latest["vol_avg"] if latest["vol_avg"] > 0 else 1
        atr_pct   = latest["atr"] / latest["close"] if latest["close"] > 0 else 0.02

        score = compute_rank_score({
            "ema_fast_gap": ema_gap,
            "rs_return":    rs_return,
            "vol_ratio":    vol_ratio,
            "momentum":     latest.get("momentum", 0),
            "atr_pct":      atr_pct,
        }, nifty_return, cfg)

        records.append({
            "symbol":     symbol,
            "date":       target_date,
            "sector":     sym_sector.get(symbol, "Unknown"),
            "rank_score": score,
            "signal":     bool(latest.get("signal", False)),
        })

    if not records:
        return pd.DataFrame()

    # Sort by score, apply sector limit
    ranked       = pd.DataFrame(records).sort_values("rank_score", ascending=False)
    sector_count = {}
    filtered     = []
    for _, r in ranked.iterrows():
        sec = r["sector"]
        if sector_count.get(sec, 0) < rk["sector_limit"]:
            filtered.append(r)
            sector_count[sec] = sector_count.get(sec, 0) + 1
    return pd.DataFrame(filtered)


# ─── CHARGES & TAX ────────────────────────────────────────────────────────────

def calc_charges(buy_val: float, sell_val: float, cfg: dict) -> dict:
    tc  = cfg["taxes_and_charges"]
    brk = tc["brokerage_per_order"] * 2          # buy + sell
    stt = sell_val  * tc["stt_pct"]          / 100
    exc = (buy_val + sell_val) * tc["exchange_charges_pct"] / 100
    sbi = (buy_val + sell_val) * tc["sebi_charges_pct"]    / 100
    stm = buy_val  * tc["stamp_duty_pct"]    / 100
    gst = (brk + exc) * tc["gst_pct"]       / 100
    return {
        "brokerage": round(brk, 2), "stt": round(stt, 2),
        "exchange":  round(exc, 2), "sebi": round(sbi, 2),
        "stamp":     round(stm, 2), "gst":  round(gst, 2),
        "total":     round(brk + stt + exc + sbi + stm + gst, 2),
    }


def calc_tax(net_profit: float, hold_days: int, cfg: dict) -> tuple:
    """
    Returns (tax_amount, tax_type_str)
    LTCG (10%) if hold >= 366 days AND ltcg_tax_pct exists in config
    STCG (15%) otherwise
    No tax on losses.
    """
    if net_profit <= 0:
        return 0.0, "NONE"
    if hold_days >= 366 and "ltcg_tax_pct" in cfg["taxes_and_charges"]:
        rate = cfg["taxes_and_charges"]["ltcg_tax_pct"] / 100
        return round(net_profit * rate, 2), "LTCG"
    else:
        rate = cfg["taxes_and_charges"]["stcg_tax_pct"] / 100
        return round(net_profit * rate, 2), "STCG"


def get_sl(row, entry_price: float, cfg: dict) -> float:
    """
    Calculate stop loss based on stop_loss_type in config.
    swing_low : lowest low of last N candles minus buffer%
    atr_based : entry - ATR × multiplier
    fixed_pct : entry × (1 - fixed%)
    Hard cap  : SL cannot be more than 15% below entry (safety net)
    """
    sl_type = cfg["exit_rules"]["stop_loss_type"]
    buf     = cfg["exit_rules"]["swing_low_buffer_pct"] / 100

    if sl_type == "swing_low":
        sl = row["swing_low"] * (1 - buf)
    elif sl_type == "atr_based":
        sl = entry_price - row["atr"] * cfg["exit_rules"]["stop_loss_atr_multiplier"]
    else:
        sl = entry_price * (1 - cfg["exit_rules"]["stop_loss_fixed_pct"] / 100)

    return round(max(sl, entry_price * 0.85), 2)  # hard cap: max 15% SL


def calc_position_size(entry_price: float, sl: float, cfg: dict) -> int:
    """
    Position sizing — two constraints, take the smaller qty:

    1. RISK CONSTRAINT (risk_per_trade_pct):
       Never lose more than X% of capital if SL hits.
       risk_amt = capital × 2% = ₹10,000
       qty_by_risk = risk_amt / (entry - SL)

    2. POSITION CAP (max_position_pct):
       Never deploy more than Y% of capital in one trade.
       max_buy = capital × 30% = ₹1,50,000
       qty_by_cap = max_buy / entry_price

    Example with max_open_trades:
      max_open_trades=5  → max_position_pct=20% → 5×20%=100% capital used
      max_open_trades=10 → max_position_pct=10% → 10×10%=100% capital used

    Current config: max_position_pct=30%, max_open_trades=5
      Worst case: 5 × ₹1.5L = ₹7.5L (over ₹5L capital)
      BUT: rarely 5 open simultaneously due to filters
      Average 2-3 open = ₹3-4.5L deployed (fits comfortably)
    """
    capital       = cfg["capital"]["initial_capital"]
    risk_pct      = cfg["capital"]["risk_per_trade_pct"]
    max_pos_pct   = cfg["capital"].get("max_position_pct", 20.0)

    risk          = entry_price - sl
    if risk <= 0:
        return 0

    risk_amt      = capital * risk_pct / 100
    qty_by_risk   = int(risk_amt / risk)

    max_buy_value = capital * max_pos_pct / 100
    qty_by_cap    = int(max_buy_value / entry_price)

    return max(1, min(qty_by_risk, qty_by_cap))


def _build_trade(symbol, entry_date, exit_date, entry, exit_p,
                 qty, sl, target, reason, cfg) -> dict:
    """Build a complete trade record with all P&L, charges, and tax."""
    buy_val     = entry  * qty
    sell_val    = exit_p * qty
    gross       = sell_val - buy_val
    ch          = calc_charges(buy_val, sell_val, cfg)
    net_pre_tax = gross - ch["total"]
    hold_days   = (pd.to_datetime(exit_date) - pd.to_datetime(entry_date)).days
    tax, tax_type = calc_tax(net_pre_tax, hold_days, cfg)
    net_pnl     = net_pre_tax - tax
    risk        = entry - sl
    rr_achieved = round((exit_p - entry) / risk, 2) if risk > 0 else 0

    return {
        "symbol":       symbol,
        "entry_date":   str(entry_date),
        "exit_date":    str(exit_date),
        "entry_price":  round(entry, 2),
        "stop_loss":    round(sl, 2),
        "target":       round(target, 2),
        "exit_price":   round(exit_p, 2),
        "qty":          qty,
        "buy_value":    round(buy_val, 2),
        "sell_value":   round(sell_val, 2),
        "gross_pnl":    round(gross, 2),
        "brokerage":    ch["brokerage"],
        "stt":          ch["stt"],
        "exchange":     ch["exchange"],
        "sebi":         ch["sebi"],
        "stamp":        ch["stamp"],
        "gst":          ch["gst"],
        "total_charges":ch["total"],
        "net_before_tax": round(net_pre_tax, 2),
        "tax_type":     tax_type,
        "stcg_tax":     tax,
        "net_pnl":      round(net_pnl, 2),
        "return_pct":   round((exit_p - entry) / entry * 100, 2),
        "rr_achieved":  rr_achieved,
        "exit_reason":  reason,
        "hold_days":    hold_days,
        "is_ltcg":      tax_type == "LTCG",
        "result":       "WIN" if net_pnl > 0 else "LOSS",
    }


# ─── BACKTEST ONE SYMBOL ──────────────────────────────────────────────────────

def backtest_symbol(df: pd.DataFrame, symbol: str, cfg: dict,
                    eligible_dates: set, vix_map: dict, nifty_trend: dict) -> list:
    """
    Exact replication of the original working LTCG backtest logic.

    How LTCG is achieved:
      1. Trade enters on pullback signal
      2. ltcg_protect prevents forced exit before 366 days if trade is profitable
      3. max_hold_exit fires at 400 days → already past 366 → LTCG 10% tax
      4. Trailing stop only activates after partial_done=True
         (partial_exit=False in config → partial_done never set → trailing never fires)
         This means the trade holds until SL, target, or max_hold_exit

    Exit hierarchy:
      - SL hit anytime → exit immediately (STCG if < 366d)
      - Target hit anytime → exit (STCG if < 366d)
      - max_hold (400d) → ltcg_protect keeps profitable trades → exits at 400d (LTCG)
      - end of backtest → forced_close
    """
    trades        = []
    rr            = cfg["trade_rules"]["reward_to_risk_ratio"]
    gap_up        = cfg["trade_rules"]["gap_filter_up_pct"]   / 100
    gap_dn        = cfg["trade_rules"]["gap_filter_down_pct"] / 100
    close_last    = cfg["backtest"]["close_all_on_last_day"]
    max_hold      = cfg["backtest"].get("max_hold_days", 400)
    capital       = cfg["capital"]["initial_capital"]
    vcfg          = cfg.get("volatility_filter", {})
    caution_score = vcfg.get("vix_caution_min_score", 0.65)

    ex           = cfg["exit_rules"]
    do_partial   = ex.get("partial_exit", False)
    partial_pct  = ex.get("partial_exit_pct", 50) / 100
    partial_rr   = ex.get("partial_exit_at_rr", 1.0)
    move_be      = ex.get("move_sl_to_breakeven", True)
    do_trail     = ex.get("trailing_stop", True)
    trail_atr    = ex.get("trailing_stop_atr_multiplier", 2.0)
    ltcg_protect = ex.get("ltcg_protect", True)
    ltcg_days    = cfg["backtest"].get("ltcg_hold_days", 366)

    # Trade state — using partial_done as breakeven flag (same as original)
    in_trade     = False
    partial_done = False   # True after partial exit OR breakeven SL set
    entry_price  = sl = target = qty = remaining_qty = rank_score = 0
    entry_date   = None

    for i in range(1, len(df) - 1):
        row      = df.iloc[i]
        next_row = df.iloc[i + 1]
        is_last  = (i == len(df) - 2)

        in_trading_period = bool(row.get("in_trading_period", True))

        if in_trade:
            hold_days = (pd.to_datetime(next_row["date"]) - pd.to_datetime(entry_date)).days

            # ── Trailing stop (only after partial_done / breakeven set) ──
            if do_trail and partial_done:
                trail_sl = next_row["close"] - row["atr"] * trail_atr
                if trail_sl > sl:
                    sl = round(trail_sl, 2)

            # ── Force close / max hold ──
            if (is_last and close_last) or hold_days >= max_hold:
                current_pnl = (next_row["open"] - entry_price) * remaining_qty
                # LTCG protect: if profitable and under 366 days, keep holding
                if ltcg_protect and hold_days < ltcg_days and current_pnl > 0 and not is_last:
                    continue
                reason = "forced_close" if (is_last and close_last) else "max_hold_exit"
                trades.append(_build_trade(symbol, entry_date, next_row["date"],
                                     entry_price, next_row["open"], remaining_qty,
                                     sl, target, reason, cfg))
                in_trade = False; partial_done = False
                continue

            # ── Partial exit (sets partial_done → enables trailing stop) ──
            if do_partial and not partial_done:
                partial_trigger = entry_price + (entry_price - sl) * partial_rr
                if next_row["high"] >= partial_trigger:
                    partial_qty = max(1, int(remaining_qty * partial_pct))
                    trades.append(_build_trade(symbol, entry_date, next_row["date"],
                                         entry_price, partial_trigger, partial_qty,
                                         sl, target, "partial_exit", cfg))
                    remaining_qty -= partial_qty
                    partial_done   = True
                    if move_be:
                        sl = entry_price
                    if remaining_qty <= 0:
                        in_trade = False; partial_done = False
                        continue

            # ── SL hit or Target hit ──
            exit_p = reason = None
            if next_row["low"] <= sl:
                exit_p, reason = sl, "stop_loss"
            elif next_row["high"] >= target:
                exit_p, reason = target, "target_hit"
            if exit_p and remaining_qty > 0:
                trades.append(_build_trade(symbol, entry_date, next_row["date"],
                                     entry_price, exit_p, remaining_qty,
                                     sl, target, reason, cfg))
                in_trade = False; partial_done = False

        else:
            # ── Entry logic ──
            if not in_trading_period:
                continue

            if not row.get("signal", False) or row["date"] not in eligible_dates:
                continue

            # VIX filter
            vix_info = vix_map.get(row["date"])
            if vix_info:
                vix_avg, regime = vix_info
                if regime == "no_entry":
                    continue
                if regime == "caution" and rank_score < caution_score:
                    continue

            # Nifty trend filter
            if nifty_trend:
                info = nifty_trend.get(row["date"])
                if info:
                    if not info.get("trend_ok", True):
                        continue
                    if info.get("adx_caution", False):
                        if rank_score < info.get("adx_caution_score", 0.65):
                            continue

            # ── Gap filter (removed — limit order handles this naturally) ──
            # Old approach: enter at next_day open → skip if gap > threshold
            # New approach: limit order at signal_day close
            #   Gap up  → stock never comes back to limit → no fill → naturally skipped
            #   Gap down → fills at limit price (better than open) → take it
            # No manual gap filter needed — limit order IS the gap filter

            # ── Limit order entry at signal day close ──
            # Place limit order at today's close (signal day)
            # Fill only if next day's LOW touches or goes below that limit price
            limit_price = row["close"]   # signal day close = your limit order price

            # Check if next day trades at or below limit price
            # next_row["low"] <= limit_price means the stock came down to your price
            if next_row["low"] > limit_price:
                # Stock gapped up and never came back — limit order not filled
                continue   # no trade today

            # Limit order filled at limit_price (not at open)
            # Even if stock opened below limit, you get limit_price (limit order guarantee)
            entry_price = limit_price
            sl            = get_sl(row, entry_price, cfg)
            if entry_price <= sl or (entry_price - sl) < 0.5:
                continue

            risk          = entry_price - sl
            target        = round(entry_price + risk * rr, 2)

            # Position sizing — min of risk-based and position-cap
            risk_amt      = capital * cfg["capital"]["risk_per_trade_pct"] / 100
            qty_by_risk   = int(risk_amt / risk) if risk > 0 else 0
            max_pos_pct   = cfg["capital"].get("max_position_pct", 20.0)
            max_buy_value = capital * max_pos_pct / 100
            qty_by_cap    = int(max_buy_value / entry_price) if entry_price > 0 else 0
            qty           = max(1, min(qty_by_risk, qty_by_cap))

            remaining_qty = qty
            entry_date    = next_row["date"]
            partial_done  = False
            in_trade      = True

    return trades


# ─── STATISTICS ───────────────────────────────────────────────────────────────

def compute_stats(trades: list, cfg: dict) -> dict:
    if not trades:
        return {}

    df   = pd.DataFrame(trades)
    wins = df[df["net_pnl"] > 0]
    loss = df[df["net_pnl"] <= 0]
    ltcg = df[df["is_ltcg"] == True]
    stcg = df[df["is_ltcg"] == False]

    cap      = cfg["capital"]["initial_capital"]
    gp       = wins["net_pnl"].sum()
    gl       = abs(loss["net_pnl"].sum())
    std      = df["net_pnl"].std()
    sharpe   = round(df["net_pnl"].mean() / std * np.sqrt(252), 2) if std > 0 else 0

    # Equity curve & drawdown
    df_s         = df.sort_values("exit_date").reset_index(drop=True)
    df_s["cum"]  = df_s["net_pnl"].cumsum()
    df_s["eq"]   = cap + df_s["cum"]
    df_s["peak"] = df_s["eq"].cummax()
    df_s["dd"]   = (df_s["eq"] - df_s["peak"]) / df_s["peak"] * 100
    df_s["dd_amt"]= df_s["eq"] - df_s["peak"]
    dd_idx       = df_s["dd"].idxmin()
    pk_idx       = df_s["eq"].idxmax()

    monthly = (df.assign(month=pd.to_datetime(df["entry_date"]).dt.to_period("M"))
                 .groupby("month")["net_pnl"].sum().reset_index())
    yearly  = (df.assign(year=pd.to_datetime(df["entry_date"]).dt.year)
                 .groupby("year")["net_pnl"].sum())

    # Equity curve for chart
    df_s["symbol"]     = df_s["symbol"]
    df_s["entry_date"] = df_s["entry_date"]
    df_s["return_pct"] = df_s["return_pct"]
    df_s["peak_equity"]= df_s["peak"]

    return {
        "total_trades":         len(df),
        "winning_trades":       len(wins),
        "losing_trades":        len(loss),
        "win_rate_pct":         round(len(wins) / len(df) * 100, 1),
        "profit_factor":        round(gp / gl, 2) if gl > 0 else 999,
        "sharpe_ratio":         sharpe,
        "total_gross_pnl":      round(df["gross_pnl"].sum(), 2),
        "total_brokerage":      round(df["brokerage"].sum(), 2),
        "total_stt":            round(df["stt"].sum(), 2),
        "total_exchange":       round(df["exchange"].sum(), 2),
        "total_sebi":           round(df["sebi"].sum(), 2),
        "total_stamp":          round(df["stamp"].sum(), 2),
        "total_gst":            round(df["gst"].sum(), 2),
        "total_charges":        round(df["total_charges"].sum(), 2),
        "total_tax":            round(df["stcg_tax"].sum(), 2),
        "total_net_pnl":        round(df["net_pnl"].sum(), 2),
        "return_on_capital_pct":round(df["net_pnl"].sum() / cap * 100, 2),
        "avg_win":              round(wins["net_pnl"].mean(), 2) if len(wins) else 0,
        "avg_loss":             round(loss["net_pnl"].mean(), 2) if len(loss) else 0,
        "largest_win":          round(df["net_pnl"].max(), 2),
        "largest_loss":         round(df["net_pnl"].min(), 2),
        "avg_hold_days":        round(df["hold_days"].mean(), 1),
        "ltcg_trades":          len(ltcg),
        "ltcg_pct":             round(len(ltcg) / len(df) * 100, 1),
        "ltcg_tax_paid":        round(ltcg["stcg_tax"].sum(), 2),
        "stcg_trades":          len(stcg),
        "stcg_tax_paid":        round(stcg["stcg_tax"].sum(), 2),
        "max_drawdown_pct":     round(df_s["dd"].min(), 2),
        "max_drawdown_amt":     round(df_s["dd_amt"].min(), 2),
        "max_drawdown_date":    str(df_s.loc[dd_idx, "exit_date"]),
        "peak_equity_date":     str(df_s.loc[pk_idx, "exit_date"]),
        "equity_curve":         df_s[["entry_date","exit_date","eq","peak_equity",
                                      "dd","symbol","net_pnl","return_pct"]].to_dict("records"),
        "monthly_pnl":          [{"month": str(r["month"]), "pnl": round(r["net_pnl"], 2)}
                                  for _, r in monthly.iterrows()],
        "yearly_pnl":           {str(yr): round(p, 2) for yr, p in yearly.items()},
        "exit_reasons":         df["exit_reason"].value_counts().to_dict(),
        "all_trades":           df.to_dict("records"),
    }


# ─── HTML REPORT ──────────────────────────────────────────────────────────────

def generate_report(stats: dict, cfg: dict, report_dir: Path):
    if not stats:
        return

    cap       = cfg["capital"]["initial_capital"]
    net       = stats["total_net_pnl"]
    gross     = stats["total_gross_pnl"]

    # Trade rows
    trade_rows = ""
    for t in sorted(stats["all_trades"], key=lambda x: x["entry_date"]):
        c  = "#1D9E75" if t["net_pnl"] > 0 else "#D85A30"
        tc = "#1D9E75" if t["tax_type"] == "LTCG" else "#BA7517"
        trade_rows += f"""<tr>
          <td>{t['symbol']}</td>
          <td>{t['entry_date']}</td><td>{t['exit_date']}</td>
          <td style="text-align:right">₹{t['entry_price']:,.2f}</td>
          <td style="text-align:right">₹{t['stop_loss']:,.2f}</td>
          <td style="text-align:right">₹{t['target']:,.2f}</td>
          <td style="text-align:right">₹{t['exit_price']:,.2f}</td>
          <td style="text-align:right">{t['qty']}</td>
          <td style="text-align:right">₹{t['buy_value']:,.0f}</td>
          <td style="text-align:right">₹{t['gross_pnl']:,.0f}</td>
          <td style="text-align:right">₹{t['total_charges']:,.0f}</td>
          <td style="text-align:right;color:{tc}">₹{t['stcg_tax']:,.0f} ({t['tax_type']})</td>
          <td style="text-align:right;font-weight:700;color:{c}">₹{t['net_pnl']:,.0f}</td>
          <td style="text-align:right;color:{c}">{t['return_pct']}%</td>
          <td style="text-align:right">{t['rr_achieved']}</td>
          <td style="text-align:right">{t['hold_days']}d</td>
          <td>{t['exit_reason']}</td>
        </tr>"""

    # Equity curve data for chart
    eq_data    = json.dumps([{"x": t["exit_date"], "y": round(t["eq"], 0)} for t in stats["equity_curve"]])
    dd_data    = json.dumps([{"x": t["exit_date"], "y": round(t["dd"], 2)}  for t in stats["equity_curve"]])
    mon_labels = json.dumps([m["month"] for m in stats["monthly_pnl"]])
    mon_values = json.dumps([m["pnl"]   for m in stats["monthly_pnl"]])
    yr_labels  = json.dumps(list(stats["yearly_pnl"].keys()))
    yr_values  = json.dumps(list(stats["yearly_pnl"].values()))

    vcfg     = cfg.get("volatility_filter", {})
    vix_info = (f"VIX no-entry>{vcfg.get('vix_no_entry_above',20)} | "
                f"Nifty 50 EMA:{'ON' if vcfg.get('nifty_trend_filter') else 'OFF'} | "
                f"Nifty 200 EMA:{'ON' if vcfg.get('nifty_200ema_filter') else 'OFF'} | "
                f"ADX≥{vcfg.get('adx_min_threshold',25)}:{'ON' if vcfg.get('adx_filter') else 'OFF'}")

    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><title>LTCG Backtest Report</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#07090f;color:#b8cfe0;font-family:'Courier New',monospace;font-size:13px}}
.hdr{{background:linear-gradient(135deg,#0b1520,#0f1e2e);border-bottom:1px solid #1a2d42;padding:20px 28px}}
.hdr h1{{color:#6db3f2;font-size:18px;margin-bottom:4px}}
.meta{{color:#3d6080;font-size:10px;letter-spacing:1px;line-height:1.9}}
.body{{max-width:1600px;margin:0 auto;padding:20px 16px 60px}}
.sec{{margin-bottom:28px}}
.sec-t{{font-size:10px;color:#3d7aaa;letter-spacing:3px;text-transform:uppercase;
        margin-bottom:10px;padding-bottom:5px;border-bottom:1px solid #132030}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:9px}}
.card{{background:#0c1520;border:1px solid #162535;border-radius:8px;padding:12px 14px}}
.cl{{font-size:10px;color:#3d6080;margin-bottom:4px}}
.cv{{font-size:17px;font-weight:700}}
.cs{{font-size:10px;color:#3d6080;margin-top:3px}}
.g{{color:#1D9E75}}.r{{color:#D85A30}}.a{{color:#BA7517}}.w{{color:#e0eaf5}}
.tw{{overflow-x:auto;border-radius:8px;border:1px solid #162535}}
table{{width:100%;border-collapse:collapse;font-size:10px}}
th{{background:#0c1825;color:#3d7aaa;padding:7px 9px;text-align:left;
    border-bottom:1px solid #1a2d42;white-space:nowrap}}
td{{padding:5px 9px;border-bottom:1px solid #0f1c28;white-space:nowrap}}
tbody tr:hover td{{background:#0f1e2e}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
.chart-box{{background:#0c1520;border:1px solid #162535;border-radius:8px;padding:14px}}
.chart-t{{font-size:10px;color:#3d7aaa;letter-spacing:2px;text-transform:uppercase;margin-bottom:10px}}
@media(max-width:700px){{.two{{grid-template-columns:1fr}}}}
</style></head><body>
<div class="hdr">
  <h1>📊 LTCG Positional Backtest</h1>
  <div class="meta">
    {cfg['backtest']['start_date']} → {cfg['backtest']['end_date']} &nbsp;·&nbsp;
    Capital ₹{cap:,} &nbsp;·&nbsp;
    RR {cfg['trade_rules']['reward_to_risk_ratio']}:1 &nbsp;·&nbsp;
    Max hold {cfg['backtest']['max_hold_days']}d &nbsp;·&nbsp;
    LTCG protect {'ON' if cfg['exit_rules'].get('ltcg_protect') else 'OFF'} &nbsp;·&nbsp;
    {vix_info} &nbsp;·&nbsp;
    Generated {datetime.now().strftime('%d %b %Y %H:%M')}
  </div>
</div>
<div class="body">

<div class="sec">
  <div class="sec-t">Performance Summary</div>
  <div class="cards">
    <div class="card"><div class="cl">Total Trades</div>
      <div class="cv w">{stats['total_trades']}</div>
      <div class="cs">{stats['winning_trades']}W / {stats['losing_trades']}L</div></div>
    <div class="card"><div class="cl">Win Rate</div>
      <div class="cv {'g' if stats['win_rate_pct']>=55 else 'a'}">{stats['win_rate_pct']}%</div></div>
    <div class="card"><div class="cl">Profit Factor</div>
      <div class="cv {'g' if stats['profit_factor']>=1.5 else 'a'}">{stats['profit_factor']}</div>
      <div class="cs">Target ≥ 1.5</div></div>
    <div class="card"><div class="cl">Sharpe Ratio</div>
      <div class="cv {'g' if stats['sharpe_ratio']>=2 else 'a'}">{stats['sharpe_ratio']}</div>
      <div class="cs">Target ≥ 2.0</div></div>
    <div class="card"><div class="cl">Gross P&L</div>
      <div class="cv w">₹{stats['total_gross_pnl']:,.0f}</div></div>
    <div class="card"><div class="cl">Total Charges</div>
      <div class="cv r">₹{stats['total_charges']:,.0f}</div>
      <div class="cs">{stats['total_charges']/gross*100:.1f}% of gross</div></div>
    <div class="card"><div class="cl">Tax Paid</div>
      <div class="cv r">₹{stats['total_tax']:,.0f}</div>
      <div class="cs">{stats['total_tax']/gross*100:.1f}% of gross</div></div>
    <div class="card"><div class="cl">Net P&L</div>
      <div class="cv {'g' if net>0 else 'r'}">₹{net:,.0f}</div>
      <div class="cs">After all deductions</div></div>
    <div class="card"><div class="cl">Return on Capital</div>
      <div class="cv {'g' if stats['return_on_capital_pct']>0 else 'r'}">{stats['return_on_capital_pct']}%</div>
      <div class="cs">on ₹{cap:,}</div></div>
    <div class="card"><div class="cl">Max Drawdown</div>
      <div class="cv {'g' if stats['max_drawdown_pct']>-15 else 'r'}">{stats['max_drawdown_pct']}%</div>
      <div class="cs">{stats['max_drawdown_date']}</div></div>
    <div class="card"><div class="cl">Avg Hold</div>
      <div class="cv w">{stats['avg_hold_days']}d</div></div>
    <div class="card"><div class="cl">LTCG Trades</div>
      <div class="cv g">{stats['ltcg_pct']}%</div>
      <div class="cs">{stats['ltcg_trades']} of {stats['total_trades']}</div></div>
  </div>
</div>

<div class="sec">
  <div class="sec-t">LTCG vs STCG Tax Breakdown</div>
  <div class="tw" style="max-width:420px">
    <table><tbody>
      <tr><td>LTCG trades (held ≥366d)</td><td style="text-align:right;color:#1D9E75">{stats['ltcg_trades']} trades | Tax ₹{stats['ltcg_tax_paid']:,.0f} @ 10%</td></tr>
      <tr><td>STCG trades (held &lt;366d)</td><td style="text-align:right;color:#BA7517">{stats['stcg_trades']} trades | Tax ₹{stats['stcg_tax_paid']:,.0f} @ 15%</td></tr>
      <tr><td>Total tax</td><td style="text-align:right;color:#D85A30;font-weight:700">₹{stats['total_tax']:,.0f} ({stats['total_tax']/gross*100:.1f}% of gross)</td></tr>
    </tbody></table>
  </div>
</div>

<div class="sec">
  <div class="sec-t">Equity Curve & Drawdown</div>
  <div class="two">
    <div class="chart-box"><div class="chart-t">Portfolio Value</div>
      <div style="position:relative;height:200px">
        <canvas id="eqChart"></canvas></div></div>
    <div class="chart-box"><div class="chart-t">Drawdown %</div>
      <div style="position:relative;height:200px">
        <canvas id="ddChart"></canvas></div></div>
  </div>
</div>

<div class="sec">
  <div class="sec-t">Yearly & Monthly P&L</div>
  <div class="two">
    <div class="chart-box"><div class="chart-t">Yearly Net P&L</div>
      <div style="position:relative;height:200px">
        <canvas id="yrChart"></canvas></div></div>
    <div class="chart-box"><div class="chart-t">Monthly Net P&L</div>
      <div style="position:relative;height:200px">
        <canvas id="monChart"></canvas></div></div>
  </div>
</div>

<div class="sec">
  <div class="sec-t">Complete Trade Log — {stats['total_trades']} Trades</div>
  <div class="tw"><table><thead><tr>
    <th>Symbol</th><th>Entry</th><th>Exit</th>
    <th style="text-align:right">Entry₹</th><th style="text-align:right">SL₹</th>
    <th style="text-align:right">Target₹</th><th style="text-align:right">Exit₹</th>
    <th style="text-align:right">Qty</th><th style="text-align:right">BuyVal</th>
    <th style="text-align:right">Gross</th><th style="text-align:right">Charges</th>
    <th style="text-align:right">Tax</th><th style="text-align:right">NetP&L</th>
    <th style="text-align:right">Ret%</th><th style="text-align:right">RR</th>
    <th style="text-align:right">Hold</th><th>Exit Reason</th>
  </tr></thead><tbody>{trade_rows}</tbody></table></div>
</div>

</div>
<script>
const gc='rgba(255,255,255,0.06)';const tc='#3d6080';
const base={{responsive:true,maintainAspectRatio:false,
  plugins:{{legend:{{display:false}},
    tooltip:{{backgroundColor:'#0c1520',borderColor:'#1a2d42',borderWidth:1,
              titleColor:'#6db3f2',bodyColor:'#b8cfe0'}}}}}};

// Equity curve
const eq={eq_data};
new Chart(document.getElementById('eqChart'),{{type:'line',
  data:{{labels:eq.map(d=>d.x),datasets:[{{data:eq.map(d=>d.y),
    borderColor:'#378ADD',backgroundColor:'rgba(55,138,221,0.06)',
    fill:true,tension:0.3,pointRadius:0}}]}},
  options:{{...base,scales:{{
    x:{{ticks:{{color:tc,maxTicksLimit:8,font:{{size:9}}}},grid:{{color:gc}}}},
    y:{{ticks:{{color:tc,callback:v=>'₹'+(v/1000).toFixed(0)+'K',font:{{size:9}}}},grid:{{color:gc}}}}
  }}}}}});

// Drawdown
const dd={dd_data};
new Chart(document.getElementById('ddChart'),{{type:'line',
  data:{{labels:dd.map(d=>d.x),datasets:[{{data:dd.map(d=>d.y),
    borderColor:'#D85A30',backgroundColor:'rgba(216,90,48,0.08)',
    fill:true,tension:0.3,pointRadius:0}}]}},
  options:{{...base,scales:{{
    x:{{ticks:{{color:tc,maxTicksLimit:8,font:{{size:9}}}},grid:{{color:gc}}}},
    y:{{ticks:{{color:tc,callback:v=>v+'%',font:{{size:9}}}},grid:{{color:gc}}}}
  }}}}}});

// Yearly
const yrL={yr_labels}; const yrV={yr_values};
new Chart(document.getElementById('yrChart'),{{type:'bar',
  data:{{labels:yrL,datasets:[{{data:yrV,
    backgroundColor:yrV.map(v=>v>=0?'rgba(29,158,117,0.8)':'rgba(216,90,48,0.8)'),
    borderRadius:4}}]}},
  options:{{...base,scales:{{
    x:{{ticks:{{color:tc,font:{{size:10}}}},grid:{{color:gc}}}},
    y:{{ticks:{{color:tc,callback:v=>'₹'+(v/1000).toFixed(0)+'K',font:{{size:9}}}},grid:{{color:gc}}}}
  }}}}}});

// Monthly
const mL={mon_labels}; const mV={mon_values};
new Chart(document.getElementById('monChart'),{{type:'bar',
  data:{{labels:mL,datasets:[{{data:mV,
    backgroundColor:mV.map(v=>v>=0?'rgba(29,158,117,0.7)':'rgba(216,90,48,0.7)'),
    borderRadius:3}}]}},
  options:{{...base,scales:{{
    x:{{ticks:{{color:tc,maxRotation:45,font:{{size:8}}}},grid:{{color:gc}}}},
    y:{{ticks:{{color:tc,callback:v=>'₹'+(v/1000).toFixed(0)+'K',font:{{size:9}}}},grid:{{color:gc}}}}
  }}}}}});
</script></body></html>"""

    out = report_dir / "backtest_report.html"
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  📊 Report    → {out}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LTCG Positional Backtest Engine")
    parser.add_argument("--symbol", type=str,   help="Run for single symbol e.g. RELIANCE")
    parser.add_argument("--config", default="config_ltcg.json")
    args = parser.parse_args()

    cfg        = load_config(args.config)
    wl_data    = load_watchlist(cfg)
    watchlist  = wl_data["stocks"]
    data_dir   = get_data_dir(cfg)
    report_dir = get_report_dir(cfg)
    interval   = cfg["backtest"]["interval"]
    start_date = cfg["backtest"]["start_date"]
    end_date   = cfg["backtest"]["end_date"]
    top_n      = cfg["stock_selection"]["ranking"]["top_n_stocks"]
    min_score  = cfg["stock_selection"]["ranking"]["min_rank_score"]
    max_hold   = cfg["backtest"].get("max_hold_days", 400)
    cap        = cfg["capital"]["initial_capital"]
    max_pos    = cfg["capital"].get("max_position_pct", 30)
    max_trades = cfg["capital"].get("max_open_trades", 5)

    # Position sizing explanation at startup
    risk_amt   = cap * cfg["capital"]["risk_per_trade_pct"] / 100
    max_buy    = cap * max_pos / 100

    print(f"\n⚡ LTCG Positional Backtester")
    print(f"   Period      : {start_date} → {end_date}")
    print(f"   Data dir    : {data_dir}/")
    print(f"   Report dir  : {report_dir}/")
    print(f"   Capital     : ₹{cap:,}")
    print(f"   R:R         : {cfg['trade_rules']['reward_to_risk_ratio']}:1")
    print(f"   Max hold    : {max_hold} days  |  LTCG protect: {cfg['exit_rules'].get('ltcg_protect','OFF')}")
    print(f"   Max trades  : {max_trades}  |  Max position: {max_pos}% = ₹{max_buy:,.0f}/trade")
    print(f"   Risk/trade  : {cfg['capital']['risk_per_trade_pct']}% = ₹{risk_amt:,.0f} max loss if SL hits")
    print(f"   Top N       : {top_n} stocks/day\n")
    print(f"   Position sizing logic:")
    print(f"     qty = min(risk-based qty, position-cap qty)")
    print(f"     risk-based : ₹{risk_amt:,.0f} ÷ (entry - SL)")
    print(f"     position-cap: ₹{max_buy:,.0f} ÷ entry_price")
    print(f"     → whichever gives FEWER shares wins\n")

    # Load with warmup
    warmup_days  = cfg["backtest"].get("warmup_days", 250)
    warmup_start = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=warmup_days)).strftime("%Y-%m-%d")
    start_date_d = pd.to_datetime(start_date).date()
    end_date_d   = pd.to_datetime(end_date).date()
    warmup_d     = pd.to_datetime(warmup_start).date()

    blacklist = set(cfg["stock_selection"].get("blacklist", []))
    symbols   = [args.symbol.upper()] if args.symbol else [s["symbol"] for s in watchlist]
    symbols   = [s for s in symbols if s not in blacklist]

    all_data = {}
    for sym in symbols:
        fp = data_dir / f"{sym}_{interval}.csv"
        if not fp.exists():
            continue
        try:
            df = pd.read_csv(fp)
            df["date"] = pd.to_datetime(df["date"]).dt.date
            df = df[(df["date"] >= warmup_d) & (df["date"] <= end_date_d)].reset_index(drop=True)
            if len(df) >= 60:
                df = add_indicators(df, cfg)
                df["in_trading_period"] = df["date"] >= start_date_d
                all_data[sym] = df
        except Exception as e:
            print(f"  ⚠ {sym}: {e}")

    if not all_data:
        print("❌ No data found. Run zerodha_downloader.py first.")
        return
    print(f"  Loaded {len(all_data)} stocks ({warmup_start} → {end_date}, {warmup_days}d warmup)\n")

    # Load Nifty 50
    nifty_path = data_dir / f"NIFTY50_{interval}.csv"
    if nifty_path.exists():
        nifty_df = pd.read_csv(nifty_path)
        nifty_df["date"] = pd.to_datetime(nifty_df["date"]).dt.date
        nifty_df = nifty_df[nifty_df["date"] >= warmup_d].reset_index(drop=True)
        print(f"  ✓ Nifty 50 loaded (incl. warmup)")
    else:
        nifty_df = next(iter(all_data.values()))[["date","close"]].copy()
        print("  ⚠ NIFTY50 not found — using proxy")

    # Load India VIX
    vix_path = data_dir / f"INDIAVIX_{interval}.csv"
    if vix_path.exists():
        vix_df = pd.read_csv(vix_path)
        vix_df["date"] = pd.to_datetime(vix_df["date"]).dt.date
        vix_df = vix_df[vix_df["date"] >= warmup_d].reset_index(drop=True)
        print(f"  ✓ India VIX loaded")
    else:
        vix_df = None
        print("  ⚠ INDIAVIX not found — VIX filter disabled")

    vix_map     = build_vix_map(vix_df, cfg) if vix_df is not None else {}
    nifty_trend = build_nifty_trend_map(nifty_df, cfg)

    vcfg = cfg.get("volatility_filter", {})
    if vcfg.get("enabled") and vix_map:
        no_e = sum(1 for v,r in vix_map.values() if r=="no_entry")
        caut = sum(1 for v,r in vix_map.values() if r=="caution")
        print(f"  VIX filter : {no_e} days blocked | {caut} caution days")
    if nifty_trend:
        b50  = sum(1 for v in nifty_trend.values() if not v.get("ema50_ok",  True))
        b200 = sum(1 for v in nifty_trend.values() if not v.get("ema200_ok", True))
        badx = sum(1 for v in nifty_trend.values() if not v.get("trend_ok",  True))
        print(f"  Nifty 50 EMA : {b50} days blocked")
        print(f"  Nifty 200 EMA: {b200} days blocked (bear market)")
        print(f"  ADX filter   : {badx} total days no new entries")
    print()

    # Daily ranking
    print("  📊 Running daily ranking...")
    all_dates = sorted(set(
        d for df in all_data.values()
        for d in df[df["in_trading_period"]]["date"].tolist()
    ))
    daily_rankings = []
    eligible       = {sym: set() for sym in all_data}

    for dt in all_dates:
        ranked = rank_stocks_on_date(all_data, dt, nifty_df, cfg, wl_data)
        if ranked.empty:
            continue
        eligible_today = ranked[ranked["rank_score"] >= min_score].head(top_n)
        for sym in eligible_today["symbol"]:
            if sym in eligible:
                eligible[sym].add(dt)
        daily_rankings.extend(eligible_today.to_dict("records"))

    if cfg["output"]["save_daily_rankings"] and daily_rankings:
        rp = report_dir / "daily_rankings.csv"
        pd.DataFrame(daily_rankings).to_csv(rp, index=False)
        print(f"  📋 Rankings  → {rp}")

    # Backtest
    print("\n  Running backtest...\n")
    all_trades = []
    for sym, df in all_data.items():
        trades = backtest_symbol(df, sym, cfg, eligible[sym], vix_map, nifty_trend)
        all_trades.extend(trades)
        if trades:
            w   = sum(1 for t in trades if t["net_pnl"] > 0)
            pnl = sum(t["net_pnl"] for t in trades)
            print(f"  ✓ {sym:15s} {len(trades):3d} trades | Win {w/len(trades)*100:.0f}% | P&L ₹{pnl:,.0f}")

    if not all_trades:
        print("\n⚠ No trades generated.")
        return

    stats = compute_stats(all_trades, cfg)

    print(f"\n{'─'*54}")
    print(f"  TOTAL TRADES      : {stats['total_trades']}")
    print(f"  WIN RATE          : {stats['win_rate_pct']}%")
    print(f"  PROFIT FACTOR     : {stats['profit_factor']}")
    print(f"  SHARPE RATIO      : {stats['sharpe_ratio']}")
    print(f"  GROSS P&L         : ₹{stats['total_gross_pnl']:,.2f}")
    print(f"  TOTAL CHARGES     : ₹{stats['total_charges']:,.2f}")
    print(f"  STCG TAX          : ₹{stats['total_tax']:,.2f}")
    print(f"  LTCG trades       : {stats['ltcg_pct']}% ({stats['ltcg_trades']} trades @ 10%)")
    print(f"  STCG trades       : {100-stats['ltcg_pct']}% ({stats['stcg_trades']} trades @ 15%)")
    print(f"  NET P&L           : ₹{stats['total_net_pnl']:,.2f}")
    print(f"  RETURN ON CAPITAL : {stats['return_on_capital_pct']}%")
    print(f"  MAX DRAWDOWN      : {stats['max_drawdown_pct']}% on {stats['max_drawdown_date']}")
    print(f"{'─'*54}\n")

    if cfg["output"]["save_trade_log"]:
        cols = ["symbol","entry_date","exit_date","entry_price","stop_loss","target",
                "exit_price","qty","buy_value","sell_value","gross_pnl",
                "brokerage","stt","exchange","sebi","stamp","gst","total_charges",
                "net_before_tax","tax_type","stcg_tax","net_pnl","return_pct",
                "rr_achieved","exit_reason","hold_days","is_ltcg","result"]
        tp = report_dir / "trade_log.csv"
        pd.DataFrame(all_trades)[cols].sort_values("entry_date").to_csv(tp, index=False)
        print(f"  📋 Trade log → {tp}")

    if cfg["output"]["save_report"]:
        generate_report(stats, cfg, report_dir)


if __name__ == "__main__":
    main()
