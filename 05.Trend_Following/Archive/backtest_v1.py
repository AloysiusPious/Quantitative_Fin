"""
Swing Trade Backtesting Engine
================================
Directory structure (auto-created):
  data/START_END/          ← all CSV files (stocks + indices)
  reports/START_END/       ← backtest_report.html, trade_log.csv, daily_rankings.csv

New features vs previous version:
  ✓ Dated data & report directories
  ✓ max_hold_days — force exit after N days (fixes 300-day stuck trades)
  ✓ India VIX filter — block entries when VIX > threshold
  ✓ Nifty 50 trend filter — no longs when Nifty below 50 EMA
  ✓ VIX caution zone — raise minimum rank score during elevated VIX

USAGE:
    python backtest.py
    python backtest.py --symbol RELIANCE
    python backtest.py --config myconfig.json
"""

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


# ─── CONFIG & PATHS ───────────────────────────────────────────────────────────

def load_config(path="config.json") -> dict:
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
    s = cfg["strategy"]
    df = df.copy()
    df["ema_fast"]  = df["close"].ewm(span=s["ema_fast"],  adjust=False).mean()
    df["ema_slow"]  = df["close"].ewm(span=s["ema_slow"],  adjust=False).mean()
    df["vol_avg"]   = df["volume"].rolling(s["volume_avg_period"]).mean()
    df["swing_low"] = df["low"].rolling(cfg["exit_rules"]["swing_low_candles"]).min()
    df["tr"]        = np.maximum(df["high"] - df["low"],
                      np.maximum(abs(df["high"] - df["close"].shift(1)),
                                 abs(df["low"]  - df["close"].shift(1))))
    df["atr"]       = df["tr"].rolling(14).mean()
    mom             = cfg["stock_selection"]["ranking"]["momentum_days"]
    df["momentum"]  = df["close"].pct_change(mom)

    df["in_uptrend"]     = (df["close"] > df["ema_fast"]) & (df["close"] > df["ema_slow"])
    tol = s["pullback_tolerance_pct"] / 100
    df["pullback"]       = (df["low"] <= df["ema_fast"] * (1 + tol)) & (df["close"] > df["ema_fast"])
    body = abs(df["close"] - df["open"])
    rng  = df["high"] - df["low"]
    df["bullish_candle"] = (df["close"] > df["open"]) & (body >= s["bullish_body_pct"] / 100 * rng)
    df["vol_confirmed"]  = df["volume"] > df["vol_avg"] * s["volume_multiplier"]
    df["signal"]         = df["in_uptrend"] & df["pullback"] & df["bullish_candle"] & df["vol_confirmed"]
    return df


# ─── VIX FILTER ───────────────────────────────────────────────────────────────

def build_vix_map(vix_df: pd.DataFrame, cfg: dict) -> dict:
    """
    Returns dict: date → (vix_avg, regime)
    regime: 'normal' | 'caution' | 'no_entry'
    """
    vcfg     = cfg.get("volatility_filter", {})
    if not vcfg.get("enabled", False) or vix_df is None or vix_df.empty:
        return {}

    no_entry = vcfg.get("vix_no_entry_above", 20.0)
    caution  = vcfg.get("vix_caution_above",  16.0)
    lookback = vcfg.get("vix_lookback_days",  5)

    vix_df   = vix_df.sort_values("date").copy()
    vix_df["vix_avg"] = vix_df["close"].rolling(lookback, min_periods=1).mean()

    result = {}
    for _, row in vix_df.iterrows():
        v = row["vix_avg"]
        if v >= no_entry:
            regime = "no_entry"
        elif v >= caution:
            regime = "caution"
        else:
            regime = "normal"
        result[row["date"]] = (round(v, 2), regime)
    return result


def build_nifty_trend_map(nifty_df: pd.DataFrame, cfg: dict) -> dict:
    """
    Returns dict: date → True if ALL active Nifty trend filters pass.

    Filters (each independently configurable):
      1. nifty_trend_filter   — Nifty close > 50 EMA  (medium-term trend)
      2. nifty_200ema_filter  — Nifty close > 200 EMA (long-term bull market)
      3. adx_filter           — Nifty ADX >= threshold (trend is strong, not choppy)

    All enabled filters must pass for entry to be allowed.
    """
    vcfg = cfg.get("volatility_filter", {})
    if nifty_df is None or nifty_df.empty:
        return {}

    df = nifty_df.copy().sort_values("date").reset_index(drop=True)

    # ── 50 EMA ──
    df["ema50"]  = df["close"].ewm(span=50,  adjust=False).mean()

    # ── 200 EMA ──
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    # ── ADX (Wilder's Average Directional Index) ──
    # Measures trend STRENGTH (0-100). >25 = strong trend. <20 = choppy/sideways.
    adx_period = vcfg.get("adx_period", 14)
    high = df["high"]; low = df["low"]; close = df["close"]

    # True Range
    df["tr"] = np.maximum(high - low,
               np.maximum(abs(high - close.shift(1)),
                          abs(low  - close.shift(1))))

    # Directional Movement
    df["dm_plus"]  = np.where((high - high.shift(1)) > (low.shift(1) - low),
                              np.maximum(high - high.shift(1), 0), 0)
    df["dm_minus"] = np.where((low.shift(1) - low) > (high - high.shift(1)),
                              np.maximum(low.shift(1) - low, 0), 0)

    # Wilder smoothing (RMA)
    def wilder_smooth(series, period):
        result = series.copy() * 0.0
        result.iloc[period] = series.iloc[1:period+1].sum()
        for i in range(period + 1, len(series)):
            result.iloc[i] = result.iloc[i-1] - (result.iloc[i-1] / period) + series.iloc[i]
        return result

    atr_w   = wilder_smooth(df["tr"],       adx_period)
    dmp_w   = wilder_smooth(df["dm_plus"],  adx_period)
    dmm_w   = wilder_smooth(df["dm_minus"], adx_period)

    df["di_plus"]  = (dmp_w / atr_w.replace(0, np.nan)) * 100
    df["di_minus"] = (dmm_w / atr_w.replace(0, np.nan)) * 100
    di_sum  = df["di_plus"] + df["di_minus"]
    df["dx"] = (abs(df["di_plus"] - df["di_minus"]) / di_sum.replace(0, np.nan)) * 100
    df["adx"] = wilder_smooth(df["dx"], adx_period)

    # ── Build result map ──
    adx_min     = vcfg.get("adx_min_threshold",    25)
    adx_caution = vcfg.get("adx_caution_threshold", 20)
    adx_caution_score = vcfg.get("adx_caution_min_score", 0.65)

    result = {}
    for _, row in df.iterrows():
        checks = []

        # 50 EMA filter
        if vcfg.get("nifty_trend_filter", False):
            checks.append(row["close"] > row["ema50"])

        # 200 EMA filter — strong bull market gate
        if vcfg.get("nifty_200ema_filter", False):
            checks.append(row["close"] > row["ema200"])

        # ADX filter — trend strength gate
        adx_val = row.get("adx", np.nan)
        if vcfg.get("adx_filter", False) and not np.isnan(adx_val):
            checks.append(adx_val >= adx_min)

        # All checks must pass
        trend_ok = all(checks) if checks else True

        # Store extra info for caution zone logic in backtest_symbol
        result[row["date"]] = {
            "trend_ok":   trend_ok,
            "adx":        round(float(adx_val), 2) if not np.isnan(adx_val) else 0,
            "ema50_ok":   bool(row["close"] > row["ema50"]),
            "ema200_ok":  bool(row["close"] > row["ema200"]),
            "adx_caution": vcfg.get("adx_filter", False) and
                           not np.isnan(adx_val) and
                           adx_caution <= adx_val < adx_min,
            "adx_caution_score": adx_caution_score,
        }
    return result


# ─── RANKING ──────────────────────────────────────────────────────────────────

def compute_rank_score(row: dict, nifty_return: float, cfg: dict) -> float:
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
    rk      = cfg["stock_selection"]["ranking"]
    rs_days = rk["rs_lookback_days"]

    nifty_past   = nifty_df[nifty_df["date"] <= target_date].tail(rs_days + 1)
    nifty_return = 0.0
    if len(nifty_past) >= 2:
        nifty_return = (nifty_past["close"].iloc[-1] - nifty_past["close"].iloc[0]) / nifty_past["close"].iloc[0]

    sym_sector = {s: sec for sec, syms in wl_meta.get("sector_groups", {}).items() for s in syms}
    records    = []

    for symbol, df in all_data.items():
        dft = df[df["date"] <= target_date]
        if len(dft) < rs_days + 5:
            continue
        latest = dft.iloc[-1]
        past   = dft.tail(rs_days + 1)
        ema_gap   = (latest["close"] - latest["ema_fast"]) / latest["ema_fast"] if latest["ema_fast"] > 0 else 0
        rs_return = (past["close"].iloc[-1] - past["close"].iloc[0]) / past["close"].iloc[0] if past["close"].iloc[0] > 0 else 0
        vol_ratio = latest["volume"] / latest["vol_avg"] if latest["vol_avg"] > 0 else 1
        atr_pct   = latest["atr"] / latest["close"] if latest["close"] > 0 else 0.02
        score = compute_rank_score({"ema_fast_gap": ema_gap, "rs_return": rs_return,
                                    "vol_ratio": vol_ratio, "momentum": latest.get("momentum", 0),
                                    "atr_pct": atr_pct}, nifty_return, cfg)
        records.append({"symbol": symbol, "date": target_date,
                        "sector": sym_sector.get(symbol, "Unknown"),
                        "rank_score": score, "signal": bool(latest.get("signal", False))})

    if not records:
        return pd.DataFrame()

    ranked = pd.DataFrame(records).sort_values("rank_score", ascending=False)
    sector_count, filtered = {}, []
    for _, r in ranked.iterrows():
        sec = r["sector"]
        if sector_count.get(sec, 0) < rk["sector_limit"]:
            filtered.append(r)
            sector_count[sec] = sector_count.get(sec, 0) + 1
    return pd.DataFrame(filtered)


# ─── CHARGES & TAX ────────────────────────────────────────────────────────────

def calc_charges(buy_val: float, sell_val: float, cfg: dict) -> dict:
    tc  = cfg["taxes_and_charges"]
    brk = tc["brokerage_per_order"] * 2
    stt = sell_val  * tc["stt_pct"]            / 100
    exc = (buy_val + sell_val) * tc["exchange_charges_pct"] / 100
    sbi = (buy_val + sell_val) * tc["sebi_charges_pct"]    / 100
    stm = buy_val  * tc["stamp_duty_pct"]      / 100
    gst = (brk + exc) * tc["gst_pct"]          / 100
    return {"brokerage": round(brk,2), "stt": round(stt,2), "exchange": round(exc,2),
            "sebi": round(sbi,2), "stamp": round(stm,2), "gst": round(gst,2),
            "total": round(brk+stt+exc+sbi+stm+gst, 2)}

def calc_tax(net: float, cfg: dict) -> float:
    return round(max(0, net) * cfg["taxes_and_charges"]["stcg_tax_pct"] / 100, 2)

def get_sl(row, entry: float, cfg: dict) -> float:
    sl_type = cfg["exit_rules"]["stop_loss_type"]
    buf     = cfg["exit_rules"]["swing_low_buffer_pct"] / 100
    if sl_type == "swing_low":
        sl = row["swing_low"] * (1 - buf)
    elif sl_type == "atr_based":
        sl = entry - row["atr"] * cfg["exit_rules"]["stop_loss_atr_multiplier"]
    else:
        sl = entry * (1 - cfg["exit_rules"]["stop_loss_fixed_pct"] / 100)
    return round(max(sl, entry * 0.85), 2)


# ─── BACKTEST ONE SYMBOL ──────────────────────────────────────────────────────

def backtest_symbol(df: pd.DataFrame, symbol: str, cfg: dict,
                    eligible_dates: set, vix_map: dict, nifty_trend: dict) -> list:
    trades        = []
    rr            = cfg["trade_rules"]["reward_to_risk_ratio"]
    gap_up        = cfg["trade_rules"]["gap_filter_up_pct"]   / 100
    gap_dn        = cfg["trade_rules"]["gap_filter_down_pct"] / 100
    close_last    = cfg["backtest"]["close_all_on_last_day"]
    max_hold      = cfg["backtest"].get("max_hold_days", 9999)
    capital       = cfg["capital"]["initial_capital"]
    vcfg          = cfg.get("volatility_filter", {})
    caution_score = vcfg.get("vix_caution_min_score", 0.70)

    ex           = cfg["exit_rules"]
    do_partial   = ex.get("partial_exit", False)
    partial_pct  = ex.get("partial_exit_pct", 50) / 100
    partial_rr   = ex.get("partial_exit_at_rr", 1.0)
    move_be      = ex.get("move_sl_to_breakeven", False)
    do_trail     = ex.get("trailing_stop", False)
    trail_atr    = ex.get("trailing_stop_atr_multiplier", 1.0)

    blacklist = set(cfg["stock_selection"].get("blacklist", []))
    if symbol in blacklist:
        return []

    in_trade     = False
    partial_done = False
    entry_price  = sl = target = qty = remaining_qty = rank_score = 0
    entry_date   = None

    for i in range(1, len(df) - 1):
        row      = df.iloc[i]
        next_row = df.iloc[i + 1]
        is_last  = (i == len(df) - 2)

        # ── Skip warmup candles for new entries ──
        # We still process in_trade logic (exits) in case a trade
        # was somehow entered — but new entries only fire in trading period
        in_trading_period = bool(row.get("in_trading_period", True))

        if in_trade:
            hold_days = (pd.to_datetime(next_row["date"]) - pd.to_datetime(entry_date)).days

            # ATR trailing stop update
            if do_trail and partial_done:
                trail_sl = next_row["close"] - row["atr"] * trail_atr
                if trail_sl > sl:
                    sl = round(trail_sl, 2)

            # Force close
            if (is_last and close_last) or hold_days >= max_hold:
                reason = "forced_close" if (is_last and close_last) else "max_hold_exit"
                trades.append(_build(symbol, entry_date, next_row["date"],
                                     entry_price, next_row["open"], remaining_qty,
                                     sl, target, reason, cfg))
                in_trade = False; partial_done = False
                continue

            # Partial exit at 1:1 RR
            if do_partial and not partial_done:
                partial_trigger = entry_price + (entry_price - sl) * partial_rr
                if next_row["high"] >= partial_trigger:
                    partial_qty = max(1, int(remaining_qty * partial_pct))
                    trades.append(_build(symbol, entry_date, next_row["date"],
                                         entry_price, partial_trigger, partial_qty,
                                         sl, target, "partial_exit", cfg))
                    remaining_qty -= partial_qty
                    partial_done   = True
                    if move_be:
                        sl = entry_price
                    if remaining_qty <= 0:
                        in_trade = False; partial_done = False
                        continue

            # SL / Target hit
            exit_p = reason = None
            if next_row["low"] <= sl:
                exit_p, reason = sl, "stop_loss"
            elif next_row["high"] >= target:
                exit_p, reason = target, "target_hit"
            if exit_p and remaining_qty > 0:
                trades.append(_build(symbol, entry_date, next_row["date"],
                                     entry_price, exit_p, remaining_qty,
                                     sl, target, reason, cfg))
                in_trade = False; partial_done = False

        else:
            # No new entries during warmup period
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

            # ── Nifty trend filter (50 EMA + 200 EMA + ADX) ──
            if nifty_trend:
                info = nifty_trend.get(row["date"])
                if info:
                    if not info.get("trend_ok", True):
                        continue
                    # ADX caution zone — weak trend forming, require higher rank score
                    if info.get("adx_caution", False):
                        if rank_score < info.get("adx_caution_score", 0.65):
                            continue

            # ── Gap filter ──
            # Simulates real workflow: you scan after market close (row = signal day).
            # Next morning you check pre-market: if open gapped too much vs signal day
            # close → cancel order and skip. Gap = (next_open - signal_close) / signal_close
            gap = (next_row["open"] - row["close"]) / row["close"]
            if gap > gap_up or gap < -gap_dn:
                continue

            entry_price   = next_row["open"]
            sl            = get_sl(row, entry_price, cfg)
            if entry_price <= sl or (entry_price - sl) < 0.5:
                continue

            risk          = entry_price - sl
            target        = round(entry_price + risk * rr, 2)

            # ── Position sizing ──
            # Two constraints applied simultaneously:
            # 1. Risk constraint  — never lose more than risk_per_trade_pct of capital
            # 2. Position cap     — never deploy more than max_position_pct of capital
            # qty = min of both → whichever is the binding constraint
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


def _build(symbol, entry_date, exit_date, entry, exit_p, qty, sl, target, reason, cfg) -> dict:
    buy_val        = entry  * qty
    sell_val       = exit_p * qty
    gross          = sell_val - buy_val
    ch             = calc_charges(buy_val, sell_val, cfg)
    net_pre_tax    = gross - ch["total"]
    tax            = calc_tax(net_pre_tax, cfg)
    final          = net_pre_tax - tax
    hold_days      = (pd.to_datetime(exit_date) - pd.to_datetime(entry_date)).days
    risk           = entry - sl
    rr_actual      = round((exit_p - entry) / risk, 2) if risk > 0 else 0
    return {
        "symbol": symbol, "entry_date": str(entry_date), "exit_date": str(exit_date),
        "entry_price": round(entry,2), "stop_loss": round(sl,2), "target": round(target,2),
        "exit_price": round(exit_p,2), "qty": qty,
        "buy_value": round(buy_val,2), "sell_value": round(sell_val,2),
        "gross_pnl": round(gross,2),
        "brokerage": ch["brokerage"], "stt": ch["stt"], "exchange": ch["exchange"],
        "sebi": ch["sebi"], "stamp": ch["stamp"], "gst": ch["gst"],
        "total_charges": ch["total"], "net_before_tax": round(net_pre_tax,2),
        "stcg_tax": tax, "net_pnl": round(final,2),
        "return_pct": round((exit_p - entry) / entry * 100, 2),
        "rr_achieved": rr_actual, "exit_reason": reason, "hold_days": hold_days,
        "result": "WIN" if final > 0 else "LOSS",
    }


# ─── STATISTICS ───────────────────────────────────────────────────────────────

def compute_stats(trades: list, cfg: dict) -> dict:
    if not trades:
        return {}
    df   = pd.DataFrame(trades).sort_values("entry_date").reset_index(drop=True)
    wins = df[df["net_pnl"] > 0]
    loss = df[df["net_pnl"] <= 0]
    cap  = cfg["capital"]["initial_capital"]

    df["cumulative_pnl"] = df["net_pnl"].cumsum()
    df["equity"]         = cap + df["cumulative_pnl"]
    df["peak_equity"]    = df["equity"].cummax()
    df["drawdown_amt"]   = df["equity"] - df["peak_equity"]
    df["drawdown_pct"]   = df["drawdown_amt"] / df["peak_equity"] * 100

    dd_idx   = df["drawdown_pct"].idxmin()
    pk_idx   = df["peak_equity"][:dd_idx].idxmax() if dd_idx > 0 else 0

    std    = df["net_pnl"].std()
    sharpe = (df["net_pnl"].mean() / std * np.sqrt(252)) if std > 0 else 0
    gp     = wins["net_pnl"].sum() if len(wins) else 0
    gl     = abs(loss["net_pnl"].sum()) if len(loss) else 1

    df["month"] = pd.to_datetime(df["entry_date"]).dt.to_period("M")
    monthly     = df.groupby("month")["net_pnl"].sum().reset_index()
    yearly      = df.groupby(pd.to_datetime(df["entry_date"]).dt.year)["net_pnl"].sum()

    return {
        "total_trades": len(df), "winning_trades": len(wins), "losing_trades": len(loss),
        "win_rate_pct":           round(len(wins)/len(df)*100, 1),
        "total_gross_pnl":        round(df["gross_pnl"].sum(), 2),
        "total_brokerage":        round(df["brokerage"].sum(), 2),
        "total_stt":              round(df["stt"].sum(), 2),
        "total_exchange":         round(df["exchange"].sum(), 2),
        "total_sebi":             round(df["sebi"].sum(), 2),
        "total_stamp":            round(df["stamp"].sum(), 2),
        "total_gst":              round(df["gst"].sum(), 2),
        "total_charges":          round(df["total_charges"].sum(), 2),
        "total_tax":              round(df["stcg_tax"].sum(), 2),
        "total_net_pnl":          round(df["net_pnl"].sum(), 2),
        "return_on_capital_pct":  round(df["net_pnl"].sum()/cap*100, 2),
        "avg_win":                round(wins["net_pnl"].mean(), 2) if len(wins) else 0,
        "avg_loss":               round(loss["net_pnl"].mean(), 2) if len(loss) else 0,
        "largest_win":            round(df["net_pnl"].max(), 2),
        "largest_loss":           round(df["net_pnl"].min(), 2),
        "avg_hold_days":          round(df["hold_days"].mean(), 1),
        "max_drawdown_pct":       round(df["drawdown_pct"].min(), 2),
        "max_drawdown_amt":       round(df["drawdown_amt"].min(), 2),
        "max_drawdown_date":      str(df.loc[dd_idx, "exit_date"]),
        "peak_before_dd_date":    str(df.loc[pk_idx, "exit_date"]),
        "sharpe_ratio":           round(sharpe, 2),
        "profit_factor":          round(gp/gl, 2) if gl > 0 else 999,
        "equity_curve":           df[["entry_date","exit_date","equity","peak_equity",
                                      "drawdown_pct","symbol","net_pnl","return_pct"]].to_dict("records"),
        "monthly_pnl":            [{"month": str(r["month"]), "pnl": round(r["net_pnl"],2)}
                                   for _, r in monthly.iterrows()],
        "yearly_pnl":             {str(yr): round(p,2) for yr,p in yearly.items()},
        "exit_reasons":           df["exit_reason"].value_counts().to_dict(),
        "top_symbols":            df.groupby("symbol")["net_pnl"].sum().sort_values(ascending=False).head(10).round(2).to_dict(),
        "worst_symbols":          df.groupby("symbol")["net_pnl"].sum().sort_values().head(5).round(2).to_dict(),
        "all_trades":             df.to_dict("records"),
    }


# ─── HTML REPORT ──────────────────────────────────────────────────────────────

def generate_report(stats: dict, cfg: dict, report_dir: Path):
    cap      = cfg["capital"]["initial_capital"]
    vcfg     = cfg.get("volatility_filter", {})
    eq_json  = json.dumps(stats["equity_curve"])
    mon_json = json.dumps(stats["monthly_pnl"])

    trade_rows = ""
    for i, t in enumerate(stats["all_trades"], 1):
        win = t["net_pnl"] > 0
        pc  = "#00dfa0" if win else "#ff4060"
        bg  = "rgba(0,223,160,0.04)" if win else "rgba(255,64,96,0.04)"
        er  = t["exit_reason"]
        erc = "#00dfa0" if er in ("target_hit","partial_exit") else "#ff4060" if er=="stop_loss" else "#f5c030"
        trade_rows += f"""<tr style="background:{bg}">
          <td style="color:#4a7090">{i}</td>
          <td><strong>{t['symbol']}</strong></td>
          <td>{t['entry_date']}</td><td>{t['exit_date']}</td>
          <td style="text-align:right">₹{t['entry_price']:,.2f}</td>
          <td style="text-align:right">₹{t['stop_loss']:,.2f}</td>
          <td style="text-align:right">₹{t['target']:,.2f}</td>
          <td style="text-align:right">₹{t['exit_price']:,.2f}</td>
          <td style="text-align:right">{t['qty']}</td>
          <td style="text-align:right">₹{t['buy_value']:,.0f}</td>
          <td style="text-align:right">₹{t['sell_value']:,.0f}</td>
          <td style="text-align:right;color:#f5c030">₹{t['gross_pnl']:,.2f}</td>
          <td style="text-align:right;color:#ff8c42">₹{t['total_charges']:,.2f}</td>
          <td style="text-align:right;color:#ff6090">₹{t['stcg_tax']:,.2f}</td>
          <td style="text-align:right;font-weight:700;color:{pc}">₹{t['net_pnl']:,.2f}</td>
          <td style="text-align:right;color:{pc}">{t['return_pct']}%</td>
          <td style="text-align:right">{t['hold_days']}d</td>
          <td style="color:{erc}">{er}</td>
          <td style="text-align:right">{t['rr_achieved']}</td>
        </tr>"""

    charges_rows = f"""
      <tr><td>Brokerage (₹20×2/trade)</td>  <td style="text-align:right;color:#ff8c42">₹{stats['total_brokerage']:,.2f}</td></tr>
      <tr><td>STT (0.1% on sell)</td>         <td style="text-align:right;color:#ff8c42">₹{stats['total_stt']:,.2f}</td></tr>
      <tr><td>Exchange Charges</td>            <td style="text-align:right;color:#ff8c42">₹{stats['total_exchange']:,.2f}</td></tr>
      <tr><td>SEBI Charges</td>               <td style="text-align:right;color:#ff8c42">₹{stats['total_sebi']:,.2f}</td></tr>
      <tr><td>Stamp Duty (0.015% buy)</td>    <td style="text-align:right;color:#ff8c42">₹{stats['total_stamp']:,.2f}</td></tr>
      <tr><td>GST (18%)</td>                  <td style="text-align:right;color:#ff8c42">₹{stats['total_gst']:,.2f}</td></tr>
      <tr style="border-top:1px solid #2a4060"><td><strong>Total Charges</strong></td>
          <td style="text-align:right;color:#ff4d6d"><strong>₹{stats['total_charges']:,.2f}</strong></td></tr>
      <tr><td><strong>STCG Tax (15%)</strong></td>
          <td style="text-align:right;color:#ff4d6d"><strong>₹{stats['total_tax']:,.2f}</strong></td></tr>"""

    sym_rows  = "".join(f'<tr><td>{s}</td><td style="text-align:right;color:{"#00dfa0" if p>0 else "#ff4060"}">₹{p:,.0f}</td></tr>'
                        for s,p in stats["top_symbols"].items())
    yr_rows   = "".join(f'<tr><td>{y}</td><td style="text-align:right;color:{"#00dfa0" if p>0 else "#ff4060"}">₹{p:,.0f}</td>'
                        f'<td style="text-align:right;color:{"#00dfa0" if p>0 else "#ff4060"}">{round(p/cap*100,1)}%</td></tr>'
                        for y,p in stats["yearly_pnl"].items())
    er_rows   = "".join(f"<tr><td>{k}</td><td style='text-align:right'>{v}</td></tr>"
                        for k,v in stats["exit_reasons"].items())
    bad_rows  = "".join(f'<tr><td>{s}</td><td style="text-align:right;color:#ff4060">₹{p:,.0f}</td></tr>'
                        for s,p in stats["worst_symbols"].items())

    vix_info = ""
    if vcfg.get("enabled"):
        vix_info = (f"VIX no-entry>{vcfg.get('vix_no_entry_above',18)} | "
                    f"caution>{vcfg.get('vix_caution_above',16)} | "
                    f"Nifty 50 EMA: {'ON' if vcfg.get('nifty_trend_filter') else 'OFF'} | "
                    f"Nifty 200 EMA: {'ON' if vcfg.get('nifty_200ema_filter') else 'OFF'} | "
                    f"ADX≥{vcfg.get('adx_min_threshold',25)}: {'ON' if vcfg.get('adx_filter') else 'OFF'}")

    html = f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="UTF-8"><title>Swing Backtest Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&display=swap" rel="stylesheet">
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#07090f;color:#b8cfe0;font-family:'IBM Plex Mono',monospace;min-height:100vh}}
  .hdr{{background:linear-gradient(135deg,#0b1520,#0f1e2e);border-bottom:1px solid #1a2d42;padding:24px 32px}}
  .hdr h1{{color:#6db3f2;font-size:20px;margin-bottom:4px}}
  .hdr .meta{{color:#3d6080;font-size:10px;letter-spacing:1px;line-height:1.8}}
  .body{{max-width:1400px;margin:0 auto;padding:24px 20px 60px}}
  .sec{{margin-bottom:32px}}
  .sec-t{{font-size:10px;color:#3d7aaa;letter-spacing:3px;text-transform:uppercase;
          margin-bottom:12px;padding-bottom:6px;border-bottom:1px solid #132030}}
  .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:10px}}
  .card{{background:#0c1520;border:1px solid #162535;border-radius:10px;padding:14px 16px}}
  .lbl{{font-size:9px;color:#3d6080;letter-spacing:2px;text-transform:uppercase;margin-bottom:6px}}
  .val{{font-size:18px;font-weight:700}}.sub{{font-size:10px;color:#3d6080;margin-top:4px}}
  .g{{color:#00dfa0}}.r{{color:#ff4060}}.y{{color:#f5c030}}.w{{color:#e0eaf5}}
  .cw{{background:#0b1420;border:1px solid #162535;border-radius:10px;padding:18px;margin-bottom:18px}}
  .ct{{font-size:10px;color:#3d7aaa;letter-spacing:2px;text-transform:uppercase;margin-bottom:12px}}
  .tw{{overflow-x:auto;border-radius:10px;border:1px solid #162535}}
  table{{width:100%;border-collapse:collapse;font-size:11px}}
  th{{background:#0c1825;color:#3d7aaa;padding:9px 11px;text-align:left;
      border-bottom:1px solid #1a2d42;white-space:nowrap;font-weight:600}}
  td{{padding:7px 11px;border-bottom:1px solid #0f1c28;white-space:nowrap}}
  tbody tr:hover td{{background:#0f1e2e}}
  .two{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}
  @media(max-width:800px){{.two{{grid-template-columns:1fr}}}}
  .vix-badge{{display:inline-block;padding:3px 10px;border-radius:5px;font-size:10px;
              background:rgba(245,192,48,0.12);color:#f5c030;border:1px solid rgba(245,192,48,0.3);margin-top:6px}}
</style></head><body>
<div class="hdr">
  <h1>⚡ Swing Trade Backtest Report</h1>
  <div class="meta">
    {cfg['backtest']['start_date']} → {cfg['backtest']['end_date']} &nbsp;·&nbsp;
    Capital ₹{cap:,} &nbsp;·&nbsp; R:R {cfg['trade_rules']['reward_to_risk_ratio']}:1 &nbsp;·&nbsp;
    Risk/trade {cfg['capital']['risk_per_trade_pct']}% &nbsp;·&nbsp;
    Max hold {cfg['backtest'].get('max_hold_days','∞')}d &nbsp;·&nbsp;
    SL: {cfg['exit_rules']['stop_loss_type']} &nbsp;·&nbsp;
    Top {cfg['stock_selection']['ranking']['top_n_stocks']} ranked &nbsp;·&nbsp;
    Generated {datetime.now().strftime('%d %b %Y %H:%M')}
    {"<br>" + vix_info if vix_info else ""}
  </div>
</div>
<div class="body">

<div class="sec">
  <div class="sec-t">Performance Summary</div>
  <div class="cards">
    <div class="card"><div class="lbl">Total Trades</div>
      <div class="val w">{stats['total_trades']}</div>
      <div class="sub">{stats['winning_trades']}W / {stats['losing_trades']}L</div></div>
    <div class="card"><div class="lbl">Win Rate</div>
      <div class="val {'g' if stats['win_rate_pct']>=50 else 'y'}">{stats['win_rate_pct']}%</div>
      <div class="sub">Target ≥ 50%</div></div>
    <div class="card"><div class="lbl">Net P&L</div>
      <div class="val {'g' if stats['total_net_pnl']>0 else 'r'}">₹{stats['total_net_pnl']:,.0f}</div>
      <div class="sub">After tax & charges</div></div>
    <div class="card"><div class="lbl">Return on Capital</div>
      <div class="val {'g' if stats['return_on_capital_pct']>0 else 'r'}">{stats['return_on_capital_pct']}%</div>
      <div class="sub">On ₹{cap:,}</div></div>
    <div class="card"><div class="lbl">Profit Factor</div>
      <div class="val {'g' if stats['profit_factor']>1.5 else 'y' if stats['profit_factor']>1 else 'r'}">{stats['profit_factor']}</div>
      <div class="sub">Target ≥ 1.5</div></div>
    <div class="card"><div class="lbl">Sharpe Ratio</div>
      <div class="val {'g' if stats['sharpe_ratio']>1 else 'y'}">{stats['sharpe_ratio']}</div>
      <div class="sub">Target ≥ 1.0</div></div>
    <div class="card"><div class="lbl">Max Drawdown</div>
      <div class="val r">{stats['max_drawdown_pct']}%</div>
      <div class="sub">on {stats['max_drawdown_date']}</div></div>
    <div class="card"><div class="lbl">Avg Hold Days</div>
      <div class="val w">{stats['avg_hold_days']}</div>
      <div class="sub">Per trade</div></div>
    <div class="card"><div class="lbl">Avg Win</div>
      <div class="val g">₹{stats['avg_win']:,.0f}</div></div>
    <div class="card"><div class="lbl">Avg Loss</div>
      <div class="val r">₹{abs(stats['avg_loss']):,.0f}</div></div>
    <div class="card"><div class="lbl">Largest Win</div>
      <div class="val g">₹{stats['largest_win']:,.0f}</div></div>
    <div class="card"><div class="lbl">Largest Loss</div>
      <div class="val r">₹{abs(stats['largest_loss']):,.0f}</div></div>
  </div>
</div>

<div class="sec">
  <div class="sec-t">P&L Breakdown — Where the Money Goes</div>
  <div class="tw" style="max-width:480px">
    <table>
      <tr><td>Gross P&L</td><td style="text-align:right;color:#f5c030;font-weight:700">₹{stats['total_gross_pnl']:,.2f}</td></tr>
      {charges_rows}
      <tr style="background:#0c1825;border-top:2px solid #1a3050">
        <td><strong>Final Net P&L</strong></td>
        <td style="text-align:right;font-weight:700;font-size:15px;color:{'#00dfa0' if stats['total_net_pnl']>0 else '#ff4060'}">
          ₹{stats['total_net_pnl']:,.2f}</td></tr>
    </table>
  </div>
</div>

<div class="sec">
  <div class="sec-t">Equity Curve — Max Drawdown Marked</div>
  <div class="cw"><div class="ct">Portfolio Value vs Peak (Buy date → Sell date shown on hover)</div>
    <canvas id="eqC" height="55"></canvas>
    <div style="font-size:10px;color:#ff4060;text-align:center;margin-top:6px">
      ▼ Max Drawdown: {stats['max_drawdown_pct']}% (₹{abs(stats['max_drawdown_amt']):,.0f}) peaked {stats['peak_before_dd_date']} → trough {stats['max_drawdown_date']}
    </div>
  </div>
  <div class="cw"><div class="ct">Drawdown % Over Time</div>
    <canvas id="ddC" height="32"></canvas></div>
</div>

<div class="sec">
  <div class="two">
    <div><div class="sec-t">Monthly Net P&L</div>
      <div class="cw"><canvas id="monC" height="110"></canvas></div></div>
    <div>
      <div class="sec-t">Yearly Summary</div>
      <div class="tw" style="margin-bottom:16px">
        <table><thead><tr><th>Year</th><th style="text-align:right">Net P&L</th><th style="text-align:right">Return%</th></tr></thead>
        <tbody>{yr_rows}</tbody></table></div>
      <div class="sec-t">Exit Reasons</div>
      <div class="tw">
        <table><thead><tr><th>Reason</th><th style="text-align:right">Count</th></tr></thead>
        <tbody>{er_rows}</tbody></table></div>
    </div>
  </div>
</div>

<div class="sec">
  <div class="two">
    <div><div class="sec-t">Top 10 Stocks</div>
      <div class="tw"><table><thead><tr><th>Symbol</th><th style="text-align:right">Net P&L</th></tr></thead>
      <tbody>{sym_rows}</tbody></table></div></div>
    <div><div class="sec-t">Bottom 5 Stocks</div>
      <div class="tw"><table><thead><tr><th>Symbol</th><th style="text-align:right">Net P&L</th></tr></thead>
      <tbody>{bad_rows}</tbody></table></div></div>
  </div>
</div>

<div class="sec">
  <div class="sec-t">Complete Trade Log — All {stats['total_trades']} Trades</div>
  <div class="tw"><table><thead><tr>
    <th>#</th><th>Symbol</th><th>Buy Date</th><th>Sell Date</th>
    <th style="text-align:right">Entry ₹</th><th style="text-align:right">SL ₹</th>
    <th style="text-align:right">Target ₹</th><th style="text-align:right">Exit ₹</th>
    <th style="text-align:right">Qty</th><th style="text-align:right">Buy Val</th>
    <th style="text-align:right">Sell Val</th><th style="text-align:right">Gross P&L</th>
    <th style="text-align:right">Charges</th><th style="text-align:right">Tax</th>
    <th style="text-align:right">Net P&L</th><th style="text-align:right">Return%</th>
    <th style="text-align:right">Hold</th><th>Exit Reason</th><th style="text-align:right">R:R</th>
  </tr></thead><tbody>{trade_rows}</tbody></table></div>
</div>

</div>
<script>
const eq={eq_json}, mon={mon_json}, cap={cap};
const maxDdPct={stats['max_drawdown_pct']};
const ddIdx=eq.findIndex(r=>r.drawdown_pct===Math.min(...eq.map(r=>r.drawdown_pct)));
const xL=eq.map(r=>r.exit_date);

const ann={{id:'ann',afterDraw(chart){{
  if(ddIdx<0)return;
  const ctx=chart.ctx,xs=chart.scales.x,ys=chart.scales.y;
  const x=xs.getPixelForValue(ddIdx),y=ys.getPixelForValue(eq[ddIdx].equity);
  ctx.save();
  ctx.setLineDash([4,4]);ctx.strokeStyle='rgba(255,64,96,0.7)';ctx.lineWidth=1.5;
  ctx.beginPath();ctx.moveTo(x,chart.chartArea.top);ctx.lineTo(x,chart.chartArea.bottom);ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle='#ff4060';ctx.beginPath();ctx.arc(x,y,7,0,Math.PI*2);ctx.fill();
  ctx.strokeStyle='#fff';ctx.lineWidth=1.5;ctx.stroke();
  const lbl=`▼ Max DD ${{maxDdPct}}%  |  ${{eq[ddIdx].exit_date}}`;
  ctx.font='700 11px IBM Plex Mono,monospace';
  const tw=ctx.measureText(lbl).width;
  const bx=Math.min(x+10,chart.chartArea.right-tw-16),by=Math.max(y-36,chart.chartArea.top+4);
  ctx.fillStyle='rgba(200,30,60,0.92)';ctx.beginPath();ctx.roundRect(bx,by,tw+14,24,5);ctx.fill();
  ctx.fillStyle='#fff';ctx.fillText(lbl,bx+7,by+16);ctx.restore();
}}}};

if(eq.length){{
  new Chart(document.getElementById('eqC'),{{
    type:'line',plugins:[ann],
    data:{{labels:xL,datasets:[
      {{label:'Portfolio (₹)',data:eq.map(r=>r.equity),borderColor:'#00dfa0',
        backgroundColor:'rgba(0,223,160,0.07)',fill:true,tension:0.3,pointRadius:0,borderWidth:2}},
      {{label:'Peak (₹)',data:eq.map(r=>r.peak_equity),borderColor:'rgba(100,160,220,0.35)',
        borderDash:[5,5],fill:false,tension:0.3,pointRadius:0,borderWidth:1}}
    ]}},
    options:{{responsive:true,interaction:{{mode:'index',intersect:false}},
      plugins:{{legend:{{labels:{{color:'#5a8aaa',font:{{size:11}}}}}},
        tooltip:{{backgroundColor:'#0c1825',borderColor:'#1a3050',borderWidth:1,
          titleColor:'#6db3f2',bodyColor:'#9abccc',
          callbacks:{{
            title:items=>{{const r=eq[items[0].dataIndex];return `${{r.symbol}}  |  Buy: ${{r.entry_date}}  →  Sell: ${{r.exit_date}}`}},
            label:item=>{{const r=eq[item.dataIndex];
              if(item.datasetIndex===0){{
                const p=r.net_pnl>=0?'+₹'+r.net_pnl.toLocaleString('en-IN'):'-₹'+Math.abs(r.net_pnl).toLocaleString('en-IN');
                return[` Portfolio: ₹${{item.raw.toLocaleString('en-IN',{{maximumFractionDigits:0}})}}`,
                       ` Trade P&L: ${{p}} (${{r.return_pct}}%)`,
                       ` Drawdown: ${{r.drawdown_pct.toFixed(2)}}%`];}}
              return ` Peak: ₹${{item.raw.toLocaleString('en-IN',{{maximumFractionDigits:0}})}}`}}
          }}}}
      }},
      scales:{{
        x:{{ticks:{{color:'#3d6080',maxTicksLimit:16,maxRotation:35,font:{{size:10}}}},grid:{{color:'#0f1c28'}}}},
        y:{{ticks:{{color:'#3d6080',callback:v=>'₹'+(v/1000).toFixed(0)+'K'}},grid:{{color:'#111e2c'}}}}
      }}
    }}
  }});

  new Chart(document.getElementById('ddC'),{{
    type:'line',
    data:{{labels:xL,datasets:[{{label:'Drawdown %',data:eq.map(r=>r.drawdown_pct),
      borderColor:'#ff4060',backgroundColor:'rgba(255,64,96,0.1)',
      fill:true,tension:0.3,pointRadius:0,borderWidth:1.5}}]}},
    options:{{responsive:true,
      plugins:{{legend:{{labels:{{color:'#5a8aaa',font:{{size:11}}}}}},
        tooltip:{{backgroundColor:'#0c1825',borderColor:'#1a3050',borderWidth:1,
          titleColor:'#6db3f2',bodyColor:'#9abccc',
          callbacks:{{
            title:items=>{{const r=eq[items[0].dataIndex];return `${{r.symbol}}  |  Buy: ${{r.entry_date}}  →  Sell: ${{r.exit_date}}`}},
            label:item=>` Drawdown: ${{item.raw.toFixed(2)}}%`
          }}}}
      }},
      scales:{{
        x:{{ticks:{{color:'#3d6080',maxTicksLimit:16,maxRotation:35,font:{{size:10}}}},grid:{{color:'#0f1c28'}}}},
        y:{{ticks:{{color:'#3d6080',callback:v=>v.toFixed(1)+'%'}},grid:{{color:'#111e2c'}}}}
      }}
    }}
  }});
}}

if(mon.length){{
  new Chart(document.getElementById('monC'),{{
    type:'bar',
    data:{{labels:mon.map(r=>r.month),datasets:[{{label:'Net P&L',
      data:mon.map(r=>r.pnl),borderRadius:4,
      backgroundColor:mon.map(r=>r.pnl>0?'rgba(0,223,160,0.75)':'rgba(255,64,96,0.75)')}}]}},
    options:{{responsive:true,
      plugins:{{legend:{{display:false}},
        tooltip:{{backgroundColor:'#0c1825',borderColor:'#1a3050',borderWidth:1,
          titleColor:'#6db3f2',bodyColor:'#9abccc',
          callbacks:{{label:item=>` ₹${{item.raw.toLocaleString('en-IN',{{maximumFractionDigits:0}})}}`}}}}
      }},
      scales:{{
        x:{{ticks:{{color:'#3d6080',maxRotation:45,font:{{size:10}}}},grid:{{color:'#0f1c28'}}}},
        y:{{ticks:{{color:'#3d6080',callback:v=>'₹'+(v/1000).toFixed(0)+'K'}},grid:{{color:'#111e2c'}}}}
      }}
    }}
  }});
}}
</script></body></html>"""

    out = report_dir / "backtest_report.html"
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  📊 Report    → {out}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str)
    parser.add_argument("--config", default="config.json")
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
    max_hold   = cfg["backtest"].get("max_hold_days", 9999)

    print(f"\n⚡ Swing Trade Backtester")
    print(f"   Period    : {start_date} → {end_date}")
    print(f"   Data dir  : {data_dir}/")
    print(f"   Report dir: {report_dir}/")
    print(f"   Capital   : ₹{cfg['capital']['initial_capital']:,}")
    print(f"   R:R       : {cfg['trade_rules']['reward_to_risk_ratio']}:1")
    print(f"   Max hold  : {max_hold} days")
    print(f"   Top N     : {top_n} stocks/day\n")

    # Load stocks — include warmup candles for indicator accuracy
    from datetime import datetime as _dt, timedelta as _td
    warmup_days  = cfg["backtest"].get("warmup_days", 100)
    warmup_start = (_dt.strptime(start_date, "%Y-%m-%d") - _td(days=warmup_days)).strftime("%Y-%m-%d")
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
            # Load from warmup_start so EMAs are accurate from start_date
            df = df[(df["date"] >= warmup_d) &
                    (df["date"] <= end_date_d)].reset_index(drop=True)
            if len(df) >= 60:
                # Add indicators on full data (warmup + trading period)
                df = add_indicators(df, cfg)
                # Mark which rows are in the actual trading period
                df["in_trading_period"] = df["date"] >= start_date_d
                all_data[sym] = df
        except Exception as e:
            print(f"  ⚠ {sym}: {e}")

    if not all_data:
        print("❌ No data found. Run zerodha_downloader.py first.")
        return
    print(f"  Loaded {len(all_data)} stocks ({warmup_start} → {end_date} incl. {warmup_days}d warmup)\n")

    # Load Nifty 50 — include warmup for EMA accuracy
    nifty_path = data_dir / f"NIFTY50_{interval}.csv"
    if nifty_path.exists():
        nifty_df = pd.read_csv(nifty_path)
        nifty_df["date"] = pd.to_datetime(nifty_df["date"]).dt.date
        nifty_df = nifty_df[nifty_df["date"] >= warmup_d].reset_index(drop=True)
        print(f"  ✓ Nifty 50 index loaded (incl. warmup)")
    else:
        nifty_df = next(iter(all_data.values()))[["date","close"]].copy()
        print("  ⚠ NIFTY50 not found — using stock average as proxy")

    # Load India VIX — include warmup
    vix_path = data_dir / f"INDIAVIX_{interval}.csv"
    if vix_path.exists():
        vix_df = pd.read_csv(vix_path)
        vix_df["date"] = pd.to_datetime(vix_df["date"]).dt.date
        vix_df = vix_df[vix_df["date"] >= warmup_d].reset_index(drop=True)
        print(f"  ✓ India VIX loaded (incl. warmup)")
    else:
        vix_df = None
        print("  ⚠ INDIAVIX not found — VIX filter disabled")

    vcfg = cfg.get("volatility_filter", {})
    vix_map      = build_vix_map(vix_df, cfg)      if vix_df is not None else {}
    nifty_trend  = build_nifty_trend_map(nifty_df, cfg)

    if vcfg.get("enabled") and vix_map:
        no_e = sum(1 for v,r in vix_map.values() if r=="no_entry")
        caut = sum(1 for v,r in vix_map.values() if r=="caution")
        print(f"  VIX filter     : {no_e} days blocked (VIX>{vcfg.get('vix_no_entry_above',18)}), "
              f"{caut} days caution (VIX>{vcfg.get('vix_caution_above',16)})")

    if nifty_trend:
        blocked_50  = sum(1 for v in nifty_trend.values() if not v.get("ema50_ok",  True))
        blocked_200 = sum(1 for v in nifty_trend.values() if not v.get("ema200_ok", True))
        blocked_adx = sum(1 for v in nifty_trend.values() if not v.get("trend_ok",  True))
        adx_caut    = sum(1 for v in nifty_trend.values() if v.get("adx_caution", False))
        avg_adx     = np.mean([v.get("adx", 0) for v in nifty_trend.values() if v.get("adx", 0) > 0])
        print(f"  50 EMA filter  : {blocked_50} days Nifty below 50 EMA")
        print(f"  200 EMA filter : {blocked_200} days Nifty below 200 EMA (bear market blocked)")
        print(f"  ADX filter     : {blocked_adx} total days blocked | {adx_caut} caution days | avg ADX: {avg_adx:.1f}")
        print(f"  Combined block : {sum(1 for v in nifty_trend.values() if not v.get('trend_ok',True))} days no new entries")
    print()

    # Daily ranking — only for actual trading period, not warmup
    print("  📊 Running daily ranking...")
    # Use only dates in the trading period (start_date → end_date)
    all_dates = sorted(set(
        d for df in all_data.values()
        for d in df[df["in_trading_period"]]["date"].tolist()
    ))
    daily_rankings, eligible = [], {sym: set() for sym in all_data}
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
    print(f"  NET P&L           : ₹{stats['total_net_pnl']:,.2f}")
    print(f"  RETURN ON CAPITAL : {stats['return_on_capital_pct']}%")
    print(f"  MAX DRAWDOWN      : {stats['max_drawdown_pct']}% on {stats['max_drawdown_date']}")
    print(f"{'─'*54}\n")

    if cfg["output"]["save_trade_log"]:
        cols = ["symbol","entry_date","exit_date","entry_price","stop_loss","target",
                "exit_price","qty","buy_value","sell_value","gross_pnl",
                "brokerage","stt","exchange","sebi","stamp","gst","total_charges",
                "net_before_tax","stcg_tax","net_pnl","return_pct","rr_achieved",
                "exit_reason","hold_days","result"]
        tp = report_dir / "trade_log.csv"
        pd.DataFrame(all_trades)[cols].sort_values("entry_date").to_csv(tp, index=False)
        print(f"  📋 Trade log → {tp}")

    if cfg["output"]["save_report"]:
        generate_report(stats, cfg, report_dir)


if __name__ == "__main__":
    main()