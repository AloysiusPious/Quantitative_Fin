"""
LTCG Evening Scanner
======================
Runs after market close (3:30 PM+).
Logs into Zerodha, downloads today's data,
applies all filters and signals, outputs
tomorrow's trade candidates.

USAGE:
    python evening_scanner.py
    python evening_scanner.py --config config_ltcg.json

OUTPUT:
    ✅ BUY signals   — stocks to place limit orders for tomorrow open
    ⚠  MARKET STATUS — VIX / Nifty trend / ADX check
    📋 OPEN POSITIONS — reminder of existing holdings to monitor

WORKFLOW:
    4:00 PM → market closes
    4:30 PM → run this script
    4:31 PM → enter User ID / Password / OTP when prompted
    4:33 PM → review signals
    4:35 PM → place limit orders in Zerodha for tomorrow 9:15 AM open
"""

import requests
import pandas as pd
import numpy as np
import json
import getpass
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta, date


# ─── LOGIN ────────────────────────────────────────────────────────────────────

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept":     "application/json, text/plain, */*",
    "Content-Type": "application/x-www-form-urlencoded",
    "Referer":    "https://kite.zerodha.com/",
    "Origin":     "https://kite.zerodha.com",
})


def login_and_get_enctoken() -> str:
    """
    Full Zerodha login flow.
    Prompts for User ID, Password (hidden), 6-digit OTP (hidden).
    Returns enctoken on success.
    """
    print("\n🔐 Zerodha Login")
    print("─" * 40)
    # user_id  = input("  User ID          : ").strip().upper()
    # password = getpass.getpass("  Password (hidden): ")
    user_id = 'XS8910'
    password ='chevisa1234'

    # Step 1 — credentials
    print("\n  [1/3] Sending credentials...", end=" ", flush=True)
    try:
        r = SESSION.post("https://kite.zerodha.com/api/login", data={
            "user_id": user_id, "password": password,
        }, timeout=15)
        data = r.json()
        if data.get("status") != "success":
            print(f"✗\n  ❌ {data.get('message', 'Login failed')}")
            sys.exit(1)
        request_id = data["data"]["request_id"]
        print("✓")
    except Exception as e:
        print(f"✗\n  ❌ Connection error: {e}")
        sys.exit(1)

    # Step 2 — OTP
    print("\n  Open your Authenticator app and enter the 6-digit OTP:")
    #otp = getpass.getpass("  OTP (hidden)     : ").strip()
    otp = input("  2FA otp          : ").strip()
    if not otp.isdigit() or len(otp) != 6:
        print("  ❌ OTP must be 6 digits")
        sys.exit(1)

    print("  [2/3] Submitting OTP...", end=" ", flush=True)
    try:
        r = SESSION.post("https://kite.zerodha.com/api/twofa", data={
            "user_id": user_id, "request_id": request_id,
            "twofa_value": otp, "twofa_type": "totp", "skip_session": "",
        }, timeout=15)
        data = r.json()
        if data.get("status") != "success":
            print(f"✗\n  ❌ {data.get('message', '2FA failed')}")
            sys.exit(1)
        print("✓")
    except Exception as e:
        print(f"✗\n  ❌ {e}")
        sys.exit(1)

    # Step 3 — enctoken
    print("  [3/3] Extracting enctoken...", end=" ", flush=True)
    enctoken = SESSION.cookies.get("enctoken")
    if not enctoken:
        for c in SESSION.cookies:
            if "enctoken" in c.name.lower():
                enctoken = c.value
                break
    if not enctoken:
        print("✗\n  ❌ enctoken not found")
        sys.exit(1)
    print("✓\n")
    return enctoken, user_id


def save_enctoken(enctoken: str, cfg_path: str):
    """Save enctoken back to config file for reuse today."""
    import re
    path = Path(cfg_path)
    if path.exists():
        raw = path.read_text(encoding="utf-8")
        updated = re.sub(r'("enctoken"\s*:\s*)"[^"]*"', rf'\g<1>"{enctoken}"', raw)
        if updated != raw:
            path.write_text(updated, encoding="utf-8")


# ─── DATA FETCH ───────────────────────────────────────────────────────────────

def get_headers(enctoken: str) -> dict:
    return {"Authorization": f"enctoken {enctoken}", "User-Agent": "Mozilla/5.0"}


def fetch_historical(token: int, from_date: str, to_date: str,
                     enctoken: str, delay: float = 0.35) -> pd.DataFrame:
    """Fetch OHLCV — auto-chunks if range > 1800 days."""
    from datetime import datetime, timedelta
    start_dt = datetime.strptime(from_date, "%Y-%m-%d")
    end_dt   = datetime.strptime(to_date,   "%Y-%m-%d")
    CHUNK    = 1800
    all_candles = []
    chunk_start = start_dt

    while chunk_start < end_dt:
        chunk_end = min(chunk_start + timedelta(days=CHUNK), end_dt)
        url    = f"https://kite.zerodha.com/oms/instruments/historical/{token}/day"
        params = {"from": chunk_start.strftime("%Y-%m-%d"),
                  "to":   chunk_end.strftime("%Y-%m-%d"), "oi": 1}
        try:
            time.sleep(delay)
            r    = SESSION.get(url, headers=get_headers(enctoken), params=params, timeout=20)
            data = r.json()
            if data.get("status") == "success":
                all_candles.extend(data["data"]["candles"])
        except Exception as e:
            print(f"    ⚠ Fetch error: {e}")
        chunk_start = chunk_end + timedelta(days=1)

    if not all_candles:
        return pd.DataFrame()
    df = pd.DataFrame(all_candles, columns=["date","open","high","low","close","volume","oi"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def get_instrument_token(symbol: str, enctoken: str) -> int | None:
    """Lookup instrument token from NSE instruments."""
    try:
        r  = SESSION.get("https://api.kite.trade/instruments",
                         headers=get_headers(enctoken), timeout=20)
        df = pd.read_csv(pd.io.common.StringIO(r.text))
        m  = df[(df["tradingsymbol"] == symbol) & (df["exchange"] == "NSE")]
        if not m.empty:
            return int(m.iloc[0]["instrument_token"])
    except Exception as e:
        print(f"    ⚠ Token lookup error {symbol}: {e}")
    return None


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
    df["momentum"]  = df["close"].pct_change(cfg["stock_selection"]["ranking"]["momentum_days"])

    # RSI
    rsi_p = s.get("rsi_period", 14)
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(rsi_p).mean()
    loss_ = (-delta.clip(upper=0)).rolling(rsi_p).mean()
    rs    = gain / loss_.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # Signal
    tol     = s["pullback_tolerance_pct"] / 100
    rsi_min = s.get("rsi_min_entry", 45)
    df["in_uptrend"]     = (df["close"] > df["ema_fast"]) & (df["close"] > df["ema_slow"])
    df["pullback"]       = (df["low"] <= df["ema_fast"] * (1 + tol)) & (df["close"] > df["ema_fast"])
    body = abs(df["close"] - df["open"])
    rng  = df["high"] - df["low"]
    df["bullish_candle"] = (df["close"] > df["open"]) & (body >= s["bullish_body_pct"] / 100 * rng.replace(0, np.nan))
    df["vol_confirmed"]  = df["volume"] > df["vol_avg"] * s["volume_multiplier"]
    df["rsi_confirmed"]  = df["rsi"] > rsi_min
    df["signal"]         = (df["in_uptrend"] & df["pullback"] &
                            df["bullish_candle"] & df["vol_confirmed"] & df["rsi_confirmed"])
    return df


# ─── MARKET FILTERS ───────────────────────────────────────────────────────────

def check_market_filters(nifty_df: pd.DataFrame, vix_df: pd.DataFrame,
                          cfg: dict) -> dict:
    """
    Checks all market-level filters for TODAY.
    Returns dict with filter status and values.
    """
    vcfg   = cfg.get("volatility_filter", {})
    result = {}

    # ── VIX ──
    if vix_df is not None and not vix_df.empty:
        vix_recent = vix_df.tail(vcfg.get("vix_lookback_days", 5))["close"].mean()
        no_entry   = vcfg.get("vix_no_entry_above", 20.0)
        caution    = vcfg.get("vix_caution_above",  16.0)
        result["vix"]         = round(vix_recent, 2)
        result["vix_regime"]  = ("no_entry" if vix_recent >= no_entry else
                                 "caution"  if vix_recent >= caution  else "normal")
    else:
        result["vix"] = None
        result["vix_regime"] = "unknown"

    # ── Nifty EMAs ──
    if nifty_df is not None and not nifty_df.empty:
        nifty_df = nifty_df.copy()
        nifty_df["ema50"]  = nifty_df["close"].ewm(span=50,  adjust=False).mean()
        nifty_df["ema200"] = nifty_df["close"].ewm(span=200, adjust=False).mean()
        latest = nifty_df.iloc[-1]
        result["nifty_close"]  = round(latest["close"], 2)
        result["nifty_ema50"]  = round(latest["ema50"],  2)
        result["nifty_ema200"] = round(latest["ema200"], 2)
        result["above_50ema"]  = bool(latest["close"] > latest["ema50"])
        result["above_200ema"] = bool(latest["close"] > latest["ema200"])

        # ── ADX on Nifty ──
        adx_p = vcfg.get("adx_period", 14)
        h = nifty_df["high"].values
        l = nifty_df["low"].values
        c = nifty_df["close"].values
        n = len(h)

        tr = np.zeros(n)
        dm_pos = np.zeros(n)
        dm_neg = np.zeros(n)

        for k in range(1, n):
            tr[k] = max(h[k] - l[k], abs(h[k] - c[k - 1]), abs(l[k] - c[k - 1]))
            up = h[k] - h[k - 1]
            dn = l[k - 1] - l[k]
            dm_pos[k] = up if (up > dn and up > 0) else 0
            dm_neg[k] = dn if (dn > up and dn > 0) else 0

        atr_w = np.zeros(n)
        dmp_w = np.zeros(n)
        dmm_w = np.zeros(n)
        atr_w[adx_p] = tr[1:adx_p + 1].sum()
        dmp_w[adx_p] = dm_pos[1:adx_p + 1].sum()
        dmm_w[adx_p] = dm_neg[1:adx_p + 1].sum()

        for k in range(adx_p + 1, n):
            atr_w[k] = atr_w[k - 1] - atr_w[k - 1] / adx_p + tr[k]
            dmp_w[k] = dmp_w[k - 1] - dmp_w[k - 1] / adx_p + dm_pos[k]
            dmm_w[k] = dmm_w[k - 1] - dmm_w[k - 1] / adx_p + dm_neg[k]

        with np.errstate(divide='ignore', invalid='ignore'):
            di_pos = np.where(atr_w > 0, dmp_w / atr_w * 100, 0)
            di_neg = np.where(atr_w > 0, dmm_w / atr_w * 100, 0)
            di_sum = di_pos + di_neg
            dx = np.where(di_sum > 0, np.abs(di_pos - di_neg) / di_sum * 100, 0)

        adx_arr = np.zeros(n)
        adx_arr[adx_p * 2] = dx[adx_p:adx_p * 2 + 1].mean()
        for k in range(adx_p * 2 + 1, n):
            adx_arr[k] = (adx_arr[k - 1] * (adx_p - 1) + dx[k]) / adx_p

        result["adx"] = round(float(adx_arr[-1]), 2)
        result["adx_ok"] = result["adx"] >= vcfg.get("adx_min_threshold", 25)
    else:
        result["nifty_close"] = result["nifty_ema50"] = result["nifty_ema200"] = None
        result["above_50ema"] = result["above_200ema"] = result["adx_ok"] = False
        result["adx"] = 0

    # ── Overall: can we trade today? ──
    result["can_trade"] = (
        result["vix_regime"] != "no_entry" and
        result.get("above_50ema",  False) and
        result.get("above_200ema", False) and
        result.get("adx_ok",       False)
    )
    return result


# ─── RANKING ──────────────────────────────────────────────────────────────────

def rank_signals(signals: list, nifty_df: pd.DataFrame, cfg: dict) -> list:
    """Rank signal stocks by composite score. Returns sorted list."""
    rk      = cfg["stock_selection"]["ranking"]
    rs_days = rk["rs_lookback_days"]
    w       = rk["weights"]

    nifty_past   = nifty_df.tail(rs_days + 1)
    nifty_return = 0.0
    if len(nifty_past) >= 2:
        nifty_return = (nifty_past["close"].iloc[-1] - nifty_past["close"].iloc[0]) / nifty_past["close"].iloc[0]

    ranked = []
    for sig in signals:
        df  = sig["df"]
        row = df.iloc[-1]
        past = df.tail(rs_days + 1)

        ema_gap   = (row["close"] - row["ema_fast"]) / row["ema_fast"] if row["ema_fast"] > 0 else 0
        rs_return = (past["close"].iloc[-1] - past["close"].iloc[0]) / past["close"].iloc[0] if past["close"].iloc[0] > 0 else 0
        vol_ratio = row["volume"] / row["vol_avg"] if row["vol_avg"] > 0 else 1
        atr_pct   = row["atr"] / row["close"] if row["close"] > 0 else 0.02

        s = {}
        s["trend_strength"]    = max(0, min(ema_gap, 0.1) / 0.1)
        rs_diff = rs_return - nifty_return
        s["relative_strength"] = min(max((rs_diff + 0.1) / 0.2, 0), 1)
        s["volume_surge"]      = min(vol_ratio, 3) / 3
        s["momentum"]          = min(max((float(row.get("momentum", 0)) + 0.05) / 0.1, 0), 1)
        s["volatility_score"]  = max(0, 1 - (atr_pct / 0.05))

        score = round(sum(s[k] * w[k] for k in w), 4)

        # Position sizing
        capital     = cfg["capital"]["initial_capital"]
        risk_pct    = cfg["capital"]["risk_per_trade_pct"]
        max_pos_pct = cfg["capital"].get("max_position_pct", 20.0)
        entry_price = row["close"]  # estimate — actual entry at tomorrow open
        sl_price    = row["swing_low"] * (1 - cfg["exit_rules"]["swing_low_buffer_pct"] / 100)
        sl_price    = max(sl_price, entry_price * 0.85)
        risk        = entry_price - sl_price
        rr          = cfg["trade_rules"]["reward_to_risk_ratio"]
        target      = round(entry_price + risk * rr, 2) if risk > 0 else 0

        risk_amt    = capital * risk_pct / 100
        qty_by_risk = int(risk_amt / risk) if risk > 0 else 0
        max_buy     = capital * max_pos_pct / 100
        qty_by_cap  = int(max_buy / entry_price) if entry_price > 0 else 0
        qty         = max(1, min(qty_by_risk, qty_by_cap))
        buy_value   = round(qty * entry_price, 2)

        ranked.append({
            **sig,
            "rank_score":  score,
            "entry_est":   round(entry_price, 2),
            "sl":          round(sl_price, 2),
            "target":      target,
            "risk_per_sh": round(risk, 2),
            "qty":         qty,
            "buy_value":   round(buy_value, 2),
            "rsi":         round(float(row.get("rsi", 0)), 1),
            "atr":         round(float(row.get("atr", 0)), 2),
        })

    return sorted(ranked, key=lambda x: x["rank_score"], reverse=True)


# ─── CSV SIGNAL LOG ───────────────────────────────────────────────────────────

def save_signals_to_csv(ranked: list, mkt: dict, cfg: dict, rr: float, capital: float):
    """
    Appends today's signals to signals_log.csv.
    Creates file with headers if it doesn't exist.
    One row per signal per scan — never overwrites existing rows.
    Duplicate check: same symbol + same scan_date = skip.

    Columns auto-filled by scanner:
      scan_date, scan_time, entry_date, symbol, rank_score,
      entry_price_est, stop_loss, sl_pct_below, target_price,
      target_pct_above, rr_ratio, quantity, buy_value,
      buy_value_pct_cap, max_risk_amt, max_risk_pct_cap,
      rsi_today, atr, vix, vix_regime, nifty_close, nifty_adx, above_200ema

    Columns to fill manually after trading:
      order_placed, actual_entry_price, actual_entry_date,
      exit_price, exit_date, exit_reason, actual_pnl, notes
    """
    csv_path  = Path("signals_log.csv")
    scan_date = datetime.now().strftime("%Y-%m-%d")
    scan_time = datetime.now().strftime("%H:%M")

    # Next trading day (skip weekends)
    entry_dt = datetime.now() + timedelta(days=1)
    while entry_dt.weekday() >= 5:
        entry_dt += timedelta(days=1)
    entry_date = entry_dt.strftime("%Y-%m-%d")

    rows = []
    for sig in ranked:
        risk_total  = sig["risk_per_sh"] * sig["qty"]
        sl_pct      = round((sig["entry_est"] - sig["sl"]) / sig["entry_est"] * 100, 2)
        target_pct  = round((sig["target"] - sig["entry_est"]) / sig["entry_est"] * 100, 2)
        buy_pct     = round(sig["buy_value"] / capital * 100, 2)
        risk_pct    = round(risk_total / capital * 100, 2)

        rows.append({
            # ── Auto-filled by scanner ──
            "scan_date":           scan_date,
            "scan_time":           scan_time,
            "entry_date":          entry_date,
            "symbol":              sig["symbol"],
            "rank_score":          sig["rank_score"],
            "entry_price_est":     sig["entry_est"],
            "stop_loss":           sig["sl"],
            "sl_pct_below":        sl_pct,
            "target_price":        sig["target"],
            "target_pct_above":    target_pct,
            "rr_ratio":            rr,
            "quantity":            sig["qty"],
            "buy_value":           sig["buy_value"],
            "buy_value_pct_cap":   buy_pct,
            "max_risk_amt":        round(risk_total, 2),
            "max_risk_pct_cap":    risk_pct,
            "rsi_today":           sig["rsi"],
            "atr":                 sig["atr"],
            "vix":                 mkt.get("vix", ""),
            "vix_regime":          mkt.get("vix_regime", ""),
            "nifty_close":         mkt.get("nifty_close", ""),
            "nifty_adx":           mkt.get("adx", ""),
            "above_200ema":        mkt.get("above_200ema", ""),
            # ── Fill manually after trade ──
            "order_placed":        "",   # YES / NO / SKIP
            "actual_entry_price":  "",   # actual fill price
            "actual_entry_date":   "",   # date order filled
            "exit_price":          "",   # price when exited
            "exit_date":           "",   # date exited
            "exit_reason":         "",   # SL / target / manual
            "actual_pnl":          "",   # net profit/loss
            "notes":               "",   # any remarks
        })

    if not rows:
        return

    df_new = pd.DataFrame(rows)

    if csv_path.exists():
        df_old = pd.read_csv(csv_path, dtype=str)
        # Skip duplicates — same symbol already logged today
        existing_today = set(
            df_old[df_old["scan_date"] == scan_date]["symbol"].tolist()
        )
        df_new = df_new[~df_new["symbol"].isin(existing_today)]
        if df_new.empty:
            print(f"\n  📋 Already logged today ({scan_date}) — no duplicates added")
            return
        df_out = pd.concat([df_old, df_new.astype(str)], ignore_index=True)
    else:
        df_out = df_new

    df_out.to_csv(csv_path, index=False)
    print(f"\n  📋 {len(rows)} signal(s) saved → {csv_path.resolve()}")
    print(f"     Open signals_log.csv in Excel to track your trades")
    print(f"     Fill in: order_placed, actual_entry_price, exit details")


# ─── MAIN SCANNER ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_ltcg.json")
    args = parser.parse_args()

    print("\n" + "═"*60)
    print("  📊 LTCG Evening Scanner")
    print(f"  {datetime.now().strftime('%A, %d %b %Y  %H:%M')}")
    print("═"*60)

    # Load config
    cfg       = json.load(open(args.config))
    watchlist = json.load(open(cfg["stock_selection"]["watchlist_file"]))["stocks"]
    blacklist = set(cfg["stock_selection"].get("blacklist", []))
    symbols   = [s["symbol"] for s in watchlist if s["symbol"] not in blacklist]
    delay     = cfg["zerodha"]["request_delay_sec"]
    min_score = cfg["stock_selection"]["ranking"]["min_rank_score"]
    top_n     = cfg["stock_selection"]["ranking"]["top_n_stocks"]
    vcfg      = cfg.get("volatility_filter", {})

    # Login
    enctoken, user_id = login_and_get_enctoken()
    save_enctoken(enctoken, args.config)

    # Date range for download
    warmup_days  = cfg["backtest"].get("warmup_days", 250)
    today_str    = datetime.now().strftime("%Y-%m-%d")
    warmup_start = (datetime.now() - timedelta(days=warmup_days + 30)).strftime("%Y-%m-%d")

    print(f"  Downloading data ({warmup_start} → {today_str})...")
    print(f"  Scanning {len(symbols)} stocks\n")

    # ── Download Nifty 50 + VIX ──
    nifty_df = vix_df = None

    print("  ↓ NIFTY50 ...", end=" ", flush=True)
    nifty_raw = fetch_historical(256265, warmup_start, today_str, enctoken, delay)
    if not nifty_raw.empty:
        nifty_df = nifty_raw
        print(f"✓ ({len(nifty_df)} candles)")
    else:
        print("⚠ failed")

    print("  ↓ INDIAVIX ...", end=" ", flush=True)
    vix_raw = fetch_historical(264969, warmup_start, today_str, enctoken, delay)
    if not vix_raw.empty:
        vix_df = vix_raw
        print(f"✓ ({len(vix_df)} candles)")
    else:
        print("⚠ failed")

    # ── Market filter check ──
    mkt = check_market_filters(nifty_df, vix_df, cfg)

    print()
    print("─"*60)
    print("  MARKET STATUS")
    print("─"*60)
    vix_icon  = "🔴" if mkt["vix_regime"]=="no_entry" else ("🟡" if mkt["vix_regime"]=="caution" else "🟢")
    ema50_icon = "🟢" if mkt.get("above_50ema")  else "🔴"
    ema200_icon= "🟢" if mkt.get("above_200ema") else "🔴"
    adx_icon   = "🟢" if mkt.get("adx_ok")       else "🔴"

    print(f"  {vix_icon} India VIX      : {mkt['vix']} ({mkt['vix_regime'].upper()}) — threshold {vcfg.get('vix_no_entry_above',20)}")
    if mkt.get("nifty_close"):
        print(f"  {ema50_icon} Nifty 50 EMA  : {mkt['nifty_close']} vs 50EMA {mkt['nifty_ema50']} ({'ABOVE' if mkt['above_50ema'] else 'BELOW'})")
        print(f"  {ema200_icon} Nifty 200 EMA : {mkt['nifty_close']} vs 200EMA {mkt['nifty_ema200']} ({'ABOVE' if mkt['above_200ema'] else 'BELOW'})")
        print(f"  {adx_icon} Nifty ADX     : {mkt['adx']} ({'TRENDING ≥25' if mkt['adx_ok'] else 'WEAK <25'})")

    if not mkt["can_trade"]:
        print()
        print("  ⛔ NO NEW ENTRIES TODAY")
        reasons = []
        if mkt["vix_regime"] == "no_entry":
            reasons.append(f"VIX too high ({mkt['vix']} ≥ {vcfg.get('vix_no_entry_above',20)})")
        if not mkt.get("above_50ema"):
            reasons.append("Nifty below 50 EMA")
        if not mkt.get("above_200ema"):
            reasons.append("Nifty below 200 EMA (bear market)")
        if not mkt.get("adx_ok"):
            reasons.append(f"ADX too low ({mkt['adx']} < {vcfg.get('adx_min_threshold',25)})")
        for r in reasons:
            print(f"     → {r}")
        print()
        print("  ✅ Existing open positions continue — SL/target unchanged")
        print("═" * 60 + "\n")
        return

        # ── Download all stocks ──
    print()
    print("─" * 60)
    print("  DOWNLOADING STOCKS")
    print("─" * 60)

    all_data = {}
    for sym in symbols:
        token = get_instrument_token(sym, enctoken)
        if not token:
            print(f"  ⚠ {sym}: token not found")
            continue
        print(f"  ↓ {sym:15s}", end=" ", flush=True)
        df = fetch_historical(token, warmup_start, today_str, enctoken, delay)
        if df.empty or len(df) < 60:
            print("skip (no data)")
            continue
        df = add_indicators(df, cfg)
        all_data[sym] = df
        print(f"✓ ({len(df)}d, close:{df.iloc[-1]['close']:.2f})")

    # ── Find signals ──
    signals_today = []
    today_date    = datetime.now().date()

    for sym, df in all_data.items():
        latest = df.iloc[-1]
        # Signal must be on TODAY's candle
        if str(latest["date"]) != today_str:
            # Market might still be today if date matches
            if latest["date"] != today_date:
                continue
        if not latest.get("signal", False):
            continue
        signals_today.append({"symbol": sym, "df": df, "latest": latest})

    # ── Rank and filter ──
    if signals_today and nifty_df is not None:
        ranked = rank_signals(signals_today, nifty_df, cfg)
        ranked = [r for r in ranked if r["rank_score"] >= min_score]
        ranked = ranked[:top_n]
    else:
        ranked = []

    # ── Output signals ──
    print()
    print("─"*60)
    if not ranked:
        print("  📭 NO SIGNALS TODAY")
        print()
        if mkt["vix_regime"] == "caution":
            print(f"  ⚠ VIX caution zone ({mkt['vix']}) — only high-rank signals qualify")
        print("  Market filters passed but no pullback signals fired.")
        print("  Check again tomorrow.")
    else:
        print(f"  🎯 {len(ranked)} SIGNAL(S) TODAY — Place orders for tomorrow open")
        print("─"*60)
        print(f"  {'#':<3} {'Symbol':<12} {'Score':>6} {'CMP':>8} {'Entry~':>8} {'SL':>8} {'Risk/sh':>8} {'Qty':>5} {'BuyVal':>9} {'RSI':>5}")
        print(f"  {'─'*85}")

        capital   = cfg["capital"]["initial_capital"]
        rr        = cfg["trade_rules"]["reward_to_risk_ratio"]

        for i, sig in enumerate(ranked, 1):
            caution_flag = "⚠" if mkt["vix_regime"]=="caution" else " "
            print(f"  {caution_flag}{i:<2} {sig['symbol']:<12} {sig['rank_score']:>6.3f} "
                  f"{sig['entry_est']:>8.2f} {sig['entry_est']:>8.2f} "
                  f"{sig['sl']:>8.2f} {sig['risk_per_sh']:>8.2f} "
                  f"{sig['qty']:>5} {sig['buy_value']:>9,.0f} {sig['rsi']:>5.1f}")

        print()
        print("  TRADE DETAILS:")
        print("─"*60)
        for sig in ranked:
            risk_total = sig["risk_per_sh"] * sig["qty"]
            print(f"\n  📌 {sig['symbol']}")
            print(f"     Entry tomorrow open : ~₹{sig['entry_est']:,.2f}  (place limit order at open price)")
            print(f"     Stop Loss           :  ₹{sig['sl']:,.2f}  ({(sig['entry_est']-sig['sl'])/sig['entry_est']*100:.1f}% below entry)")
            print(f"     Target (RR={rr})     :  ₹{sig['target']:,.2f}  ({(sig['target']-sig['entry_est'])/sig['entry_est']*100:.1f}% above entry)")
            print(f"     Quantity            :  {sig['qty']} shares")
            print(f"     Buy value           :  ₹{sig['buy_value']:,.0f}  ({sig['buy_value']/capital*100:.1f}% of capital)")
            print(f"     Max risk if SL hits :  ₹{risk_total:,.0f}  ({risk_total/capital*100:.1f}% of capital)")
            print(f"     RSI today           :  {sig['rsi']:.1f}  (>45 = momentum recovering)")
            print(f"     Rank score          :  {sig['rank_score']:.3f}")

        # ── Append signals to CSV log ──
        save_signals_to_csv(ranked, mkt, cfg, rr, capital)

        print()
        print("─"*60)
        print("  ⚡ ACTION CHECKLIST:")
        print("  [ ] Open Zerodha Kite")
        print("  [ ] Place LIMIT orders for tomorrow 9:15 AM open")
        print("  [ ] Set GTT (Good Till Triggered) SL orders immediately after fill")
        print(f"  [ ] Max {cfg['capital']['max_open_trades']} open positions total — check existing before ordering")
        if mkt["vix_regime"] == "caution":
            print(f"  ⚠  VIX caution ({mkt['vix']}) — only take highest ranked signal if unsure")

    # ── Today's date reminder ──
    print()
    print("─"*60)
    print("  📅 REMINDERS")
    print("─"*60)
    print(f"  Today         : {datetime.now().strftime('%d %b %Y')}")
    print(f"  Market opens  : Tomorrow 9:15 AM")
    print(f"  Strategy      : LTCG Pullback — hold winners past 366 days")
    print(f"  Capital       : ₹{capital:,} | Risk/trade: ₹{capital*cfg['capital']['risk_per_trade_pct']/100:,.0f}")
    print()
    print("  ⚠  enctoken saved to config — re-run scanner daily (expires ~8hrs)")
    print("═"*60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled.")
