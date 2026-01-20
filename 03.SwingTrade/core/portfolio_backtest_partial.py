import math
import pandas as pd
from .signal_engine import generate_signal
######### Partial Book ##########
# ---------------- CONFIG (can be moved to config.cfg later) ----------------
MAX_HOLD_DAYS = 60
PARTIAL_PROFIT_PCT = 0.08
PARTIAL_EXIT_RATIO = 0.5


def run_portfolio_backtest(scanned_data, vix_df, portfolio):
    """
    scanned_data: output of scan_backtest()
    vix_df: VIX dataframe (Date, Close)
    portfolio: Portfolio object (fixed position sizing, max cap enforced)
    """

    all_trades = []

    # ---------------- BUILD UNIFIED TIMELINE ----------------
    timeline = []
    for item in scanned_data:
        symbol = item["symbol"]
        df = item["df"]

        for i, row in df.iterrows():
            timeline.append((row["Date"], symbol, df, i))

    timeline.sort(key=lambda x: x[0])  # CRITICAL

    open_positions = {}

    # ---------------- MAIN LOOP ----------------
    for date, symbol, df, i in timeline:
        row = df.iloc[i]

        # ---------------- MAP VIX ----------------
        vix_row = vix_df[vix_df["Date"] == date]
        if vix_row.empty:
            continue

        vix_close = vix_row.iloc[0]["Close"]

        # ================= ENTRY =================
        if symbol not in open_positions and portfolio.can_enter():
            signal = generate_signal(row, vix_close)

            if signal:
                entry_price = signal["entry_price"]

                open_pos_count_at_entry = len(open_positions)

                qty = math.floor(portfolio.position_size / entry_price)
                if qty <= 0:
                    continue

                invested_amount = round(qty * entry_price, 2)

                open_positions[symbol] = {
                    "Symbol": symbol,
                    "Buy Date": date,
                    "Bought Price": entry_price,
                    "Quantity Bought": qty,
                    "Remaining Qty": qty,
                    "Invested Amount": invested_amount,
                    "Stop Loss": signal["stop_price"],
                    "Target": signal["target_price"],
                    "Partial_Exited": False,
                    "Open_Positions_At_Entry": open_pos_count_at_entry
                }

                portfolio.enter_trade(symbol)

        # ================= EXIT MANAGEMENT =================
        if symbol not in open_positions:
            continue

        trade = open_positions[symbol]
        holding_days = (date - trade["Buy Date"]).days

        # ---------------- PARTIAL PROFIT EXIT ----------------
        partial_target_price = trade["Bought Price"] * (1 + PARTIAL_PROFIT_PCT)

        if (
            not trade["Partial_Exited"]
            and row["High"] >= partial_target_price
        ):
            exit_qty = int(trade["Quantity Bought"] * PARTIAL_EXIT_RATIO)

            if exit_qty > 0:
                profit = (
                    (partial_target_price - trade["Bought Price"])
                    * exit_qty
                )

                all_trades.append({
                    "Symbol": symbol,
                    "Buy Date": trade["Buy Date"],
                    "Bought Price": trade["Bought Price"],
                    "Quantity Bought": exit_qty,
                    "Invested Amount": round(exit_qty * trade["Bought Price"], 2),
                    "Stop Loss": trade["Stop Loss"],
                    "Target": trade["Target"],
                    "Exited Date": date,
                    "Exited Price": round(partial_target_price, 2),
                    "Profit Amount": round(profit, 2),
                    "Trade Status": "Partial_Target",
                    "No of holding Days": holding_days,
                    "Profit %": round((profit / (exit_qty * trade["Bought Price"])) * 100, 2),
                    "Open_Positions_At_Entry": trade["Open_Positions_At_Entry"]
                })

                trade["Remaining Qty"] -= exit_qty
                trade["Partial_Exited"] = True

        # ---------------- FINAL EXIT (SL / TARGET / TIME) ----------------
        exit_price = None
        status = None

        if row["Low"] <= trade["Stop Loss"]:
            exit_price = trade["Stop Loss"]
            status = "StopLoss"

        elif row["High"] >= trade["Target"]:
            exit_price = trade["Target"]
            status = "Target"

        elif holding_days >= MAX_HOLD_DAYS:
            exit_price = row["Close"]
            status = "TimeExit"

        if exit_price is None or trade["Remaining Qty"] <= 0:
            continue

        profit_amount = round(
            (exit_price - trade["Bought Price"])
            * trade["Remaining Qty"],
            2
        )

        profit_pct = round(
            (profit_amount / (trade["Remaining Qty"] * trade["Bought Price"])) * 100,
            2
        )

        all_trades.append({
            "Symbol": symbol,
            "Buy Date": trade["Buy Date"],
            "Bought Price": trade["Bought Price"],
            "Quantity Bought": trade["Remaining Qty"],
            "Invested Amount": round(trade["Remaining Qty"] * trade["Bought Price"], 2),
            "Stop Loss": trade["Stop Loss"],
            "Target": trade["Target"],
            "Exited Date": date,
            "Exited Price": round(exit_price, 2),
            "Profit Amount": profit_amount,
            "Trade Status": status,
            "No of holding Days": holding_days,
            "Profit %": profit_pct,
            "Open_Positions_At_Entry": trade["Open_Positions_At_Entry"]
        })

        # Cleanup
        del open_positions[symbol]
        portfolio.exit_trade(symbol)

    return pd.DataFrame(all_trades)
