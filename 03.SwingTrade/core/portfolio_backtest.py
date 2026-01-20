# core/portfolio_backtest.py

import math
import pandas as pd
from .signal_engine import generate_signal


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

    # Sort chronologically (CRITICAL)
    timeline.sort(key=lambda x: x[0])

    open_positions = {}

    # ---------------- MAIN LOOP ----------------
    for date, symbol, df, i in timeline:
        row = df.iloc[i]

        # ---------------- MAP VIX ----------------
        vix_row = vix_df[vix_df["Date"] == date]
        #print(vix_row)
        if vix_row.empty:
            continue

        vix_close = vix_row.iloc[0]["Close"]

        # ================= ENTRY =================
        if symbol not in open_positions and portfolio.can_enter():
            signal = generate_signal(row, vix_close)

            if signal:
                entry_price = signal["entry_price"]

                # 🔹 capture open positions count BEFORE entry
                open_pos_count_at_entry = len(open_positions)

                qty = math.floor(
                    portfolio.position_size / entry_price
                )

                if qty <= 0:
                    continue

                invested_amount = round(qty * entry_price, 2)

                open_positions[symbol] = {
                    "Symbol": symbol,
                    "Buy Date": date,
                    "Bought Price": entry_price,
                    "Quantity Bought": qty,
                    "Invested Amount": invested_amount,
                    "Stop Loss": signal["stop_price"],
                    "Target": signal["target_price"],
                    "Open_Positions_At_Entry": open_pos_count_at_entry
                }

                portfolio.enter_trade(symbol)

        # ================= EXIT =================
        if symbol in open_positions:
            trade = open_positions[symbol]

            exit_price = None
            status = None

            if row["High"] >= trade["Target"]:
                exit_price = trade["Target"]
                status = "Target"

            elif row["Low"] <= trade["Stop Loss"]:
                exit_price = trade["Stop Loss"]
                status = "StopLoss"

            elif i == len(df) - 1:
                exit_price = row["Close"]
                status = "LastDayClose"

            if exit_price is None:
                continue

            holding_days = (date - trade["Buy Date"]).days

            profit_amount = round(
                (exit_price - trade["Bought Price"])
                * trade["Quantity Bought"],
                2
            )

            profit_pct = round(
                (profit_amount / trade["Invested Amount"]) * 100,
                2
            )

            all_trades.append({
                "Symbol": symbol,
                "Buy Date": trade["Buy Date"],
                "Bought Price": trade["Bought Price"],
                "Quantity Bought": trade["Quantity Bought"],
                "Invested Amount": trade["Invested Amount"],
                "Stop Loss": trade["Stop Loss"],
                "Target": trade["Target"],
                "Exited Date": date,
                "Exited Price": exit_price,
                "Profit Amount": profit_amount,
                "Trade Status": status,
                "No of holding Days": holding_days,
                "Profit %": profit_pct,

                # ✅ NEW COLUMN
                "Open_Positions_At_Entry": trade["Open_Positions_At_Entry"]
            })

            # Cleanup
            del open_positions[symbol]
            portfolio.exit_trade(symbol)

    return pd.DataFrame(all_trades)
