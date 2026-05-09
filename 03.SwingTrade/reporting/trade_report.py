# reporting/trade_report.py

import os
import pandas as pd


# -------------------------------------------------
# Charges & Tax
# -------------------------------------------------

def calculate_charges_and_tax(trades_df: pd.DataFrame):
    """
    Zerodha CNC (Equity Delivery) charges for swing trades
    Assumes holding period > 1 day (CNC, not intraday)
    """

    total_charges = 0.0
    total_tax = 0.0

    for _, row in trades_df.iterrows():
        qty = row["Quantity Bought"]

        buy_value = row["Bought Price"] * qty
        sell_value = row["Exited Price"] * qty

        # -------------------------------
        # Zerodha CNC Charges
        # -------------------------------

        # 1️⃣ Brokerage (Delivery)
        brokerage = 0.0  # Zerodha CNC = ₹0

        # 2️⃣ STT (Sell side only)
        stt = sell_value * 0.001  # 0.1%

        # 3️⃣ Exchange Transaction Charges (Buy + Sell)
        exchange_charges = (buy_value + sell_value) * 0.0000325  # ~0.00325%

        # 4️⃣ GST (18% on exchange charges + brokerage)
        gst = (brokerage + exchange_charges) * 0.18

        # 5️⃣ Stamp Duty (Buy side only)
        stamp_duty = buy_value * 0.00015  # ~0.015%

        trade_charges = (
            brokerage +
            stt +
            exchange_charges +
            gst +
            stamp_duty
        )

        total_charges += trade_charges

        # -------------------------------
        # Capital Gains Tax
        # -------------------------------
        profit = row["Profit Amount"]
        holding_days = row["No of holding Days"]

        if profit > 0:
            if holding_days <= 365:
                total_tax += profit * 0.15   # STCG
            else:
                total_tax += profit * 0.10   # LTCG

    return round(total_charges, 2), round(total_tax, 2)



# -------------------------------------------------
# CAGR
# -------------------------------------------------

def calculate_cagr(trades_df: pd.DataFrame, initial_capital: float, net_profit: float):
    start_date = trades_df["Buy Date"].min()
    end_date = trades_df["Exited Date"].max()

    years = (end_date - start_date).days / 365.25
    if years <= 0:
        return 0.0

    final_capital = initial_capital + net_profit
    cagr = (final_capital / initial_capital) ** (1 / years) - 1
    return round(cagr * 100, 2)


def calculate_yearly_cagr(trades_df: pd.DataFrame, initial_capital: float):
    trades_df = trades_df.copy()
    trades_df["Year"] = trades_df["Exited Date"].dt.year

    yearly_rows = []
    capital = initial_capital

    for year in sorted(trades_df["Year"].unique()):
        df = trades_df[trades_df["Year"] == year]
        year_profit = df["Profit Amount"].sum()

        year_end_capital = capital + year_profit
        year_return = (year_end_capital / capital) - 1

        yearly_rows.append({
            "Year": year,
            "Year Profit": round(year_profit, 2),
            "Year CAGR %": round(year_return * 100, 2)
        })

        capital = year_end_capital

    return pd.DataFrame(yearly_rows)


# -------------------------------------------------
# Stock Summary
# -------------------------------------------------

def generate_stock_summary(trades_df: pd.DataFrame) -> pd.DataFrame:
    summary = []

    for symbol, df in trades_df.groupby("Symbol"):
        total_trades = len(df)
        wins = (df["Profit Amount"] > 0).sum()
        losses = (df["Profit Amount"] <= 0).sum()

        summary.append({
            "Stock Name": symbol,
            "Total Trades": total_trades,
            "No of Winning Trade": wins,
            "No of Losing Trade": losses,
            "Winning Trade Percentage": round((wins / total_trades) * 100, 2),
            "Losing Trade Percentage": round((losses / total_trades) * 100, 2),
            "Total Profit": round(df["Profit Amount"].sum(), 2)
        })

    return pd.DataFrame(summary)


# -------------------------------------------------
# Overall Summary
# -------------------------------------------------

def generate_overall_summary(trades_df: pd.DataFrame, initial_capital: float):
    # -----------------------------
    # Trade stats
    # -----------------------------
    total_trades = len(trades_df)
    winning_trades = (trades_df["Profit Amount"] > 0).sum()
    losing_trades = (trades_df["Profit Amount"] <= 0).sum()

    win_pct = round((winning_trades / total_trades) * 100, 2) if total_trades else 0.0
    loss_pct = round((losing_trades / total_trades) * 100, 2) if total_trades else 0.0

    # -----------------------------
    # Profit & cost
    # -----------------------------
    gross_profit = trades_df["Profit Amount"].sum()

    total_charges, total_tax = calculate_charges_and_tax(trades_df)
    net_profit = gross_profit - total_charges - total_tax

    # -----------------------------
    # Return metrics
    # -----------------------------
    gross_profit_pct = round((gross_profit / initial_capital) * 100, 2)
    net_profit_pct = round((net_profit / initial_capital) * 100, 2)

    cagr = calculate_cagr(trades_df, initial_capital, net_profit)

    return {
        # Capital
        "Initial Capital": round(initial_capital, 2),

        # Trade counts
        "Total Trades": total_trades,
        "Winning Trades": int(winning_trades),
        "Losing Trades": int(losing_trades),
        "Winning Trade %": win_pct,
        "Losing Trade %": loss_pct,

        # Profit
        "Gross Profit": round(gross_profit, 2),
        "Total Charges": total_charges,
        "Total Tax": total_tax,
        "Net Profit After Charges": round(net_profit, 2),

        # Returns
        "Profit % (Gross)": gross_profit_pct,
        "Profit % (Net)": net_profit_pct,
        "CAGR %": cagr
    }


# -------------------------------------------------
# Save Reports
# -------------------------------------------------

def save_reports(
    trades_df: pd.DataFrame,
    initial_capital: float,
    output_prefix: str = "reports"
):
    if trades_df.empty:
        raise ValueError("No trades to report")

    # Ensure datetime
    trades_df = trades_df.copy()
    trades_df["Buy Date"] = pd.to_datetime(trades_df["Buy Date"])
    trades_df["Exited Date"] = pd.to_datetime(trades_df["Exited Date"])

    os.makedirs(output_prefix, exist_ok=True)

    # 1️⃣ All trades
    trades_df.to_csv(f"{output_prefix}/all_trades.csv", index=False)

    # 2️⃣ Stock summary
    generate_stock_summary(trades_df).to_csv(
        f"{output_prefix}/stock_summary.csv",
        index=False
    )

    # 3️⃣ Overall summary
    pd.DataFrame([
        generate_overall_summary(trades_df, initial_capital)
    ]).to_csv(
        f"{output_prefix}/overall_summary.csv",
        index=False
    )

    # 4️⃣ Yearly CAGR
    calculate_yearly_cagr(trades_df, initial_capital).to_csv(
        f"{output_prefix}/yearly_cagr.csv",
        index=False
    )

    print("Reports generated:")
    print(f" - {output_prefix}/all_trades.csv")
    print(f" - {output_prefix}/stock_summary.csv")
    print(f" - {output_prefix}/overall_summary.csv")
    print(f" - {output_prefix}/yearly_cagr.csv")
