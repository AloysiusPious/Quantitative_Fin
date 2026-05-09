# reporting/equity_curve.py

import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_portfolio_vs_nifty(
    trades_df: pd.DataFrame,
    nifty_csv_path: str,
    initial_capital: float,
    output_path: str = "reports/portfolio_vs_nifty.png"
):
    # ---------------- PREPARE TRADES ----------------
    df = trades_df.copy()
    df["Exited Date"] = pd.to_datetime(df["Exited Date"])
    df = df.sort_values("Exited Date")

    capital = initial_capital
    equity_points = []

    for _, row in df.iterrows():
        capital += row["Profit Amount"]
        equity_points.append({
            "Date": row["Exited Date"],
            "Capital": capital
        })

    equity_df = pd.DataFrame(equity_points)

    # Insert initial capital
    start_date = df["Exited Date"].min()
    equity_df = pd.concat([
        pd.DataFrame([{
            "Date": start_date,
            "Capital": initial_capital
        }]),
        equity_df
    ]).sort_values("Date").reset_index(drop=True)

    # ---------------- DRAW DOWN CALC ----------------
    equity_df["Peak"] = equity_df["Capital"].cummax()
    equity_df["Drawdown"] = (
        equity_df["Capital"] - equity_df["Peak"]
    ) / equity_df["Peak"] * 100

    # Pick 5 worst drawdowns (most negative)
    worst_dd = (
        equity_df.nsmallest(5, "Drawdown")
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # ---------------- LOAD NIFTY ----------------
    nifty = pd.read_csv(nifty_csv_path)
    nifty["Date"] = pd.to_datetime(nifty["Date"])
    nifty = nifty.sort_values("Date")

    start = equity_df["Date"].min()
    end = equity_df["Date"].max()

    nifty = nifty[(nifty["Date"] >= start) & (nifty["Date"] <= end)]

    nifty_norm = (nifty["Close"] / nifty["Close"].iloc[0]) * initial_capital

    # ---------------- PLOT ----------------
    plt.figure(figsize=(15, 8))

    # Portfolio equity
    plt.plot(
        equity_df["Date"],
        equity_df["Capital"],
        label="Portfolio Equity",
        linewidth=2
    )

    # NIFTY
    plt.plot(
        nifty["Date"],
        nifty_norm,
        label="NIFTY 50 (Normalized)",
        linestyle="--",
        alpha=0.8
    )

    # ---------------- ANNOTATIONS ----------------
    # Initial capital
    plt.annotate(
        f"Start: ₹{initial_capital:,.0f}",
        xy=(equity_df.iloc[0]["Date"], equity_df.iloc[0]["Capital"]),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="black")
    )

    # Final capital & profit %
    final_capital = equity_df.iloc[-1]["Capital"]
    profit_pct = ((final_capital - initial_capital) / initial_capital) * 100

    plt.annotate(
        f"End: ₹{final_capital:,.0f}\nProfit: {profit_pct:.2f}%",
        xy=(equity_df.iloc[-1]["Date"], final_capital),
        xytext=(10, -30),
        textcoords="offset points",
        fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="black")
    )

    # ---------------- DRAW DOWN MARKERS ----------------
    y_offsets = [30, -40, 50, -60, 70]  # prevent overlap

    for i, row in worst_dd.iterrows():
        plt.scatter(row["Date"], row["Capital"], color="red", zorder=5)

        plt.annotate(
            f"DD {abs(row['Drawdown']):.1f}%",
            xy=(row["Date"], row["Capital"]),
            xytext=(0, y_offsets[i]),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            arrowprops=dict(arrowstyle="->", lw=0.8)
        )

    # ---------------- FINAL TOUCH ----------------
    plt.title("Portfolio Equity Curve vs NIFTY 50")
    plt.xlabel("Date")
    plt.ylabel("Value (₹)")
    plt.legend()
    plt.grid(True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Equity curve saved to: {output_path}")
    return equity_df