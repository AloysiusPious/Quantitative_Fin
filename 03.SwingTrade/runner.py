# runner.py

from utils.config_loader import load_config
from core.scanner import scan_signals
from core.portfolio import Portfolio
from execution.execution_factory import get_broker
from core.portfolio_backtest import run_portfolio_backtest
from reporting.trade_report import save_reports
from reporting.equity_curve import plot_portfolio_vs_nifty
from reporting.rolling_returns import plot_rolling_returns
from reporting.pdf_report import generate_pdf_report
from clean_up import *

import pandas as pd

def main():
    remove_directory
    # ---------------- LOAD CONFIG ----------------
    config = load_config("config.cfg")

    # ---------------- CREATE BROKER ----------------
    broker = get_broker(config)

    # ==========================================================
    # BACKTEST / PAPER MODE
    # ==========================================================
    if not config.live:
        print("Running in BACKTEST / PAPER mode")

        scanned_data = scan_signals(config)

        print(f"Scanned {len(scanned_data)} symbols for backtest period "
              f"{config.from_date.date()} → {config.to_date.date()}")

        # At this stage we only validate scanning
        # Full portfolio backtest will be added later
        # Optional: print symbols for verification
        symbols = [item["symbol"] for item in scanned_data]
        print("Symbols scanned:")
        print(symbols)
        # Stop here intentionally
        print("Scan validation completed successfully.")
        print("Running PORTFOLIO BACKTEST")

        scanned_data = scan_signals(config)

        vix_df = pd.read_csv(config.vix_csv)
        vix_df["Date"] = pd.to_datetime(vix_df["Date"])

        portfolio = Portfolio(
            initial_capital=config.initial_capital,
            max_positions=config.max_positions
        )

        trades_df = run_portfolio_backtest(
            scanned_data=scanned_data,
            vix_df=vix_df,
            portfolio=portfolio
        )
        trades_df.sort_values(
            ["Buy Date", "Symbol"],
            inplace=True
        )

        save_reports(trades_df, initial_capital=config.initial_capital)

        print("Portfolio backtest & reports completed.")

        equity_df = plot_portfolio_vs_nifty(
            trades_df=trades_df,
            nifty_csv_path="data/nifty_50/nifty_50.csv",
            initial_capital=config.initial_capital
        )
        plot_rolling_returns(equity_df, 1)
        plot_rolling_returns(equity_df, 3, "reports/rolling_returns_3y.png")
        generate_pdf_report()
        return
    # ==========================================================
    # LIVE MODE (SAFE, NON-EXECUTING)
    # ==========================================================
    print("Running in LIVE mode")

    portfolio = Portfolio(
        initial_capital=config.initial_capital,
        max_positions=config.max_positions
    )

    signals = scan_signals(config)

    for s in signals:
        if not portfolio.can_enter():
            print(f"[REJECTED] {s['symbol']} MAX_POSITION_REACHED")
            continue

        qty = int(portfolio.position_size / s["entry_price"])
        if qty <= 0:
            continue

        # Safe print-only execution
        broker.place_amo_buy(s["symbol"], qty)
        broker.place_target_sell(
            s["symbol"],
            qty,
            s["target_price"]
        )

        portfolio.enter_trade(s)

    print("LIVE cycle completed (no forced exits).")


if __name__ == "__main__":
    main()