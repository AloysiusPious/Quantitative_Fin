import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_rolling_returns(
    equity_df,
    window_years=1,
    output_path="reports/rolling_returns.png"
):
    df = equity_df.copy().set_index("Date")

    window_days = int(window_years * 252)

    rolling_cagr = (
        df["Capital"]
        .pct_change(window_days)
        .apply(lambda x: ((1 + x) ** (1 / window_years) - 1) * 100)
    )

    plt.figure(figsize=(14, 4))
    plt.plot(rolling_cagr, label=f"{window_years}Y Rolling CAGR")

    plt.axhline(0, color="black", linewidth=0.8)
    plt.grid(True)
    plt.legend()
    plt.title(f"{window_years}-Year Rolling CAGR")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Rolling returns saved: {output_path}")
