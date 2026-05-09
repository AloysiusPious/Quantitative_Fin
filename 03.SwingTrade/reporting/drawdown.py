import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_drawdown(equity_df, output_path="reports/drawdown.png"):
    df = equity_df.copy()

    df["Peak"] = df["Capital"].cummax()
    df["Drawdown"] = (df["Capital"] - df["Peak"]) / df["Peak"] * 100

    max_dd = df["Drawdown"].min()

    plt.figure(figsize=(14, 4))
    plt.plot(df["Date"], df["Drawdown"], color="red", linewidth=1.5)
    plt.axhline(0, color="black", linewidth=0.8)

    plt.title(f"Drawdown Curve (Max DD: {max_dd:.2f}%)")
    plt.xlabel("Date")
    plt.ylabel("Drawdown %")
    plt.grid(True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Drawdown chart saved: {output_path}")