import matplotlib.pyplot as plt


def add_trade_markers(ax, trades_df, equity_df):
    status_color = {
        "Target": "green",
        "StopLoss": "red",
        "LastDayClose": "orange"
    }

    for _, row in trades_df.iterrows():
        color = status_color.get(row["Trade Status"], "gray")

        ax.scatter(
            row["Exited Date"],
            equity_df.loc[
                equity_df["Date"] == row["Exited Date"], "Capital"
            ].values[0],
            color=color,
            s=40,
            zorder=6
        )
