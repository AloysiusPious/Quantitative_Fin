import os
import pandas as pd
import matplotlib.pyplot as plt

def read_and_process_csv(filepath):
    df = pd.read_csv(filepath, parse_dates=['Buy Date', 'Exited Date'])
    return df

def process_all_files(directory):
    all_trades = []

    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            df = read_and_process_csv(filepath)
            all_trades.append(df)

    return pd.concat(all_trades, ignore_index=True)

def plot_profit_trend(all_trades, initial_capital):
    # Ensure trades are sorted by 'Buy Date'
    all_trades = all_trades.sort_values(by='Buy Date')

    # Calculate cumulative profit
    all_trades['Cumulative Profit'] = all_trades['Profit Amount'].cumsum()

    # Calculate capital over time
    all_trades['Capital'] = initial_capital + all_trades['Cumulative Profit']

    plt.figure(figsize=(14, 7))
    plt.plot(all_trades['Buy Date'], all_trades['Capital'], marker='.', linestyle='-', color='b', label='Capital Over Time')
    # Plot the line
    #plt.plot(all_trades['Buy Date'], all_trades['Capital'], linestyle='-', color='b', label='Capital Over Time')

    # Plot the markers
    #plt.scatter(all_trades['Buy Date'], all_trades['Capital'], color='r', marker='.', label='Buy Points')

    # Annotate initial capital and final capital
    plt.annotate(f'Start: ₹{initial_capital}', xy=(all_trades['Buy Date'].iloc[0], initial_capital), xytext=(all_trades['Buy Date'].iloc[0], initial_capital),
                 arrowprops=dict(facecolor='green', shrink=0.05))
    plt.annotate(f'End: ₹{all_trades["Capital"].iloc[-1]:.2f}', xy=(all_trades['Buy Date'].iloc[-1], all_trades['Capital'].iloc[-1]), xytext=(all_trades['Buy Date'].iloc[-1], all_trades['Capital'].iloc[-1]),
                 arrowprops=dict(facecolor='red', shrink=0.05))
    plt.title('Capital Growth Over Time')
    plt.xlabel('Date')
    plt.ylabel('Capital (₹)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'capital_drawdown.png')
    plt.close()
    plt.show()

def main(directory, initial_capital):
    all_trades = process_all_files(directory)
    plot_profit_trend(all_trades, initial_capital)

if __name__ == "__main__":
    directory = 'nifty_100_Reports_2016-01-01_to_2020-12-31'  # Change this to your directory containing the .csv files
    initial_capital = 1000000  # Initial capital
    main(directory, initial_capital)