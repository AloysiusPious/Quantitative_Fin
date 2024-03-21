import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from backtesting import Backtest, Strategy
### Custom Lib ############
from yfin import *
from bokeh.plotting import figure
from bokeh.models import DatetimeTickFormatter


# Create a DatetimeTickFormatter with a single format string
formatter = DatetimeTickFormatter(
    seconds=["%Y"],
    minsec=["%Y"],
    minutes=["%b %Y"],
    hourmin=["%b %Y"],
    hours=["%b %Y"],
    days=["%d %b %Y"],
    months=["%b %Y"],
    years=["%Y"]
)


class MeanReversion(Strategy):
    n1 = 30  # Period for the moving average
    offset = 0.01  # Buy/sell when price is 1% below/above the moving average

    def init(self):
        # Compute moving average
        self.ma = self.I(self.compute_rolling_mean, self.data['Close'], self.n1)

    def compute_rolling_mean(self, prices, window):
        if not isinstance(prices, pd.DataFrame):
            prices = pd.DataFrame(prices, columns=['Close'])  # Convert to DataFrame if not already

        return prices.rolling(window=window).mean()

    def next(self):
        size = 0.1
        # If price drops to more than offset% below n1-day moving average, buy
        if self.data['Close'] < self.ma[-1] * (1 - self.offset):
            if self.position.size < 0:  # Check for existing short position
                self.buy()  # Close short position
            self.buy(size=size)

        # If price rises to more than offset% above n1-day moving average, sell
        elif self.data['Close'] > self.ma[-1] * (1 + self.offset):
            if self.position.size > 0:  # Check for existing long position
                self.sell()  # Close long position
            self.sell(size=size)


def plot_stationary_stocks(df, stock):
    data = df[stock]

    # Calculate rolling statistics
    rolling_mean = data.rolling(window=30).mean()  # 30-day rolling mean
    rolling_std = data.rolling(window=30).std()  # 30-day rolling standard deviation

    # Plot the statistics
    plt.figure(figsize=(12, 6))
    plt.plot(data, label=f'{stock} Prices', color='blue')
    plt.plot(rolling_mean, label='Rolling Mean', color='red')
    plt.plot(rolling_std, label='Rolling Std. Dev.', color='black')
    plt.title(f'Stationarity Check for {stock}')
    plt.xlabel('Date')
    plt.ylabel('Prices')
    plt.legend()
    plt.grid(True)
    plt.show()

# Read stocks from adf_stocks.txt file
with open('adf_stocks.txt', 'r') as file:
    symbols = [line.strip() for line in file if not line.startswith("#")]

# Initialize an empty DataFrame to store closing prices of stocks
data = {}
# Iterate over symbols to fetch data and populate DataFrame
for symbol in symbols:
    data[symbol] = yf.download(symbol + '.NS', start='2019-01-01', end='2024-01-31')
    # Select a symbol for backtesting
    stock_to_backtest = symbols[0]

    df_backtest = pd.DataFrame(data[stock_to_backtest])  # DataFrame with OHLCV data
    #print(df_backtest)
    bt = Backtest(df_backtest, MeanReversion, cash=100000, commission=.002)
    stats = bt.run()
    # Apply the formatter to the x-axis
    # Create a figure
    p = figure(x_axis_type="datetime")
    p.xaxis.formatter = formatter
    bt.plot()
    print(stats)