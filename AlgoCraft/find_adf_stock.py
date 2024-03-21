# Import necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from statsmodels.tsa.stattools import adfuller
### Custom Lib ############
from yfin import *

# Read stocks from EMA_Swing.txt file
with open('nifty_50.txt', 'r') as file:
    symbols = [line.strip() for line in file if not line.startswith("#")]
# Initialize an empty DataFrame to store closing prices of stocks
df = pd.DataFrame()

# Iterate over symbols to fetch data and populate DataFrame
for symbol in symbols:
    data = yf.download(symbol + '.NS', start='2019-01-01', end='2024-01-31')
    df[symbol] = data['Close']

    stationary_stocks = []
    p_values = []

    # Iterate over the DataFrame items (symbol, data) to perform ADF test
    for symbol, data in df.items():
        result = adfuller(data)  # No need for ['Close'], as data is already the Close prices
    p_value = result[1]
    if p_value <= 0.5: # SHould be p_value <= 0.05 as per ADF, but for testing purpose iam giving 0.5
        stationary_stocks.append(symbol)
        p_values.append(p_value)
        #print(symbol + " ==> suitable for mean reversion strategy:")
        for stock, p_value in zip(stationary_stocks, p_values):
            #print(f"Stock: {stock}, p-value: {p_value:.4f}")
            print(stock)