import pandas as pd
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import yfinance as yf
import talib as ta
import plotly.graph_objects as go
from tabulate import tabulate


# Define function to fetch Yahoo Finance data
def fetch_yahoo_finance_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

# Define function to calculate RSI and buy signals
def calculate_rsi_buy_signals(data, lookback=8):
    # Calculate RSI
    data['RSI'] = ta.RSI(data['Close'], timeperiod=lookback)

    # Calculate buy signals
    data['Buy_Signal'] = (data['RSI'] > 20) & (data['RSI'].shift(1) < 20) & (data['RSI'].shift(2) > 20)

    # Assign Bought_Date based on Buy_Signal
    data.loc[data['Buy_Signal'], 'Bought_Date'] = data.loc[data['Buy_Signal']].index

    return data


# Define function to calculate stop loss and target
def calculate_stop_loss_target(data, stop_loss_period=3, target_period=15):
    # Calculate stop loss
    data['Stop_Loss'] = data['Low'].rolling(stop_loss_period).min().shift(-stop_loss_period)

    # Calculate target
    data['Target'] = data['Close'].shift(-target_period)

    return data

def plot_candlestick_with_signals(data, buy_signals, target_levels, stop_loss_levels):
    # Create candlestick chart
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close'])])

    # Mark buy signals with blue circles
    fig.add_trace(go.Scatter(x=buy_signals.index,
                             y=data.loc[buy_signals.index, 'High'] + 10,
                             mode='markers',
                             marker=dict(size=10, color='blue', symbol='circle'),
                             name='Buy Signal'))

    # Mark target levels with red circles
    fig.add_trace(go.Scatter(x=target_levels.index + pd.DateOffset(days=15),
                             y=target_levels['Target'],
                             mode='markers',
                             marker=dict(size=10, color='red', symbol='circle'),
                             name='Target'))

    # Mark stop loss levels with red circles
    fig.add_trace(go.Scatter(x=stop_loss_levels.index,
                             y=stop_loss_levels['Stop_Loss'],
                             mode='markers',
                             marker=dict(size=10, color='red', symbol='circle'),
                             name='Stop Loss'))

    # Update layout
    fig.update_layout(title='Candlestick Chart with Buy Signals, Target, and Stop Loss Levels',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      xaxis_rangeslider_visible=False,
                      xaxis=dict(tickformat='%d-%m-%Y'))

    # Show plot
    fig.show()


def main(symbol, start_date, end_date, lookback, stop_loss_period, target_period):
    # Fetch data from Yahoo Finance
    data = yf.download(symbol, start=start_date, end=end_date)
    data['RSI'] = ta.RSI(data['Close'], timeperiod=lookback)
    # Calculate RSI and buy signals
    data = calculate_rsi_buy_signals(data, lookback)

    # Calculate stop loss and target levels
    data = calculate_stop_loss_target(data, stop_loss_period, target_period)

    # Filter data to include only buy signals
    buy_signals = data[data['Buy_Signal']]

    # Filter target levels and stop loss levels based on buy signals
    target_levels = data.loc[buy_signals.index, ['Target']]
    stop_loss_levels = data.loc[buy_signals.index, ['Stop_Loss']]

    # Plot candlestick chart with signals
    plot_candlestick_with_signals(data, buy_signals, target_levels, stop_loss_levels)

    # Calculate invested amount, profit, and return
    invested_amount = 100000
    data['Exited_Date'] = data['Bought_Date'] + timedelta(days=target_period)
    data['Return_Amount'] = data['Target'] - data['Close']
    data['Return_Percentage'] = (data['Return_Amount'] / data['Close']) * 100
    data['Profit'] = data['Return_Amount'] * 100 / invested_amount

    # Print stock name in chart
    print(f"Stock Name: {symbol}")

    # Extract date from index and format as string
    data['Signal_Date'] = data.index.strftime('%Y-%m-%d')

    # Drop the timestamp from the index in the output table
    data.index = data.index.date

    # Print the modified table
    print(tabulate(data[['Close', 'Signal_Date', 'Exited_Date', 'Return_Amount', 'Profit', 'Return_Percentage']],
                   headers=['Close', 'Signal Date', 'Exited Date', 'Return Amount', 'Profit', 'Return Percentage'],
                   tablefmt='pretty'))

    # Create CSV file
    data[['Close', 'Exited_Date', 'Return_Amount', 'Return_Percentage']].to_csv(f'{symbol}_trade_results.csv', index=False)

if __name__ == "__main__":
    # Define parameters
    symbol = 'AAPL'  # Example symbol
    start_date = '2019-01-01'
    end_date = '2024-01-01'
    lookback = 8
    stop_loss_period = 3
    target_period = 10

    # Call main function with parameters
    main(symbol, start_date, end_date, lookback, stop_loss_period, target_period)
