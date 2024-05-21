import pandas as pd
import yfinance as yf
#import talib as ta
import os
import numpy as np
import configparser
import shutil
import glob
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
def create_directory(symbols_type):
    directories_to_create = [f'Reports_{from_date}_to_{to_date}', f'Charts_{from_date}_to_{to_date}',
                             f'Summary_{from_date}_to_{to_date}', f'Master_{from_date}_to_{to_date}']
    # Iterate over each directory and create it if it does not exist
    for directory in directories_to_create:
        directory_name = symbols_type + "_" + directory
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
def create_master_file(summary_dir):
    # Initialize lists to store data
    stock_names = []
    total_trades = []
    total_winning_trades = []
    total_losing_trades = []
    total_winning_trade_percentage = []
    total_losing_trade_percentage = []
    total_profit = []
    total_cumulative_return_percentage = []
    total_charges_paid_list = []
    total_profit_after_charges = []
    total_profit_percentage_after_charges = []
    total_charges_percentage_list = []

    # Iterate over the summary files for each stock
    for filename in os.listdir(summary_dir):
        if filename.endswith("_summary.csv"):
            stock_name = filename.split("_")[0]  # Extract stock name from filename
            df_summary = pd.read_csv(os.path.join(summary_dir, filename))

            # Extract data from the summary DataFrame
            stock_names.append(stock_name)
            total_trades.append(df_summary['Total Trades'].values[0])
            total_winning_trades.append(df_summary['No of Winning Trade'].values[0])
            total_losing_trades.append(df_summary['No of Losing Trade'].values[0])
            total_winning_trade_percentage.append(round(df_summary['Winning Trade Percentage'].values[0], 2))
            total_losing_trade_percentage.append(round(df_summary['Losing Trade Percentage'].values[0], 2))
            total_profit.append(round(df_summary['Total Profit'].values[0], 2))
            total_cumulative_return_percentage.append(round(df_summary['Cumulative Return Percentage'].values[0], 2))
            total_charges_paid = round(df_summary['Total Charges Paid'].values[0], 2)
            total_charges_paid_list.append(total_charges_paid)

            # Calculate profit after charges and its percentage
            profit_after_charges = round(total_profit[-1] - total_charges_paid, 2)
            total_profit_after_charges.append(profit_after_charges)
            profit_percentage_after_charges = round((profit_after_charges / capital) * 100, 2)
            total_profit_percentage_after_charges.append(profit_percentage_after_charges)

            # Calculate total charges as a percentage of capital
            total_charges_percentage = round((total_charges_paid / capital) * 100, 2)
            total_charges_percentage_list.append(total_charges_percentage)

    # Create the Master DataFrame
    master_df = pd.DataFrame({
        'Stock Name': stock_names,
        'Total Trades': total_trades,
        'No of Winning Trade': total_winning_trades,
        'No of Losing Trade': total_losing_trades,
        'Winning Trade Percentage': total_winning_trade_percentage,
        'Losing Trade Percentage': total_losing_trade_percentage,
        'Total Profit': total_profit,
        'Total Cumulative Return Percentage': total_cumulative_return_percentage,
        'Total Charges Paid': total_charges_paid_list,
        'Profit After Charges': total_profit_after_charges,
        'Profit Percentage After Charges': total_profit_percentage_after_charges,
        'Total Charges Percentage': total_charges_percentage_list
    })

    # Calculate overall totals
    overall_totals = {
        'Stock Name': 'Overall',
        'Total Trades': sum(total_trades),
        'No of Winning Trade': sum(total_winning_trades),
        'No of Losing Trade': sum(total_losing_trades),
        'Winning Trade Percentage': round((sum(total_winning_trades) / sum(total_trades)) * 100, 2),
        'Losing Trade Percentage': round((sum(total_losing_trades) / sum(total_trades)) * 100, 2),
        'Total Profit': round(sum(total_profit), 2),
        'Total Cumulative Return Percentage': round((sum(total_profit) / capital) * 100, 2),
        'Total Charges Paid': round(sum(total_charges_paid_list), 2),
        'Profit After Charges': round(sum(total_profit_after_charges), 2),
        'Profit Percentage After Charges': round((sum(total_profit_after_charges) / capital) * 100, 2),
        'Total Charges Percentage': round((sum(total_charges_paid_list) / capital) * 100, 2)
    }

    # Append overall totals to the Master DataFrame
    master_df = pd.concat([master_df, pd.DataFrame(overall_totals, index=[0])], ignore_index=True)

    # Save Master DataFrame to CSV
    master_df.to_csv(f"{Master_Dir}/Master.csv", index=False)



def visualize_capital_and_drawdown(capital_history, drawdown_history):
    plt.figure(figsize=(12, 6))
    # Plotting the capital history
    plt.plot(capital_history, label='Capital', color='blue')
    # Plotting the drawdown history
    plt.plot(drawdown_history, label='Drawdown', color='red')
    plt.xlabel('Time')
    plt.ylabel('Amount')
    plt.title(f'{stock} Capital and Drawdown Over Time')
    plt.legend()
    plt.savefig(f'{Charts_Dir}/capital_drawdown.png')
    plt.close()
def visualize(data, target_col, stop_loss_col, stock, Charts_Dir):
    plt.figure(figsize=(12, 6))
    # Plotting the close price
    plt.plot(data.index, data['Close'], label='Close Price', color='black')
    # Plotting the buy signals
    if 'Buy Signal' in data.columns:
        plt.scatter(data.index, data['Buy Signal'], color='green', marker='^', label='Buy Signal')
    # Plotting the target levels
    if target_col in data.columns:
        plt.scatter(data.index, data[target_col], color='blue', marker='o', label='Target')
    # Plotting the stop loss levels
    if stop_loss_col in data.columns:
        plt.scatter(data.index, data[stop_loss_col], color='red', marker='o', label='Stop Loss')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'{stock} Chart with Buy Signals, Target, and Stop Loss')
    plt.legend()

    plt.savefig(f'{Charts_Dir}/{stock}_plot.png')
    plt.close()

def remove_directory():
    directories_to_remove = ["Reports", "Charts", "Summary", "Master"]
    for directory in directories_to_remove:
        for dir_path in glob.glob(f'*{directory}*'):
            """Remove directory if it exists"""
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                print(f"Directory '{dir_path}' removed successfully.")
            else:
                print(f"Directory '{dir_path}' not found.")
def volume_increase(data,i):
    # Calculate the 20-day moving average of the volume for the entire DataFrame
    data['20d_avg_volume'] = data['Volume'].rolling(window=20).mean()

    # Check if the specified day's volume is at least 50% higher than the 20-day average
    if i >= 19 and data.iloc[i]['Volume'] > (data.iloc[i]['20d_avg_volume'] * 1.50):
        return True
    else:
        return False
def volume_increase_and_open_below_yday_close(data, i):
    # Ensure there's enough data for the calculations
    if i < 19:
        return False

    # Calculate the 20-day moving average of the volume for the entire DataFrame
    data['20d_avg_volume'] = data['Volume'].rolling(window=20).mean()

    # Check if yesterday's volume was at least 50% higher than the 20-day average
    if data.iloc[i - 1]['Volume'] > (data.iloc[i - 1]['20d_avg_volume'] * 1.50):
        # Check if today's open is less than yesterday's close
        if data.iloc[i]['Open'] < data.iloc[i - 1]['Close']:
            return True

    return False
def volume_increase_and_retracement(data, i):
    # Ensure there's enough data for the calculations
    if i < 19:
        return False
    # Calculate the 20-day moving average of the volume for the entire DataFrame
    data['20d_avg_volume'] = data['Volume'].rolling(window=20).mean()
    # Check for high volume in one of the past 10 days and if it closed above the 7-day EMA
    high_volume_day = -1
    for j in range(i - 10, i):
        if data.iloc[j]['Volume'] > (data.iloc[j]['20d_avg_volume'] * 1.50) and data.iloc[j]['Close'] > data.iloc[j]['EMA_7']:
            high_volume_day = j
            break
    if high_volume_day == -1:
        return False
    # Calculate today's retracement
    today_open = data.iloc[i]['Open']
    today_low = data.iloc[i]['Low']
    today_high = data.iloc[i]['High']
    # Calculate the high volume candle's retracement level
    high_volume_open = data.iloc[high_volume_day]['Open']
    high_volume_low = data.iloc[high_volume_day]['Low']
    high_volume_high = data.iloc[high_volume_day]['High']
    if high_volume_high == high_volume_low:
        return False
    high_volume_retracement_level = high_volume_low + 0.1 * (high_volume_high - high_volume_low)
    # Check if today's retracement is equal to or more than 50% of the high volume candle
    if today_low <= high_volume_retracement_level:
        return True
    return False
def convert_col_digit(data,column):
    for col in column:
        data[col] = data[col].apply(round_to_nearest_five_cents)
    return data
def round_to_nearest_five_cents(value):
    """Rounds the value to the nearest multiple of 0.05 and formats to two decimal places."""
    rounded_value = np.round(value / 0.05) * 0.05
    formatted_value = np.format_float_positional(rounded_value, precision=2, trim='-')
    return float(formatted_value)
# Define function to fetch Yahoo Finance data
def fetch_yahoo_finance_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    col = ['Open', 'High', 'Low', 'Close', 'Adj Close']
    data = convert_col_digit(data, col)
    return data
# Define function to fetch Yahoo Finance data
# Define function to calculate EMA using pandas
def calculate_ema(data, ema_period):
    data['EMA_' + str(ema_period)] = data['Close'].rolling(window=ema_period).mean()
    return data

# Define function to check buying conditions and track trades
def pd_rsi_below_n(filtered_df, i, window = 14, n = 30):
    # Calculate RSI with a period of 14 days
    delta = filtered_df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    # Check if RSI is less than n
    rsi_below = rsi.iloc[i] < n
    return rsi_below
def pd_rsi_above_n(filtered_df, i, window = 14, n = 30):
    # Calculate RSI with a period of 14 days
    delta = filtered_df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    # Check if RSI is less than n
    rsi_below = rsi.iloc[i] > n
    return rsi_below
def ta_rsi_above_n(filtered_df, i, n):
    ################## RSI - Begin ################
    # Calculate RSI with a period of 14 days
    filtered_df['RSI'] = ta.RSI(filtered_df['Close'])
    # Check if RSI is less than 32
    filtered_df[f'RSI_Less_{n}'] = filtered_df['RSI'].iloc[i] < n
    return filtered_df[f'RSI_Less_{n}'].iloc[i]
def pd_rsi_cross_n(filtered_df, i, window = 14, n = 30):
    # Calculate RSI with a period of 14 days
    delta = filtered_df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # Check if RSI crosses n
    rsi_cross = (rsi.iloc[i - 2] < n) and (rsi.iloc[i] > n)
    return rsi_cross

def visualize(data, target_col, stop_loss_col, stock, Charts_Dir):
    plt.figure(figsize=(12, 6))
    # Plotting the close price
    plt.plot(data.index, data['Close'], label='Close Price', color='black')
    # Plotting the buy signals
    if 'Buy Signal' in data.columns:
        plt.scatter(data.index, data['Buy Signal'], color='green', marker='^', label='Buy Signal')
    # Plotting the target levels
    if target_col in data.columns:
        plt.scatter(data.index, data[target_col], color='blue', marker='o', label='Target')
    # Plotting the stop loss levels
    if stop_loss_col in data.columns:
        plt.scatter(data.index, data[stop_loss_col], color='red', marker='o', label='Stop Loss')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(f'{stock} Chart with Buy Signals, Target, and Stop Loss')
    plt.legend()

    plt.savefig(f'{Charts_Dir}/{stock}_plot.png')
    plt.close()

def remove_directory():
    directories_to_remove = ["Reports", "Charts", "Summary", "Master"]
    for directory in directories_to_remove:
        for dir_path in glob.glob(f'*{directory}*'):
            """Remove directory if it exists"""
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                print(f"Directory '{dir_path}' removed successfully.")
            else:
                print(f"Directory '{dir_path}' not found.")
def citadel(data):
    data['HL_avg'] = data['High'].rolling(window=25).mean() - data['Low'].rolling(window=25).mean()
    data['IBS'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'])
    data['Band'] = data['High'].rolling(window=25).mean() - (data['HL_avg'] * 2.25)
    # Trading strategy simulation
    for i in range(25, len(data)):
        if data.iloc[i]['Close'] < data.iloc[i]['Band'] and data.iloc[i]['IBS'] < 0.6:
            return True


# Replace the existing volume_increase function with the nr7_breakout function
def nr7_breakout(data, i):
    # Ensure there are enough data points to compare
    if i < 6:
        return False

    # Check if the current day is an NR-7 day
    current_range = data.iloc[i]['High'] - data.iloc[i]['Low']
    past_7_ranges = [data.iloc[j]['High'] - data.iloc[j]['Low'] for j in range(i - 6, i + 1)]

    if current_range != min(past_7_ranges):
        return False

    # Calculate the 20-day moving average of the volume for the entire DataFrame
    data['20d_avg_volume'] = data['Volume'].rolling(window=20).mean()

    # Check if the specified day's volume is at least 50% higher than the 20-day average
    if data.iloc[i]['Volume'] > (data.iloc[i]['20d_avg_volume'] * 1.50):
        return True
    else:
        return False
def volume_increase(data,i):
    # Calculate the 20-day moving average of the volume for the entire DataFrame
    data['20d_avg_volume'] = data['Volume'].rolling(window=20).mean()

    # Check if the specified day's volume is at least 50% higher than the 20-day average
    if i >= 19 and data.iloc[i]['Volume'] > (data.iloc[i]['20d_avg_volume'] * 1.50):
        return True
    else:
        return False
def convert_col_digit(data,column):
    for col in column:
        data[col] = data[col].apply(round_to_nearest_five_cents)
    return data
def round_to_nearest_five_cents(value):
    """Rounds the value to the nearest multiple of 0.05 and formats to two decimal places."""
    rounded_value = np.round(value / 0.05) * 0.05
    formatted_value = np.format_float_positional(rounded_value, precision=2, trim='-')
    return float(formatted_value)

# Define function to fetch Yahoo Finance data
def fetch_yahoo_finance_data(symbol, start_date, end_date):
    # Convert from_date to a datetime object
    from_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
    # Subtract one year
    adjusted_from_date_obj = from_date_obj.replace(year=from_date_obj.year - 1)
    # Convert the adjusted date back to a string
    start_date = adjusted_from_date_obj.strftime('%Y-%m-%d')
    data = yf.download(symbol, start=start_date, end=end_date)
    col = ['Open', 'High', 'Low', 'Close', 'Adj Close']
    data = convert_col_digit(data, col)
    return data

# Define function to calculate EMA using pandas
def calculate_ema(data, ema_period):
    data['EMA_' + str(ema_period)] = data['Close'].rolling(window=ema_period).mean()
    return data
'''
# Define function to calculate EMA
def calculate_ema(data, ema_period=200):
    data['EMA_'+str(ema_period)] = ta.EMA(data['Close'], timeperiod=ema_period)
    return data
'''
# Define function to check buying conditions and track trades
def pd_rsi_below_n(filtered_df, i, window = 14, n = 30):
    # Calculate RSI with a period of 14 days
    delta = filtered_df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    # Check if RSI is less than n
    rsi_below = rsi.iloc[i] < n
    return rsi_below
def pd_rsi_above_n(filtered_df, i, window = 14, n = 30):
    # Calculate RSI with a period of 14 days
    delta = filtered_df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    # Check if RSI is less than n
    rsi_below = rsi.iloc[i] > n
    return rsi_below
def ta_rsi_above_n(filtered_df, i, n):
    ################## RSI - Begin ################
    # Calculate RSI with a period of 14 days
    filtered_df['RSI'] = ta.RSI(filtered_df['Close'])
    # Check if RSI is less than 32
    filtered_df[f'RSI_Less_{n}'] = filtered_df['RSI'].iloc[i] < n
    return filtered_df[f'RSI_Less_{n}'].iloc[i]


def pd_rsi_cross_n(filtered_df, i, window = 14, n = 30):
    # Calculate RSI with a period of 14 days
    delta = filtered_df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # Check if RSI crosses n
    rsi_cross = (rsi.iloc[i - 2] < n) and (rsi.iloc[i] > n)
    return rsi_cross


def macd_cross(data, i):
    # Calculate the 12-day and 26-day EMAs
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()

    # Calculate the MACD line
    data['MACD'] = data['EMA_12'] - data['EMA_26']

    # Calculate the signal line (9-day EMA of the MACD line)
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # Check if there is a MACD crossover at the specified index
    if i >= 1:  # Ensure there's a previous day to compare
        if data['MACD'].iloc[i] > data['Signal'].iloc[i] and data['MACD'].iloc[i - 1] <= data['Signal'].iloc[i - 1]:
            return "buy"
        elif data['MACD'].iloc[i] < data['Signal'].iloc[i] and data['MACD'].iloc[i - 1] >= data['Signal'].iloc[i - 1]:
            return "sell"

    # Return None if no crossover is detected
    return None

def check_buy_conditions(data, capital, capital_per_stock, target_percentage, stop_loss_percentage,):
    #total_charges_percentage = 0.2
    trade = None
    trades = []
    total_charges_paid = 0
    ###
    capital_history = [capital]
    drawdown = 0
    drawdown_history = [drawdown]
    for i in range(100, len(data)):  # Start from the 100th day to have enough data for calculations
        if not trade and data.index[i].date() >= datetime.strptime(from_date, '%Y-%m-%d').date():
            is_previous_red = (data.iloc[i - 1]['Close'] < data.iloc[i - 1]['Open'])
            is_previous_three_red = (data.iloc[i - 1]['Close'] < data.iloc[i - 1]['Open']) and (
                    data.iloc[i - 2]['Close'] < data.iloc[i - 2]['Open']) and (
                                            data.iloc[i - 3]['Close'] < data.iloc[i - 3]['Open'])
            is_current_green = (data.iloc[i]['Close'] > data.iloc[i]['Open'])
            is_current_close_above_previous_open = (data.iloc[i]['Close'] > data.iloc[i - 1]['Open'])
            is_close_above_200EMA = data.iloc[i]['Close'] > data.iloc[i]['EMA_200']
            is_close_below_200EMA = data.iloc[i]['Close'] < data.iloc[i]['EMA_200']
            is_close_above_50EMA = data.iloc[i]['Close'] > data.iloc[i]['EMA_50']
            is_close_below_50EMA = data.iloc[i]['Close'] < data.iloc[i]['EMA_50']
            is_close_below_20EMA = data.iloc[i]['Close'] < data.iloc[i]['EMA_20']
            is_close_above_20EMA = data.iloc[i]['Close'] > data.iloc[i]['EMA_20']
            is_close_above_7EMA = data.iloc[i]['Close'] > data.iloc[i]['EMA_7']
            yday_close_above_7EMA = data.iloc[i - 1]['Close'] > data.iloc[i - 1]['EMA_7']
            is_close_below_7EMA = data.iloc[i]['Close'] < data.iloc[i]['EMA_7']
            is_50EMA_above_200EMA = data.iloc[i]['EMA_50'] > data.iloc[i]['EMA_200']
            is_50EMA_below_200EMA = data.iloc[i]['EMA_50'] < data.iloc[i]['EMA_200']
            is_20EMA_below_50EMA = data.iloc[i]['EMA_20'] < data.iloc[i]['EMA_50']
            is_current_open_less_than_previous_close = data.iloc[i]['Open'] < data.iloc[i - 1]['Close']
            rsi_above = pd_rsi_above_n(data, i, 14, 30)
            rsi_cross = pd_rsi_cross_n(data, i, 14, 30)
            rsi_below = pd_rsi_below_n(data, i, 14, 30)
            macd_signal = macd_cross(data, i)
            # Define your conditions
            sce_1 = volume_increase(data, i) and is_20EMA_below_50EMA and is_50EMA_above_200EMA and is_current_green # Good One with Compound
            sce_2 = volume_increase(data, i) and is_close_above_7EMA and is_50EMA_above_200EMA and is_current_green #***** 266.23 with no Compounding 300+ trades
            sce_3 = volume_increase(data, i) and is_close_below_7EMA and is_50EMA_above_200EMA and is_current_green  # *****  196.49 with no Compounding 230 Trades
            sce_4 = volume_increase(data, i) and data.iloc[i]['Close'] < data.iloc[i]['EMA_200'] #***** no Compounding
            sce_5 = volume_increase(data, i) and pd_rsi_below_n(data, i, 14,40) and is_current_green #***** No Compunding 285% without Green/ with Green 216%
            sce_6 = nr7_breakout(data, i) and is_20EMA_below_50EMA and is_50EMA_above_200EMA #**** 90 % Accuracy, no of Stock to Trade n/2
            sce_7 = volume_increase_and_retracement(data, i) and is_50EMA_above_200EMA
            sce_8 = volume_increase_and_open_below_yday_close(data, i) and yday_close_above_7EMA and is_50EMA_above_200EMA
            if sce_8:
                buy_date = data.index[i].date()
                bought_price = round(data.iloc[i]['Close'], 2)
                quantity_bought = int(capital_per_stock / bought_price)
                stop_loss = round_to_nearest_five_cents(bought_price * (1 - stop_loss_percentage / 100))
                target = round_to_nearest_five_cents(bought_price * (1 + target_percentage / 100))

                # Calculate charges for buying
                buy_turnover = bought_price * quantity_bought
                buy_charges = (total_charges_percentage / 100) * buy_turnover
                total_charges_paid += round_to_nearest_five_cents(buy_charges)

                trade = {
                    'Buy Date': buy_date,
                    'Bought Price': bought_price,
                    'Quantity Bought': quantity_bought,
                    'Invested Amount': capital_per_stock,
                    'Stop Loss': stop_loss,
                    'Target': target,
                    'Exited Date': None,
                    'Profit Amount': None
                }
                data.loc[data.index[i], 'Buy Signal'] = data.iloc[i]['Low'].astype(float)
                data.loc[data.index[i], 'Target Level'] = target
                data.loc[data.index[i], 'Stop Loss Level'] = stop_loss

        elif trade and (trade['Stop Loss'] >= data.iloc[i]['Low'] or trade['Target'] <= data.iloc[i]['High']):
            if trade['Target'] <= data.iloc[i]['High']:
                profit_amount = round_to_nearest_five_cents((trade['Target'] - trade['Bought Price']) * trade['Quantity Bought'])
            elif trade['Stop Loss'] >= data.iloc[i]['Low']:
                profit_amount = round_to_nearest_five_cents((trade['Stop Loss'] - trade['Bought Price']) * trade['Quantity Bought'])

            sell_date = data.index[i].date()
            sell_turnover = data.iloc[i]['Close'] * trade['Quantity Bought']

            # Calculate charges for selling
            sell_charges = (total_charges_percentage / 100) * sell_turnover
            total_charges_paid += round_to_nearest_five_cents(sell_charges)

            trade['Exited Date'] = sell_date
            trade['Profit Amount'] = round_to_nearest_five_cents(profit_amount - buy_charges - sell_charges)
            trades.append(trade)
            trade = None

            if compound:
                capital_per_stock += int(profit_amount)

        capital_history.append(capital)
        max_capital = max(capital_history)
        drawdown = (max_capital - capital) / max_capital
        drawdown_history.append(drawdown)

    if create_chart:
        visualize(data, 'Target Level', 'Stop Loss Level', stock, Charts_Dir)

    # Print total charges paid for the entire trade

    return trades, total_charges_paid


def main(symbol, start_date, end_date, capital, target_percentage, stop_loss_percentage):
    try:
        # Fetch data from Yahoo Finance
        data = fetch_yahoo_finance_data(symbol, start_date, end_date)
        if data.empty:
            print(f"No data found for {symbol}. Skipping...")
            return 0, 0  # Skip this stock and return 0 charges and 0 trades

        # Calculate EMA
        data = calculate_ema(data, 200)
        data = calculate_ema(data, 50)
        data = calculate_ema(data, 20)
        data = calculate_ema(data, 7)

        # Calculate capital per stock
        capital_per_stock = round(capital / no_of_stock_to_trade, 2)

        # Check buying conditions and track trades
        trades, total_charges_paid = check_buy_conditions(data, capital, capital_per_stock, target_percentage, stop_loss_percentage)
        if not trades:
            print(f"No trades found for {symbol}. Skipping...")
            return total_charges_paid, trades  # Return charges paid, even if no trades found

        # Calculate No of holding Days
        df = pd.DataFrame(trades)
        df['Buy Date'] = pd.to_datetime(df['Buy Date'])  # Convert Buy Date to datetime format
        df['Exited Date'] = pd.to_datetime(df['Exited Date'])  # Convert Exited Date to datetime format
        df['No of holding Days'] = round((df['Exited Date'] - df['Buy Date']).dt.days, 2)  # Calculate holding days

        # Calculate Profit %
        df['Profit %'] = round((df['Profit Amount'] / df['Invested Amount']) * 100, 2)

        # Save DataFrame to CSV with Bought Price field
        df.to_csv(f"{Reports_Dir}/{symbol}_trades.csv", index=False)

        # Create summary DataFrame
        summary_df = pd.DataFrame({
            'Total Trades': [len(trades)],
            'No of Winning Trade': [
                len([trade for trade in trades if trade['Profit Amount'] and trade['Profit Amount'] > 0])],
            'No of Losing Trade': [
                len([trade for trade in trades if trade['Profit Amount'] and trade['Profit Amount'] < 0])],
            'Winning Trade Percentage': [
                round(len([trade for trade in trades if trade['Profit Amount'] and trade['Profit Amount'] > 0]) / len(
                    trades) * 100, 2)],
            'Losing Trade Percentage': [
                round(len([trade for trade in trades if trade['Profit Amount'] and trade['Profit Amount'] < 0]) / len(
                    trades) * 100, 2)],
            'Total Profit': [round(sum([trade['Profit Amount'] for trade in trades if trade['Profit Amount']]), 2)],
            'Cumulative Return Percentage': [
                round((sum([trade['Profit Amount'] for trade in trades if trade['Profit Amount']]) / capital) * 100, 2)],
            'Total Charges Paid': [total_charges_paid]
        })

        # Save summary DataFrame to CSV
        summary_df.to_csv(f"{Summary_Dir}/{symbol}_summary.csv", index=False)

        return total_charges_paid, trades  # Return the total charges paid and trades for this stock

    except Exception as e:
        print(f"Error processing {symbol}: {e}")
        return 0, []  # Return 0 charges and an empty list of trades in case of error


if __name__ == "__main__":
    # Create a ConfigParser object
    config = configparser.ConfigParser()
    # Read the cfg file
    config.read('config.cfg')
    symbols_file = str(config['trade_symbol']['symbols_file'])
    create_chart = str(config['trade_symbol']['create_chart'])
    create_chart = True if create_chart == 'true' else False
    # Access sections and keys
    from_date = str(config['time_management']['from_date'])
    to_date = str(config['time_management']['to_date'])
    capital = float(config['risk_management']['capital'])
    no_of_stock_to_trade = int(config['risk_management']['no_of_stock_to_trade'])
    compound = str(config['risk_management']['compound'])
    compound = True if compound == 'true' else False
    target_percentage = float(config['risk_management']['target_percentage'])
    stop_loss_percentage = float(config['risk_management']['stop_loss_percentage'])
    total_charges_percentage = float(config['risk_management']['charges_percentage'])

    ###
    cleanup_logs = str(config['house_keeping']['cleanup_logs'])
    cleanup_logs = True if cleanup_logs == 'true' else False
    if cleanup_logs:
        remove_directory()
    ##############
    symbols_type = symbols_file.split('.')[0]
    Reports_Dir = f'{symbols_type}_Reports_{from_date}_to_{to_date}'
    Charts_Dir = f'{symbols_type}_Charts_{from_date}_to_{to_date}'
    Summary_Dir = f'{symbols_type}_Summary_{from_date}_to_{to_date}'
    Master_Dir = f'{symbols_type}_Master_{from_date}_to_{to_date}'
    ##############
    create_directory(symbols_type)

    with open('./symbols/' + symbols_file, 'r') as file:
        stocks = [line.split('#')[0].strip() for line in file if not line.lstrip().startswith('#')]
    total_stocks = len(stocks)
    print(f"Total number of stocks: {total_stocks}")
    # Initialize lists to store summarized information
    summary_data = []

    # Initialize total charges variable
    total_charges_for_all_stocks = 0

    for count, stock in enumerate(stocks, 1):
        print(f"Processing stock {count}/{total_stocks}: {stock}")
        # Call main function with parameters and aggregate total charges
        charges_paid, trades = main(stock + ".NS", from_date, to_date, capital, target_percentage, stop_loss_percentage)
        total_charges_for_all_stocks += charges_paid

    # Call function to create the Master_no_Compound_sce_5.csv file
    create_master_file(Summary_Dir)

    # Print total charges paid for all stocks
    print(f"Total charges paid for all stocks: ₹{total_charges_for_all_stocks:.2f}")
    print("All stocks processed.")