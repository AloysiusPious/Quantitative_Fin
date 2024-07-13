import matplotlib.pyplot as plt
import yfinance as yf
import glob
import shutil
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time
import re
def get_stock_for_date_refrence(cvs_data_dir, from_date, to_date):
    print(f'Downloading stock for Date reference..')
    nifty50_data = get_nifty50_data(from_date, to_date)
    nifty50_data.reset_index(inplace=True)
    nifty50_data.rename(columns={'index': 'Date'})
    nifty50_data = nifty50_data[['Date']]
    nifty50_data.to_csv(f"{cvs_data_dir}/stock_date_ref.csv", index=False, date_format='%Y-%m-%d')
    print(f'Ok.')
def get_nifty50_data(from_date, to_date):
    # Fetch Nifty 50 data within the specified date range
    #nifty50_data = yf.Ticker("^NSEI")
    nifty50_data = yf.download("^NSEI", start=from_date, end=to_date)
    #print(nifty50_data)
    #nifty50_data = nifty50.history(start=from_date, end=to_date)
    return nifty50_data
def fetch_yahoo_finance_data(symbol, start_date, end_date):
    # Convert from_date to a datetime object
    from_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
    # Subtract one year
    adjusted_from_date_obj = from_date_obj.replace(year=from_date_obj.year - 1)
    # Convert the adjusted date back to a string
    start_date = adjusted_from_date_obj.strftime('%Y-%m-%d')
    data = yf.download(symbol, start=start_date, end=end_date)
    #print(data)
    col = ['Open', 'High', 'Low', 'Close', 'Adj Close']

    return data
def create_directory(symbols_type, from_date, to_date):
    directories_to_create = [f'Reports_{from_date}_to_{to_date}', f'Charts_{from_date}_to_{to_date}',
                             f'Summary_{from_date}_to_{to_date}', f'Master_{from_date}_to_{to_date}', f'Cvs_Data_{from_date}_to_{to_date}', f'Raw_Data_{from_date}_to_{to_date}']
    # Iterate over each directory and create it if it does not exist
    for directory in directories_to_create:
        directory_name = symbols_type + "_" + directory
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
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
def visualize_capital_and_drawdown(stock, Charts_Dir, capital_history, drawdown_history):
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
def round_to_nearest_0_05(value):
    return round(round(value * 20) / 20, 2)
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


def yday_unusual_volume(data, i):
    # Ensure there's enough data for the calculations
    if i < 20:
        return False

    # Calculate the 20-day moving average of the volume for the entire DataFrame
    data['20d_avg_volume'] = data['Volume'].rolling(window=20).mean()

    # Check if yesterday's volume was at least 50% higher than the 20-day average up to 1 day ago
    if data.iloc[i - 1]['Volume'] > data.iloc[i - 1]['20d_avg_volume'] * 1.5 and data.iloc[i - 1]['Volume'] > data.iloc[i - 2]['Volume']:
        return True

    return False
def yday_unusual_volume_old(data, i):
    # Ensure there's enough data for the calculations
    if i < 19:
        return False
    # Calculate the 20-day moving average of the volume for the entire DataFrame
    data['20d_avg_volume'] = data['Volume'].rolling(window=20).mean()
    volume_avg = data.iloc[i - 20:i]['Volume'].mean()
    # Check if yesterday's volume was at least 50% higher than the 20-day average
    if volume_avg * 1.5 < data.iloc[i - 1]['Volume'] and data.iloc[i - 1]['Volume'] > data.iloc[i - 2]['Volume']:
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
def convert_specific_col_digit(data,column):
    for col in column:
        data[col] = data[col].apply(round_to_nearest_five_cents)
    return data
def get_ref_stock_date(from_date, to_date):
    ref_stock_data = yf.download("TCS.NS", start=from_date, end=to_date)
    print(f'{cvs_data_dir} Data not found., Downloading...')
    if not os.path.exists(cvs_data_dir):
        os.makedirs(cvs_data_dir)
    ref_stock_data.reset_index(inplace=True)
    ref_stock_data.rename(columns={'index': 'Date'})
    ref_stock_data = ref_stock_data[['Date']]
    ref_stock_data.to_csv(f"{cvs_data_dir}/stock_date_ref.csv", index=False, date_format='%Y-%m-%d')
def convert_all_col_digit(data):
    for col in data.columns:
        if col != 'Date' and data[col].dtype != 'object':  # Check if column is not string/object type
            data.loc[:, col] = data[col].apply(round_to_nearest_five_cents)
    return data
def extract_date_range_from_filename(filename):
    match = re.search(r'Master_(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})\.csv', filename)
    if match:
        return match.group(1), match.group(2)
    return None, None
def round_to_nearest_five_cents(value):
    """Rounds the value to the nearest multiple of 0.05 and formats to two decimal places."""
    rounded_value = np.round(value / 0.05) * 0.05
    formatted_value = np.format_float_positional(rounded_value, precision=2, trim='-')
    return float(formatted_value)
# Define function to fetch Yahoo Finance data
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

def visualize(data, from_date, to_date, target_col, stop_loss_col, stock, Charts_Dir):
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

    plt.savefig(f'{Charts_Dir}/{stock}_{from_date}_to_{to_date}_plot.png')
    plt.close()

def remove_directory():
    #directories_to_remove = ["Reports", "Charts", "Summary", "Master"]
    directories_to_remove = ["Reports", "Charts", "Summary", "Master", "Cvs_Data"]
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


def yday_close_less_than_highest_close(data, i):
    if i < 1:  # Ensure there is at least one previous day to compare
        return False
    # Calculate the 7-day EMA
    data['7EMA'] = data['Close'].ewm(span=7, adjust=False).mean()
    # Check if index is not integer-based and reset it if necessary
    if isinstance(data.index, pd.DatetimeIndex):
        data = data.reset_index()
    # Find the previous highest close which was above the 7EMA
    previous_highs = data.loc[:i-1]
    highest_close_above_7ema = previous_highs[previous_highs['Close'] > previous_highs['7EMA']]['Close'].max()
    if pd.isna(highest_close_above_7ema):  # Check if there is no previous high above 7EMA
        return False
    # Calculate 5% less than the highest close
    target_value = highest_close_above_7ema * 0.95
    # Check if yesterday's close is 5% less than the highest close
    if data.loc[i, 'Close'] < target_value:
        return True
    else:
        return False

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


def analyze_csv_files(provided_date, no_of_stock_to_trade, Reports_Dir):
    open_positions = 0
    csv_files = [f for f in os.listdir(Reports_Dir) if f.endswith('.csv')]

    if len(csv_files) < no_of_stock_to_trade:
        return 0, True

    for csv_file in csv_files:
        file_path = os.path.join(Reports_Dir, csv_file)
        data = pd.read_csv(file_path)

        # Convert dates to datetime for comparison
        data['Buy Date'] = pd.to_datetime(data['Buy Date'])
        data['Exited Date'] = pd.to_datetime(data['Exited Date'])
        provided_date = pd.to_datetime(provided_date)

        # Check for open positions
        for i, row in data.iterrows():
            if row['Buy Date'] <= provided_date and (pd.isna(row['Exited Date']) or row['Exited Date'] > provided_date):
                open_positions += 1
    if open_positions > no_of_stock_to_trade:
        print(f'Current Open Position more then {no_of_stock_to_trade} : {open_positions}')
    return open_positions, open_positions < no_of_stock_to_trade