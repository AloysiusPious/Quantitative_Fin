import pandas as pd
import yfinance as yf
#import talib as ta
import os
import numpy as np
import configparser
import shutil
import glob
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, time
import re
def get_nifty50_data():
    # Fetch Nifty 50 data within the specified date range
    nifty50 = yf.Ticker("^NSEI")
    nifty50_data = nifty50.history(start=from_date, end=to_date)
    return nifty50_data


def draw_down_chart():
    all_trades = []
    for filename in os.listdir(Reports_Dir):
        if filename.endswith(".csv"):
            filepath = os.path.join(Reports_Dir, filename)
            df = pd.read_csv(filepath, parse_dates=['Buy Date', 'Exited Date'])
            all_trades.append(df)
    all_trades = pd.concat(all_trades, ignore_index=True)
    all_trades = all_trades.sort_values(by='Buy Date')

    # Calculate cumulative profit
    all_trades['Cumulative Profit'] = all_trades['Profit Amount'].cumsum()

    # Calculate capital over time
    all_trades['Capital'] = capital + all_trades['Cumulative Profit']

    # Fetch Nifty 50 data within the specified date range
    nifty50_data = get_nifty50_data()

    # Calculate the percentage increase
    final_capital = all_trades['Capital'].iloc[-1]
    percentage_increase = ((final_capital - capital) / capital) * 100

    # Create a figure and a set of subplots
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Plotting the capital growth
    ax1.plot(all_trades['Buy Date'], all_trades['Capital'], marker='', linestyle='-', color='b',
             label='Capital Over Time')
    ax1.annotate(f'Start: ₹{capital}', xy=(all_trades['Buy Date'].iloc[0], capital),
                 xytext=(all_trades['Buy Date'].iloc[0], capital),
                 arrowprops=dict(facecolor='green', shrink=0.05))
    ax1.annotate(f'End: ₹{final_capital:.2f} ({percentage_increase:.2f}%)',
                 xy=(all_trades['Buy Date'].iloc[-1], final_capital),
                 xytext=(all_trades['Buy Date'].iloc[-1], final_capital),
                 arrowprops=dict(facecolor='red', shrink=0.05))

    # Set labels for the first y-axis
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Capital (₹)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.legend(loc='upper left')

    # Create a second y-axis to plot the Nifty 50 index
    ax2 = ax1.twinx()
    ax2.plot(nifty50_data.index, nifty50_data['Close'], linestyle='--', color='orange', label='Nifty 50')
    ax2.set_ylabel('Nifty 50 Index', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax2.legend(loc='upper center')

    # Add grid, title, and layout settings
    plt.title('Capital Growth Over Time and Nifty 50 Index')
    fig.tight_layout()
    plt.grid(True)
    plt.savefig(f'{Charts_Dir}/capital_drawdown_with_nifty50.png')
    #plt.show()
    plt.close()
def macd_ema_200_yesterday(stock_data, i):
    # Calculate EMA 200 for yesterday and the day before yesterday
    ema_200 = stock_data['Adj Close'].ewm(span=200, adjust=False).mean()
    ema_200_yesterday = ema_200.iloc[i - 1]
    ema_200_day_before_yesterday = ema_200.iloc[i - 2]

    # Calculate MACD for yesterday
    short_ema = stock_data['Adj Close'].ewm(span=12, adjust=False).mean()
    long_ema = stock_data['Adj Close'].ewm(span=26, adjust=False).mean()
    macd_line = short_ema - long_ema
    macd_line_yesterday = macd_line.iloc[i - 1]
    signal_line_yesterday = macd_line.ewm(span=9, adjust=False).mean().iloc[i - 1]
    macd_histogram_yesterday = macd_line_yesterday - signal_line_yesterday

    # Calculate buy signal for yesterday
    buy_signal_yesterday = (macd_line_yesterday > signal_line_yesterday) & (macd_line_yesterday < 0) & (ema_200_yesterday > ema_200_day_before_yesterday)
    return buy_signal_yesterday



def draw_down_chart_1():
    all_trades = []
    for filename in os.listdir(Reports_Dir):
        if filename.endswith(".csv"):
            filepath = os.path.join(Reports_Dir, filename)
            df = pd.read_csv(filepath, parse_dates=['Buy Date', 'Exited Date'])
            all_trades.append(df)
    all_trades = pd.concat(all_trades, ignore_index=True)
    all_trades = all_trades.sort_values(by='Buy Date')
    # Calculate cumulative profit
    all_trades['Cumulative Profit'] = all_trades['Profit Amount'].cumsum()
    # Calculate capital over time
    all_trades['Capital'] = capital + all_trades['Cumulative Profit']
    final_capital = all_trades['Capital'].iloc[-1]
    percentage_increase = ((final_capital - capital) / capital) * 100
    plt.figure(figsize=(14, 7))
    plt.plot(all_trades['Buy Date'], all_trades['Capital'], marker='', linestyle='-', color='b',
             label='Capital Over Time')
    # Annotate initial capital and final capital
    plt.annotate(f'Start: ₹{capital}', xy=(all_trades['Buy Date'].iloc[0], capital),
                 xytext=(all_trades['Buy Date'].iloc[0], capital),
                 arrowprops=dict(facecolor='green', shrink=0.05))
    plt.annotate(f'End: ₹{all_trades["Capital"].iloc[-1]:.2f}',
                 xy=(all_trades['Buy Date'].iloc[-1], all_trades['Capital'].iloc[-1]),
                 xytext=(all_trades['Buy Date'].iloc[-1], all_trades['Capital'].iloc[-1]),
                 arrowprops=dict(facecolor='red', shrink=0.05))
    plt.title('Capital Growth Over Time')
    plt.xlabel('Date')
    plt.ylabel('Capital (₹)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{Charts_Dir}/capital_drawdown.png')
    plt.close()
    #plt.show()
def extract_date_range_from_filename(filename):
    match = re.search(r'Master_(\d{4}-\d{2}-\d{2})_to_(\d{4}-\d{2}-\d{2})\.csv', filename)
    if match:
        return match.group(1), match.group(2)
    return None, None

def process_files(directory):
    final_data = []

    for filename in os.listdir(directory):
        if filename.startswith("Master_") and filename.endswith(".csv"):
            start_date, end_date = extract_date_range_from_filename(filename)
            if start_date and end_date:
                filepath = os.path.join(directory, filename)
                df = pd.read_csv(filepath)
                overall_row = df[df['Stock Name'] == 'Overall'].iloc[0]
                final_data.append([start_date, end_date] + overall_row.tolist())

    final_df = pd.DataFrame(final_data, columns=[
        'Start Year', 'End Year', 'Stock Name', 'Total Trades', 'No of Winning Trade', 'No of Losing Trade',
        'Winning Trade Percentage', 'Losing Trade Percentage', 'Total Profit', 'Total Cumulative Return Percentage',
        'Total Charges Paid', 'Profit After Charges', 'Profit Percentage After Charges', 'Total Charges Percentage'
    ])

    # Calculate grand totals and averages
    grand_totals = {
        'Start Year': 'Grand Total',
        'End Year': '',
        'Stock Name': 'Overall',
        'Total Trades': final_df['Total Trades'].sum(),
        'No of Winning Trade': final_df['No of Winning Trade'].sum(),
        'No of Losing Trade': final_df['No of Losing Trade'].sum(),
        'Winning Trade Percentage': round((final_df['No of Winning Trade'].sum() / final_df['Total Trades'].sum()) * 100, 2) if final_df['Total Trades'].sum() > 0 else 0,
        'Losing Trade Percentage': round((final_df['No of Losing Trade'].sum() / final_df['Total Trades'].sum()) * 100, 2) if final_df['Total Trades'].sum() > 0 else 0,
        'Total Profit': final_df['Total Profit'].sum(),
        'Total Cumulative Return Percentage': round_to_nearest_five_cents(final_df['Total Cumulative Return Percentage'].mean()),
        'Total Charges Paid': final_df['Total Charges Paid'].sum(),
        'Profit After Charges': round_to_nearest_five_cents(final_df['Profit After Charges'].sum()),
        'Profit Percentage After Charges': round_to_nearest_five_cents(final_df['Profit Percentage After Charges'].mean()),
        'Total Charges Percentage': round_to_nearest_five_cents(final_df['Total Charges Percentage'].mean())
    }

    overall_df = pd.DataFrame([grand_totals])
    final_df = pd.concat([final_df, overall_df], ignore_index=True)

    # Save the consolidated DataFrame to CSV
    final_df.to_csv(f"{directory}/Final_Consolidated.csv", index=False)
    print("Final_Consolidated.csv created successfully.")

def create_directory(symbols_type):
    directories_to_create = [f'Reports_{from_date}_to_{to_date}', f'Charts_{from_date}_to_{to_date}',
                             f'Summary_{from_date}_to_{to_date}', f'Master_{from_date}_to_{to_date}', f'Cvs_Data_{from_date}_to_{to_date}']
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
        if filename.endswith(f"_summary_{from_date}_to_{to_date}.csv"):
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
        'Winning Trade Percentage': round((sum(total_winning_trades) / sum(total_trades)) * 100, 2) if sum(total_trades) > 0 else 0,
        'Losing Trade Percentage': round((sum(total_losing_trades) / sum(total_trades)) * 100, 2) if sum(total_trades) > 0 else 0,
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
    master_df.to_csv(f"{Master_Dir}/Master_{from_date}_to_{to_date}.csv", index=False)



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

    plt.savefig(f'{Charts_Dir}/{stock}_{from_date}_to_{to_date}_plot.png')
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



def check_target_stop_loss_trades(data, capital_per_stock, target_percentage, stop_loss_percentage,):
    #total_charges_percentage = 0.2
    trade = None
    trades = []
    total_charges_paid = 0
    ###
    for i in range(100, len(data)):  # Start from the 100th day to have enough data for calculations
        if not trade and data.index[i].date() >= datetime.strptime(from_date, '%Y-%m-%d').date():
            is_previous_green = (data.iloc[i - 1]['Close'] > data.iloc[i - 1]['Open'])

            is_current_red = (data.iloc[i]['Close'] < data.iloc[i]['Open'])
            is_50EMA_above_200EMA = data.iloc[i]['EMA_50'] > data.iloc[i]['EMA_200']
            yday_50EMA_above_200EMA = data.iloc[i - 1]['EMA_50'] > data.iloc[i - 1]['EMA_200']
            yday_close_above_7EMA = data.iloc[i - 1]['Close'] > data.iloc[i - 1]['EMA_7']
            is_open_below_yday_close = data.iloc[i]['Open'] < data.iloc[i - 1]['Close']
            is_tday_high_break_yday_high = data.iloc[i]['High'] > data.iloc[i - 1]['High']
            ######
            sce_1 = is_previous_green and yday_50EMA_above_200EMA and yday_close_above_7EMA and yday_unusual_volume(data, i) and is_tday_high_break_yday_high and is_open_below_yday_close
            if sce_1:
                buy_date = data.index[i].date()
                bought_price = round_to_nearest_five_cents(data.iloc[i - 1]['High'])
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
                    'Exited Price': None,
                    'Profit Amount': None,
                    'Trade Status': None
                }
                data.loc[data.index[i], 'Buy Signal'] = data.iloc[i]['Low'].astype(float)
                data.loc[data.index[i], 'Target Level'] = target
                data.loc[data.index[i], 'Stop Loss Level'] = stop_loss
        elif trade and (trade['Stop Loss'] >= data.iloc[i]['Low'] or trade['Target'] <= data.iloc[i]['High']):
            if trade['Target'] <= data.iloc[i]['High']:
                #data.loc[data.index[i], 'Trade Status'] = 'Target'
                profit_amount = round_to_nearest_five_cents((trade['Target'] - trade['Bought Price']) * trade['Quantity Bought'])
                trade['Exited Price'] = trade['Target']
                trade['Trade Status'] = 'Target'
                trade['Exited Date'] = data.index[i].date()
            elif trade['Stop Loss'] >= data.iloc[i]['Low']:
                #data.loc[data.index[i], 'Trade Status'] = 'StopLoss'
                profit_amount = round_to_nearest_five_cents((trade['Stop Loss'] - trade['Bought Price']) * trade['Quantity Bought'])
                trade['Exited Price'] = trade['Stop Loss']
                trade['Trade Status'] = 'StopLoss'
                trade['Exited Date'] = data.index[i].date()
            sell_turnover = bought_price * trade['Quantity Bought']
            # Calculate charges for selling
            sell_charges = (total_charges_percentage / 100) * sell_turnover
            total_charges_paid += round_to_nearest_five_cents(sell_charges)
            trade['Profit Amount'] = round_to_nearest_five_cents(profit_amount - buy_charges - sell_charges)
            trades.append(trade)
            #print(trade)
            trade = None
            if compound:
                capital_per_stock += int(profit_amount)
    if create_chart and len(trades) != 0:
        visualize(data, 'Target Level', 'Stop Loss Level', stock, Charts_Dir)
    num_buy_signals = data["Buy Signal"].notna().sum()
    #print(len(trades))
    #print(num_buy_signals)
    if len(trades) != num_buy_signals:
        data.to_csv(f"{cvs_data_dir}/{stock}_data.csv")
        data, trades, total_charges_paid_cl = get_last_unclosed_trade(data, trades, capital_per_stock, total_charges_paid)
        total_charges_paid = total_charges_paid_cl + total_charges_paid
    return trades, total_charges_paid
def get_last_unclosed_trade(data, trades, capital_per_stock, total_charges_paid):
        ########
        non_empty_indices = data[data["Buy Signal"].notna()].index
        last_buy_signal_index = non_empty_indices[len(non_empty_indices) - 1]
        buy_date = last_buy_signal_index.strftime("%Y-%m-%d")
        #print(buy_date)
        #print(non_empty_indices)
        #
        prev_buy_signal_index = data[data["Buy Signal"].notna()].index[-1]
        #print(prev_buy_signal_index)
        #previous_day_data = data.loc[prev_buy_signal_index - data.Timedelta(days=1)]
        previous_trading_day_index = data.index[data.index.get_loc(prev_buy_signal_index) - 1].strftime("%Y-%m-%d")
        #print(previous_trading_day_index)
        previous_day_data = data.loc[previous_trading_day_index]
        bought_price = previous_day_data['High']
        #print(bought_price)
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
            'Exited Price': None,
            'Profit Amount': None,
            'Trade Status': None
        }
        profit_amount = round_to_nearest_five_cents((data.iloc[-1]['Close'] - trade['Bought Price']) * trade['Quantity Bought'])
        sell_turnover = bought_price * trade['Quantity Bought']
        trade['Exited Date'] = data.index[-1].strftime("%Y-%m-%d")
        trade['Exited Price'] = data.iloc[-1]['Close']
        trade['Trade Status'] = 'Cl_Open'
        # Calculate charges for selling
        sell_charges = (total_charges_percentage / 100) * sell_turnover
        total_charges_paid += round_to_nearest_five_cents(sell_charges)
        trade['Profit Amount'] = round_to_nearest_five_cents(profit_amount - buy_charges - sell_charges)
        trades.append(trade)
        return data, trades, total_charges_paid
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
        trades, total_charges_paid = check_target_stop_loss_trades(data, capital_per_stock, target_percentage, stop_loss_percentage)
        #trades, total_charges_paid = check_open_tardes(data, capital, capital_per_stock, target_percentage, stop_loss_percentage, )
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
        df.to_csv(f"{Reports_Dir}/{symbol}_trades_{from_date}_to_{to_date}.csv", index=False)
        print("Creating Draw-Down Chart.....")
        draw_down_chart()
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
        summary_df.to_csv(f"{Summary_Dir}/{symbol}_summary_{from_date}_to_{to_date}.csv", index=False)
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
    #from_date = datetime.strptime(config['time_management']['from_date'], '%Y-%m-%d').date()
    #to_date = datetime.strptime(config['time_management']['to_date'], '%Y-%m-%d').date()
    year_wise = str(config['time_management']['year_wise'])
    year_wise = True if year_wise == 'true' else False
    year_split = int(config['time_management']['year_split'])
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
    cvs_data_dir = f'{symbols_type}_Cvs_Data_{from_date}_to_{to_date}'

    ##############
    create_directory(symbols_type)

    with open('./symbols/' + symbols_file, 'r') as file:
        stocks = [line.split('#')[0].strip() for line in file if not line.lstrip().startswith('#')]
    total_stocks = len(stocks)
    print(f"Total number of stocks: {total_stocks}")
    # Initialize lists to store summarized information
    summary_data = []
    end_year_c = 0
    # Initialize total charges variable
    total_charges_for_all_stocks = 0
    if year_wise:
        from_date = datetime.strptime(from_date, '%Y-%m-%d').date()
        to_date = datetime.strptime(to_date, '%Y-%m-%d').date()
        start_year = from_date.year
        end_year = to_date.year
        mult = round((end_year - start_year) / year_split)
        #print((end_year - start_year))
        #print(mult)
        while end_year_c < mult:
            from_date = str(int(start_year))+'-'+'01-01'
            to_date = str(int(start_year + (year_split - 1))) + '-' + '12-31'
            print(f"{from_date}----{to_date}")
            #year_start_date = max(from_date, datetime.date(year, 1, 1))
            #year_end_date = min(to_date, datetime.date(year, 12, 31))
            for count, stock in enumerate(stocks, 1):
                print(f"Processing stock {count}/{total_stocks}: {stock}")
                # Call main function with parameters and aggregate total charges
                charges_paid, trades = main(stock + ".NS", from_date, to_date, capital, target_percentage, stop_loss_percentage)
                total_charges_for_all_stocks += charges_paid
            # Call function to create the Master_no_Compound_sce_5.csv file
            create_master_file(Summary_Dir)
            start_year= start_year + year_split
            to_date = start_year + year_split
            end_year_c = end_year_c + 1

        print("Processing Consolidated Master File....")
        process_files(Master_Dir)
    else:
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
