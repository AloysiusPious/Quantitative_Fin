import matplotlib.pyplot as plt
import yfinance as yf
import glob
import shutil
import os
import numpy as np
import requests
from io import StringIO
import pandas as pd
from datetime import datetime, timedelta, time
import re
import math
def macd_buy(df, i):
    # Ensure that the index 'i' is valid (i.e., it must have enough previous data points)
    if i < 1 or i >= len(df):
        raise ValueError("Index 'i' must be between 1 and the length of the DataFrame - 1.")

    # Calculate MACD
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Calculate Bollinger Bands
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Std_Dev'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['SMA_20'] + (2 * df['Std_Dev'])
    df['Lower_Band'] = df['SMA_20'] - (2 * df['Std_Dev'])

    # Check buy signal for yesterday's candle (index 'i')
    is_buy_signal = df['MACD'].iloc[i] > df['Signal_Line'].iloc[i]
    return is_buy_signal

def buy_signal_macd_rsi_bband(df, i):
    # Ensure that the index 'i' is valid (i.e., it must have enough previous data points)
    if i < 1 or i >= len(df):
        raise ValueError("Index 'i' must be between 1 and the length of the DataFrame - 1.")

    # Calculate MACD
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Calculate Bollinger Bands
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Std_Dev'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['SMA_20'] + (2 * df['Std_Dev'])
    df['Lower_Band'] = df['SMA_20'] - (2 * df['Std_Dev'])

    # Check buy signal for yesterday's candle (index 'i')
    is_buy_signal = (
        (df['MACD'].iloc[i-1] > df['Signal_Line'].iloc[i-1]) &  # MACD crosses above Signal Line
        (df['RSI'].iloc[i-1] > 40) & (df['RSI'].iloc[i-1] < 50) and (df['RSI'].iloc[i-10] < 30))# &                           # RSI below 30 (Oversold)
        #(df['Close'].iloc[i-1] < df['Lower_Band'].iloc[i-1])   # Price touches or is below the Lower Bollinger Band
    return is_buy_signal

def calculate_supertrend(df, period=7, multiplier=3):
    """
    Calculate SuperTrend indicator.

    Parameters:
    df (DataFrame): DataFrame with columns ['Date', 'Open', 'High', 'Low', 'Close']
    period (int): The lookback period for the ATR calculation
    multiplier (int): The multiplier for the ATR to calculate the SuperTrend

    Returns:
    DataFrame: DataFrame with SuperTrend values and buy/sell signals
    """
    # Calculate ATR
    df['HL'] = df['High'] - df['Low']
    df['HC'] = abs(df['High'] - df['Close'].shift())
    df['LC'] = abs(df['Low'] - df['Close'].shift())
    df['TR'] = df[['HL', 'HC', 'LC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=period).mean()

    # Calculate SuperTrend
    df['Upper Basic'] = (df['High'] + df['Low']) / 2 + (multiplier * df['ATR'])
    df['Lower Basic'] = (df['High'] + df['Low']) / 2 - (multiplier * df['ATR'])
    df['Upper Band'] = df['Upper Basic']
    df['Lower Band'] = df['Lower Basic']

    for i in range(1, len(df)):
        if df.loc[i - 1, 'Close'] <= df.loc[i - 1, 'Upper Band']:
            df.loc[i, 'Upper Band'] = min(df.loc[i, 'Upper Basic'], df.loc[i - 1, 'Upper Band'])
        else:
            df.loc[i, 'Upper Band'] = df.loc[i, 'Upper Basic']

        if df.loc[i - 1, 'Close'] >= df.loc[i - 1, 'Lower Band']:
            df.loc[i, 'Lower Band'] = max(df.loc[i, 'Lower Basic'], df.loc[i - 1, 'Lower Band'])
        else:
            df.loc[i, 'Lower Band'] = df.loc[i, 'Lower Basic']

    df['SuperTrend'] = df['Upper Band']
    for i in range(1, len(df)):
        if df.loc[i, 'Close'] > df.loc[i - 1, 'Upper Band']:
            df.loc[i, 'SuperTrend'] = df.loc[i, 'Lower Band']
        elif df.loc[i, 'Close'] < df.loc[i - 1, 'Lower Band']:
            df.loc[i, 'SuperTrend'] = df.loc[i, 'Upper Band']
        else:
            df.loc[i, 'SuperTrend'] = df.loc[i - 1, 'SuperTrend']

    # Determine buy/sell signals
    df['Signal'] = 0
    for i in range(1, len(df)):
        if df.loc[i, 'Close'] > df.loc[i, 'SuperTrend'] and df.loc[i - 1, 'Close'] <= df.loc[i - 1, 'SuperTrend']:
            df.loc[i, 'Signal'] = 1  # Buy signal
        elif df.loc[i, 'Close'] < df.loc[i, 'SuperTrend'] and df.loc[i - 1, 'Close'] >= df.loc[i - 1, 'SuperTrend']:
            df.loc[i, 'Signal'] = -1  # Sell signal
    return df
def candle_50_high_low_swing(df, i):
    # Ensure there are enough data points
    if i < 50:
        return False

    # Condition 1: Difference between past 50 days low and high should be 15% more
    high_50 = df['High'].iloc[i-50:i].max()
    low_50 = df['Low'].iloc[i-50:i].min()
    if (high_50 - low_50) / low_50 > 0.15:
        return False

    # Condition 2: 1 day before candle low greater than 3% and less than 8%
    prev_day_low = df['Low'].iloc[i-1]
    prev_day_open = df['Open'].iloc[i-1]
    low_percent_change = (prev_day_open - prev_day_low) / prev_day_open
    if not (0.03 < low_percent_change < 0.08):
        return False

    # Condition 3: 1 day before candle should be green and bullish candle
    prev_day_close = df['Close'].iloc[i-1]
    if prev_day_close > prev_day_open:
        return False

    # If all conditions are true, return True
    return True
def copy_specific_files(file_paths, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for file_path in file_paths:
        if os.path.isfile(file_path):
            shutil.copy(file_path, dest_dir)
            print(f"Copied {file_path} to {dest_dir}")
        else:
            print(f"File not found: {file_path}")


def get_stock_for_date_refrence(cvs_data_dir, from_date, to_date):
    print('Downloading stock for Date reference..')

    nifty50_data = get_nifty50_data(from_date, to_date)

    # Convert index to column and rename it to 'Date'
    date_ref = pd.DataFrame(nifty50_data.index)
    date_ref.columns = ['Date']

    # Drop any empty rows
    date_ref.dropna(inplace=True)

    # Save to CSV
    date_ref.to_csv(f"{cvs_data_dir}/stock_date_ref.csv", index=False, date_format='%Y-%m-%d')

    print('Ok.')




def get_nifty50_data(from_date, to_date):
    # Fetch Nifty 50 data within the specified date range
    #nifty50_data = yf.Ticker("^NSEI")
    nifty50_data = yf.download('TCS.NS', start=from_date, end=to_date)
    #print(nifty50_data)
    #nifty50_data = nifty50.history(start=from_date, end=to_date)
    return nifty50_data
def fetch_yahoo_finance_data_old(symbol, start_date, end_date):
    try:
        from_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
        try:
            adjusted_from_date_obj = from_date_obj.replace(year=from_date_obj.year - 1)
        except ValueError:
            adjusted_from_date_obj = from_date_obj.replace(month=2, day=28, year=from_date_obj.year - 1)
        start_date = adjusted_from_date_obj.strftime('%Y-%m-%d')

        data = yf.download(symbol, start=start_date, end=end_date)
        if data.empty:
            print(f"No data found for {symbol}")
            return None

        col = ['Open', 'High', 'Low', 'Close']
        return data[col]
        print(data[col])
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def fetch_yahoo_finance_data_02_NOV_2025(symbol, start_date, end_date):
    try:
        # Adjust start date back by one year
        from_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
        try:
            adjusted_from_date_obj = from_date_obj.replace(year=from_date_obj.year - 1)
        except ValueError:
            adjusted_from_date_obj = from_date_obj.replace(month=2, day=28, year=from_date_obj.year - 1)
        start_date = adjusted_from_date_obj.strftime('%Y-%m-%d')

        # Download data
        data = yf.download(symbol, start=start_date, end=end_date)

        if data.empty:
            print(f"No data found for {symbol}")
            return None

        # Reset index to expose 'Date'
        data.reset_index(inplace=True)

        # Handle multi-level columns (happens sometimes in Yahoo Finance)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if col[0] else col[1] for col in data.columns]

        # If 'Adj Close' exists, drop it
        if 'Adj Close' in data.columns:
            data.drop(columns=['Adj Close'], inplace=True)

        # Select only required columns if others exist
        expected_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        data = data[[col for col in data.columns if col in expected_cols]]

        # Rename columns to ensure consistency
        data.columns = expected_cols

        # Ensure numeric types
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        data.dropna(inplace=True)

        return data

    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None
from kiteconnect import KiteConnect


def fetch_kite_data(enctoken, symbol, start_date, end_date, interval='day'):
    """
    Fetch historical OHLCV data from Zerodha using enctoken (no API key needed).
    - Automatically prefixes 'NSE:' if missing
    - Caches instrument list locally
    - Splits requests if date range exceeds 2000 days
    - Handles extra columns like OI
    - Returns only Date, Open, High, Low, Close, Volume
    - Date is returned as YYYY-MM-DD string
    """
    try:
        # Auto-add NSE prefix if missing
        if ":" not in symbol:
            symbol = f"NSE:{symbol}"

        cache_file = "instruments.csv"

        # Adjust start date back by 1 year
        from_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
        try:
            adjusted_from_date_obj = from_date_obj.replace(year=from_date_obj.year - 1)
        except ValueError:
            adjusted_from_date_obj = from_date_obj.replace(month=2, day=28, year=from_date_obj.year - 1)
        start_date_dt = adjusted_from_date_obj
        end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')

        # Headers
        headers = {
            "Authorization": f"enctoken {enctoken}",
            "Content-Type": "application/json",
            "X-Kite-Version": "3"
        }

        root_url = "https://kite.zerodha.com/oms"

        # Load or download instrument list
        if os.path.exists(cache_file):
            instruments_df = pd.read_csv(cache_file)
        else:
            print("📥 Downloading instrument list...")
            resp = requests.get("https://api.kite.trade/instruments")
            if resp.status_code != 200:
                print("❌ Failed to download instrument list:", resp.text)
                return None
            instruments_df = pd.read_csv(StringIO(resp.text))
            instruments_df.to_csv(cache_file, index=False)

        # Match symbol
        tradingsymbol = symbol.split(":")[-1]
        exchange = symbol.split(":")[0]

        instrument_row = instruments_df[
            (instruments_df["tradingsymbol"] == tradingsymbol)
            & (instruments_df["exchange"] == exchange)
        ]
        if instrument_row.empty:
            print(f"⚠️ Symbol not found in instrument list: {symbol}")
            return None

        token = int(instrument_row["instrument_token"].values[0])

        # Split into max 2000-day chunks
        max_days = 2000
        df_list = []
        chunk_start = start_date_dt
        while chunk_start <= end_date_dt:
            chunk_end = min(chunk_start + timedelta(days=max_days-1), end_date_dt)
            hist_url = f"{root_url}/instruments/historical/{token}/{interval}"
            params = {"oi": "1", "from": chunk_start.strftime('%Y-%m-%d'), "to": chunk_end.strftime('%Y-%m-%d')}
            response = requests.get(hist_url, headers=headers, params=params)

            if response.status_code != 200:
                print(f"❌ HTTP {response.status_code} from Zerodha: {response.text}")
                return None

            try:
                js = response.json()
            except Exception:
                print(f"❌ Non-JSON response: {response.text[:500]}")
                return None

            if not js or "data" not in js or not js["data"] or "candles" not in js["data"]:
                print(f"⚠️ No candle data found for {symbol}: {js}")
                return None

            # Determine columns dynamically
            sample_len = len(js["data"]["candles"][0])
            if sample_len >= 6:
                cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
                if sample_len > 6:
                    extra_cols = [f"extra{i}" for i in range(sample_len-6)]
                    cols += extra_cols
            else:
                print(f"⚠️ Unexpected candle format for {symbol}: {js['data']['candles'][0]}")
                return None

            chunk_df = pd.DataFrame(js["data"]["candles"], columns=cols)
            chunk_df["Date"] = pd.to_datetime(chunk_df["Date"])
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                chunk_df[col] = pd.to_numeric(chunk_df[col], errors="coerce")
            chunk_df.dropna(subset=["Open", "High", "Low", "Close", "Volume"], inplace=True)

            df_list.append(chunk_df)

            # Next chunk
            chunk_start = chunk_end + timedelta(days=1)

        # Concatenate all chunks
        df_final = pd.concat(df_list).reset_index(drop=True)

        # Keep only OHLCV and convert date to string YYYY-MM-DD
        df_final = df_final[["Date", "Open", "High", "Low", "Close", "Volume"]]
        df_final["Date"] = df_final["Date"].dt.strftime('%Y-%m-%d')

        return df_final

    except Exception as e:
        print(f"❌ {symbol} failed: {e}")
        return None
def fetch_yahoo_finance_data(symbol, start_date, end_date):
    try:
        # Adjust start date back by one year
        from_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
        try:
            adjusted_from_date_obj = from_date_obj.replace(year=from_date_obj.year - 1)
        except ValueError:
            adjusted_from_date_obj = from_date_obj.replace(month=2, day=28, year=from_date_obj.year - 1)
        start_date = adjusted_from_date_obj.strftime('%Y-%m-%d')

        # Download data
        data = yf.download(symbol, start=start_date, end=end_date, progress=False, threads=True)

        if data.empty:
            print(f"⚠️ No data found for {symbol}")
            return None

        # Reset index
        data.reset_index(inplace=True)

        # Handle multi-index columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if col[0] else col[1] for col in data.columns]

        # Drop extra columns like 'Adj Close'
        if 'Adj Close' in data.columns:
            data.drop(columns=['Adj Close'], inplace=True)

        # Keep only required columns
        expected_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        data = data[[col for col in data.columns if col in expected_cols]]

        # Rename columns (ensure order)
        data.columns = expected_cols

        # Ensure numeric
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        data.dropna(inplace=True)
        return data

    except Exception as e:
        print(f"❌ Error fetching {symbol}: {e}")
        return None







def fetch_weekly_data(ticker, start_date, end_date):
    # Convert from_date to a datetime object
    from_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
    # Subtract one year
    adjusted_from_date_obj = from_date_obj.replace(year=from_date_obj.year - 1)
    # Convert the adjusted date back to a string
    start_date = adjusted_from_date_obj.strftime('%Y-%m-%d')
    # Download data with weekly interval
    data = yf.download(ticker, start=start_date, end=end_date, interval="1wk")
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
    import math, pandas as pd
    if value is None or pd.isna(value) or math.isnan(value):
        return 0.0
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


def is_ema_rising(data, i, ema_period):
    """
    Check if the specified EMA is rising on day 'i'.

    Parameters:
    data (pd.DataFrame): DataFrame containing stock data with 'Close' prices.
    i (int): The specific row index to check.
    ema_period (int): The period for the EMA calculation.

    Returns:
    bool: True if the EMA is rising on day 'i', False otherwise.
    """
    # Calculate the EMA for the specified period
    data[f'EMA_{ema_period}'] = data['Close'].ewm(span=ema_period, adjust=False).mean()

    # Check if the EMA is rising on the specified day
    if i >= 1 and data.iloc[i][f'EMA_{ema_period}'] > data.iloc[i - 1][f'EMA_{ema_period}']:
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


def calculate_cci(df, length=50):
    # Typical Price
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    # Simple Moving Average of Typical Price
    sma = tp.rolling(window=length).mean()
    # Mean Deviation
    mean_dev = tp.rolling(window=length).apply(lambda x: abs(x - x.mean()).mean())
    # CCI Calculation
    cci = (tp - sma) / (0.015 * mean_dev)
    return cci


def check_cci_condition_close(df, i):
    # Calculate CCI with length of 50
    df['CCI'] = calculate_cci(df, length=50)

    # Check if CCI at day i is below -150 and the Close price of that day meets the condition
    if df['CCI'].iloc[i] < -150:# and df['Close'].iloc[i] < -150:
    #if df['CCI'].iloc[i] > 100 and df['CCI'].iloc[i - 1] < 100 and df['CCI'].iloc[i - 2] < 100 and df['CCI'].iloc[i - 3] < 100:
        return True
    return False
def get_heikin_ashi_candle(prev_ha_open, prev_ha_close, open_, high, low, close):
    # Calculate current Heikin-Ashi close
    ha_close = (open_ + high + low + close) / 4

    # Calculate current Heikin-Ashi open
    ha_open = (prev_ha_open + prev_ha_close) / 2

    # Calculate current Heikin-Ashi high and low
    ha_high = max(high, ha_open, ha_close)
    ha_low = min(low, ha_open, ha_close)

    return ha_open, ha_high, ha_low, ha_close
def calculate_heikin_ashi(df):
    # Make a deep copy of the DataFrame to avoid chaining issues
    heikin_ashi_df = df.copy()
    # Calculate Heikin-Ashi close
    heikin_ashi_df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    # Calculate Heikin-Ashi open
    heikin_ashi_df['HA_Open'] = (df['Open'].shift(1) + df['Close'].shift(1)) / 2
    heikin_ashi_df.loc[0, 'HA_Open'] = df['Open'].iloc[0]  # Initialize the first HA_Open using .loc to avoid chained assignment
    # Calculate Heikin-Ashi high and low with .loc[] to avoid chained assignment
    heikin_ashi_df['HA_High'] = heikin_ashi_df[['HA_Open', 'HA_Close', 'High']].max(axis=1)
    heikin_ashi_df['HA_Low'] = heikin_ashi_df[['HA_Open', 'HA_Close', 'Low']].min(axis=1)
    return heikin_ashi_df


def check_heikin_ashi_conditions(df, i):
    # Convert to Heikin Ashi candles
    ha_df = calculate_heikin_ashi(df)
    # Calculate 20-day and 200-day EMAs
    ha_df['20_EMA'] = ha_df['Close'].ewm(span=20, adjust=False).mean()
    ha_df['200_EMA'] = ha_df['Close'].ewm(span=200, adjust=False).mean()
    # Condition 1: 20-day lowest low < 20 EMA, 20-day highest high > 20 EMA, current close < 20 EMA
    low_20_days = ha_df['Low'].iloc[i - 19:i + 1].min()
    high_20_days = ha_df['High'].iloc[i - 19:i + 1].max()
    current_close = ha_df['HA_Close'].iloc[i]
    ema_20 = ha_df['20_EMA'].iloc[i]
    condition1 = (
            low_20_days < ema_20 and
            high_20_days > ema_20 and
            current_close < ema_20
    )
    # Condition 2: 20 EMA < 200 EMA
    ema_200 = ha_df['200_EMA'].iloc[i]
    condition2 = ema_20 < ema_200
    # Condition 3: Current close < 200 EMA
    condition3 = current_close < ema_200
    # Return True only if all conditions are met
    return condition1 and condition2 and condition3


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
def get_ref_stock_date(cvs_data_dir, from_date, to_date):
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
            data[col] = data[col].astype(float)
            #data.loc[:, col] = data[col].apply(round_to_nearest_five_cents)
            data.loc[:, col] = data[col].apply(round_to_nearest_five_cents)
    #print(data)
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
def is_52_weeks_high(filtered_df, i):
    # Define the lookback period for 52 weeks (assuming 5 trading days per week)
    lookback_period = 252

    # Get the high prices for the lookback period
    high_prices = filtered_df['High'].iloc[max(0, i - lookback_period):i + 1]

    # Check if the current closing price is the highest in the lookback period
    is_highest = filtered_df['Close'].iloc[i] == high_prices.max()

    return is_highest
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
def calculate_ema_vol(data, ema_period):
    data['EMA_VOL_' + str(ema_period)] = data['Volume'].rolling(window=ema_period).mean()
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