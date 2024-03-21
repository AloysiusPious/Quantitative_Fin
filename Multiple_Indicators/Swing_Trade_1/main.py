import pandas as pd
import yfinance as yf
import talib as ta
import os

# Define function to fetch Yahoo Finance data
def fetch_yahoo_finance_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

# Define function to calculate EMA
def calculate_ema(data, ema_period=200):
    data['EMA_'+str(ema_period)] = ta.EMA(data['Close'], timeperiod=ema_period)
    return data

# Define function to check buying conditions and track trades
def check_buy_conditions(data, capital_per_stock, target_percentage, stop_loss_percentage):
    trade = None
    trades = []
    invested_amount = 0

 
    for i in range(4, len(data)):  # Start from the 4th day to have enough data for calculations
        if not trade:
            is_previous_three_red = (data.iloc[i - 1]['Close'] < data.iloc[i - 1]['Open']) and (
                    data.iloc[i - 2]['Close'] < data.iloc[i - 2]['Open']) and (
                                            data.iloc[i - 3]['Close'] < data.iloc[i - 3]['Open'])
            is_current_green = (data.iloc[i]['Close'] > data.iloc[i]['Open'])
            is_current_close_above_previous_open = (data.iloc[i]['Close'] > data.iloc[i - 1]['Open'])
            if is_previous_three_red and is_current_green and is_current_close_above_previous_open and \
                    data.iloc[i]['Close'] > data.iloc[i]['EMA_200'] and data.iloc[i]['Close'] > data.iloc[i]['EMA_50']:
                # Buy condition met
                buy_date = data.index[i].date()
                quantity_bought = capital_per_stock / data.iloc[i]['Close']
                invested_amount += capital_per_stock
                stop_loss = data['Close'].iloc[i] * (1 - stop_loss_percentage / 100)
                target = data['Close'].iloc[i] * (1 + target_percentage / 100)
                trade = {'Buy Date': buy_date, 'Quantity Bought': quantity_bought, 'Invested Amount': capital_per_stock,
                         'Stop Loss': stop_loss, 'Target': target, 'Exited Date': None, 'Profit Amount': None}
        elif trade and ((trade['Stop Loss'] >= data.iloc[i]['Low']) or (trade['Target'] <= data.iloc[i]['High'])):
            # Sell condition met
            sell_date = data.index[i].date()
            profit_amount = (data.iloc[i]['Close'] - trade['Invested Amount'] / trade['Quantity Bought']) * \
                            trade['Quantity Bought']
            trade['Exited Date'] = sell_date  # Update Exited Date to target or stop loss hit date
            trade['Profit Amount'] = profit_amount
            invested_amount -= trade['Invested Amount']
            trades.append(trade)
            trade = None  # Reset trade

    return trades
def create_directory(symbols_type):
    directories_to_create = [f'Reports_{from_date}_to_{to_date}', f'Charts_{from_date}_to_{to_date}', f'Summary_{from_date}_to_{to_date}', f'Master_{from_date}_to_{to_date}']
    # Iterate over each directory and create it if it does not exist
    for directory in directories_to_create:
        directory_name = symbols_type + "_" + directory
        #print(directory_name)
        """Create directory if it does not exist"""
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
            #print(f"Directory '{directory_name}' created successfully.")
        #else:
         #   print(f"Directory '{directory_name}' already exists.")


def main(symbol, start_date, end_date, capital, target_percentage, stop_loss_percentage):
    try:
        # Fetch data from Yahoo Finance
        data = fetch_yahoo_finance_data(symbol, start_date, end_date)

        if data.empty:
            print(f"No data found for {symbol}. Skipping...")
            return  # Skip this stock

        # Calculate EMA
        data = calculate_ema(data, 200)
        data = calculate_ema(data, 50)

        # Calculate capital per stock
        capital_per_stock = capital / 10  # Assuming you're allowed to trade 10 stocks at a time

        # Check buying conditions and track trades
        trades = check_buy_conditions(data, capital_per_stock, target_percentage, stop_loss_percentage)

        if not trades:
            print(f"No trades found for {symbol}. Skipping...")
            return  # Skip this stock

        # Calculate No of holding Days
        df = pd.DataFrame(trades)
        df['Buy Date'] = pd.to_datetime(df['Buy Date'])  # Convert Buy Date to datetime format
        df['Exited Date'] = pd.to_datetime(df['Exited Date'])  # Convert Exited Date to datetime format
        df['No of holding Days'] = (df['Exited Date'] - df['Buy Date']).dt.days  # Calculate holding days

        # Calculate Profit %
        df['Profit %'] = (df['Profit Amount'] / df['Invested Amount']) * 100

        # Save DataFrame to CSV
        df.to_csv(f"{Reports_Dir}/{stock}_trades.csv", index=False)

        # Create summary DataFrame
        summary_df = pd.DataFrame({
            'Total Trades': [len(trades)],
            'No of Winning Trade': [len([trade for trade in trades if trade['Profit Amount'] and trade['Profit Amount'] > 0])],
            'No of Losing Trade': [len([trade for trade in trades if trade['Profit Amount'] and trade['Profit Amount'] < 0])],
            'Winning Trade Percentage': [len([trade for trade in trades if trade['Profit Amount'] and trade['Profit Amount'] > 0]) / len(trades) * 100],
            'Losing Trade Percentage': [len([trade for trade in trades if trade['Profit Amount'] and trade['Profit Amount'] < 0]) / len(trades) * 100],
            'Total Profit': [sum([trade['Profit Amount'] for trade in trades if trade['Profit Amount']])],
            'Cumulative Return Percentage': [(sum([trade['Profit Amount'] for trade in trades if trade['Profit Amount']]) / capital) * 100]
        })

        # Save summary DataFrame to CSV
        summary_df.to_csv(f"{Summary_Dir}/{stock}_summary.csv", index=False)

    except Exception as e:
        print(f"Error processing {symbol}: {e}")

if __name__ == "__main__":
    # Define parameters
    symbols_file = 'custom.txt'
    #symbol = 'INFY.NS'  # Example symbol
    from_date = '2017-01-01'  # 5 years ago from now
    to_date = '2022-01-01'  # Current date
    capital = 100000
    target_percentage = 10  # Example target percentage
    stop_loss_percentage = 5  # Example stop loss percentage
    ##############
    symbols_type = symbols_file.split('.')[0]
    Reports_Dir = f'{symbols_type}_Reports_{from_date}_to_{to_date}'
    Charts_Dir = f'{symbols_type}_Charts_{from_date}_to_{to_date}'
    Summary_Dir = f'{symbols_type}_Summary_{from_date}_to_{to_date}'
    Master_Dir = f'{symbols_type}_Master_{from_date}_to_{to_date}'
    ##############
    create_directory(symbols_type)

    with open('../symbols/' + symbols_file, 'r') as file:
        stocks = [line.split('#')[0].strip() for line in file if not line.lstrip().startswith('#')]

    # Initialize lists to store summarized information
    summary_data = []

    for stock in stocks:
        # Call main function with parameters
        main(stock+str(".NS"), from_date, to_date, capital, target_percentage, stop_loss_percentage)
