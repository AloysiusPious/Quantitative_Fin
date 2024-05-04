import pandas as pd
import yfinance as yf
#import talib as ta
import os
import matplotlib.pyplot as plt

# Define function to fetch Yahoo Finance data
def fetch_yahoo_finance_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data
# Define function to fetch Yahoo Finance data
# Define function to calculate EMA using pandas
def calculate_ema(data, ema_period=200):
    data['EMA_' + str(ema_period)] = data['Close'].rolling(window=ema_period).mean()
    return data
'''
# Define function to calculate EMA
def calculate_ema(data, ema_period=200):
    data['EMA_'+str(ema_period)] = ta.EMA(data['Close'], timeperiod=ema_period)
    return data
'''
# Define function to check buying conditions and track trades
def pd_rsi_below_n(filtered_df, i, window = 14, n = 30, ohlc = "Close"):
    # Calculate RSI with a period of 14 days
    delta = filtered_df[ohlc].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    # Check if RSI is less than n
    rsi_below = rsi.iloc[i] < n
    return rsi_below
def pd_rsi_above_n(filtered_df, i, window = 14, n = 30, ohlc = "Close"):
    # Calculate RSI with a period of 14 days
    delta = filtered_df[ohlc].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    # Check if RSI is less than n
    rsi_below = rsi.iloc[i] > n
    return rsi_below
def ta_rsi_above_n(filtered_df, i, n, ohlc = "Close"):
    ################## RSI - Begin ################
    # Calculate RSI with a period of 14 days
    filtered_df['RSI'] = ta.RSI(filtered_df[ohlc])
    # Check if RSI is less than 32
    filtered_df[f'RSI_Less_{n}'] = filtered_df['RSI'].iloc[i] < n
    return filtered_df[f'RSI_Less_{n}'].iloc[i]


def pd_rsi_cross_n(filtered_df, i, window = 14, n = 30, ohlc = "Close"):
    # Calculate RSI with a period of 14 days
    delta = filtered_df[ohlc].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # Check if RSI crosses n
    rsi_cross = (rsi.iloc[i - 2] < n) and (rsi.iloc[i] > n)
    return rsi_cross


def check_buy_conditions(data, capital_per_stock, target_percentage, stop_loss_percentage):
    trade = None
    trades = []
    invested_amount = 0

    # Create a new column in the DataFrame to track buy signals
    data['Buy Signal'] = 0

    for i in range(350, len(data)):  # Start from the 4th day to have enough data for calculations
        if not trade:
            is_previous_red = (data.iloc[i - 1]['Close'] < data.iloc[i - 1]['Open'])
            is_previous_three_red = (data.iloc[i - 1]['Close'] < data.iloc[i - 1]['Open']) and (
                    data.iloc[i - 2]['Close'] < data.iloc[i - 2]['Open']) and (
                                            data.iloc[i - 3]['Close'] < data.iloc[i - 3]['Open'])
            is_current_green = (data.iloc[i]['Close'] > data.iloc[i]['Open'])
            is_current_close_above_previous_open = (data.iloc[i]['Close'] > data.iloc[i - 1]['Open'])
            is_close_above_200EMA = data.iloc[i]['Close'] > data.iloc[i]['EMA_200']
            is_close_above_300EMA = data.iloc[i]['Close'] > data.iloc[i]['EMA_300']
            is_close_below_200EMA = data.iloc[i]['Close'] < data.iloc[i]['EMA_200']
            is_close_above_50EMA = data.iloc[i]['Close'] > data.iloc[i]['EMA_50']
            is_close_below_50EMA = data.iloc[i]['Close'] < data.iloc[i]['EMA_50']
            is_close_above_20EMA = data.iloc[i]['Close'] > data.iloc[i]['EMA_20']
            is_close_below_20EMA = data.iloc[i]['Close'] < data.iloc[i]['EMA_20']
            is_50EMA_above_200EMA = data.iloc[i]['EMA_50'] > data.iloc[i]['EMA_200']
            is_current_open_less_than_previous_close = data.iloc[i]['Open'] < data.iloc[i - 1]['Close']

            ################### RSI ############
            rsi_above_clo = pd_rsi_above_n(data, i, 55, 50, "Close")
            rsi_cross_clo = pd_rsi_cross_n(data, i, 14, 30, "Close")
            rsi_below_clo = pd_rsi_below_n(data, i, 14, 30, "Close")
            rsi_point = 27
            rsi_bullish_cand_1 = pd_rsi_below_n(data, i, 13, rsi_point, "Low") and pd_rsi_above_n(data, i, 13, rsi_point, "Open") and pd_rsi_above_n(data, i, 13, rsi_point, "Close")
            rsi_bullish_cand_2 = pd_rsi_below_n(data, i - 1, 13, rsi_point, "Low") and pd_rsi_above_n(data, i - 1, 13, rsi_point, "Open") and pd_rsi_above_n(data, i - 1, 13, rsi_point, "Close")
            rsi_bullish_cand_3 = pd_rsi_below_n(data, i - 2, 13, rsi_point, "Low") and pd_rsi_above_n(data, i - 2, 13, rsi_point, "Open") and pd_rsi_above_n(data, i - 2, 13, rsi_point, "Close")
            rsi_bullish_cand_4 = pd_rsi_below_n(data, i - 3, 13, rsi_point, "Low") and pd_rsi_above_n(data, i - 3, 13, rsi_point, "Open") and pd_rsi_above_n(data, i - 3, 13, rsi_point, "Close")
            rsi_bullish_cand_5 = pd_rsi_below_n(data, i - 4, 13, rsi_point, "Low") and pd_rsi_above_n(data, i - 4, 13, rsi_point, "Open") and pd_rsi_above_n(data, i - 4, 13, rsi_point, "Close")
            rsi_bullish_pattern = rsi_bullish_cand_1 and rsi_bullish_cand_2 and rsi_bullish_cand_3 and rsi_bullish_cand_4 and rsi_bullish_cand_5
            ################
            high_break = data.iloc[i]['High'] > data.iloc[i - 1]['High'] and data.iloc[i]['High'] > \
                         data.iloc[i - 2]['High']
            ################
            condition_1 = pd_rsi_above_n(data, i, 14, 20, "Close") and pd_rsi_below_n(data, i - 1, 14, 20, "Close") and pd_rsi_above_n(data, i - 2, 14, 20, "Close") and is_close_above_300EMA
            condition_2 = rsi_bullish_cand_2 and is_close_above_50EMA
            condition_3 = is_close_above_20EMA and pd_rsi_cross_n(data, i, 14, 50, "Close")
            if condition_1:
                #if high_break:
                # Buy condition met
                buy_date = data.index[i].date()
                bought_price = round(data.iloc[i]['High'], 2)  # Bought at the closing price
                quantity_bought = round(capital_per_stock / bought_price, 2)
                invested_amount += capital_per_stock
                #stop_loss = round((data.iloc[i - 1]['Low']), 2) # Set the Stop Loss as Previous candle Low
                #target = round(data.iloc[i + 9]['Close'], 2)  # Set the target as the 10th day close
                stop_loss = round(bought_price * (1 - stop_loss_percentage / 100), 2)
                target = round(bought_price * (1 + target_percentage / 100), 2)
                trade = {'Buy Date': buy_date, 'Bought Price': bought_price, 'Quantity Bought': quantity_bought,
                         'Invested Amount': capital_per_stock, 'Stop Loss': stop_loss, 'Target': target,
                         'Exited Date': None, 'Profit Amount': None}

                # Mark the buy signal in the DataFrame
                data.loc[data.index[i], 'Buy Signal'] = data.iloc[i]['Low'].astype(float)

                if trade and ((trade['Stop Loss'] >= data.iloc[i]['Low']) or (trade['Target'] <= data.iloc[i]['High'])):
                    # Sell condition met
                    sell_date = data.index[i].date()
                    profit_amount = round((data.iloc[i]['Close'] - trade['Bought Price']) * trade['Quantity Bought'], 2)
                    trade['Exited Date'] = sell_date  # Update Exited Date to target or stop loss hit date
                    trade['Profit Amount'] = profit_amount
                    invested_amount -= trade['Invested Amount']
                    trades.append(trade)
                    trade = None  # Reset trade

                    # Plotting the chart with buy signals
                    plt.figure(figsize=(12, 6))
                    plt.plot(data.index, data['Close'], label='Close Price')
                    plt.scatter(data.index, data['Buy Signal'], color='red', marker='^', label='Buy Signal')
                    plt.xlabel('Date')
                    plt.ylabel('Price')
                    plt.title('Chart with Buy Signals')
                    plt.legend()
                    plt.savefig(f'{Charts_Dir}/' + f'{stock}_plot.png')
                    #plt.show()

                    return trades


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

    # Create the Master DataFrame
    master_df = pd.DataFrame({
        'Stock Name': stock_names,
        'Total Trades': total_trades,
        'No of Winning Trade': total_winning_trades,
        'No of Losing Trade': total_losing_trades,
        'Winning Trade Percentage': total_winning_trade_percentage,
        'Losing Trade Percentage': total_losing_trade_percentage,
        'Total Profit': total_profit,
        'Total Cumulative Return Percentage': total_cumulative_return_percentage
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
        'Total Cumulative Return Percentage': round((sum(total_profit) / capital) * 100, 2)
    }

    # Append overall totals to the Master DataFrame
    master_df = pd.concat([master_df, pd.DataFrame(overall_totals, index=[0])], ignore_index=True)

    # Save Master DataFrame to CSV
    master_df.to_csv(f"{Master_Dir}/Master.csv", index=False)



import pandas as pd

def main(symbol, start_date, end_date, capital, target_percentage, stop_loss_percentage):
    try:
        # Fetch data from Yahoo Finance
        data = fetch_yahoo_finance_data(symbol, start_date, end_date)

        if data.empty:
            print(f"No data found for {symbol}. Skipping...")
            return  # Skip this stock

        # Calculate EMA
        data = calculate_ema(data, 300)
        data = calculate_ema(data, 200)
        data = calculate_ema(data, 50)
        data = calculate_ema(data, 20)

        # Calculate capital per stock
        capital_per_stock = round(capital / 10, 2)  # Assuming you're allowed to trade 10 stocks at a time

        # Check buying conditions and track trades
        trades = check_buy_conditions(data, capital_per_stock, target_percentage, stop_loss_percentage)

        if not trades:
            print(f"No trades found for {symbol}. Skipping...")
            return  # Skip this stock

        # Calculate No of holding Days
        df = pd.DataFrame(trades)
        df['Buy Date'] = pd.to_datetime(df['Buy Date'])  # Convert Buy Date to datetime format
        df['Exited Date'] = pd.to_datetime(df['Exited Date'])  # Convert Exited Date to datetime format
        df['No of holding Days'] = round((df['Exited Date'] - df['Buy Date']).dt.days, 2)  # Calculate holding days

        # Calculate Profit %
        df['Profit %'] = round((df['Profit Amount'] / df['Invested Amount']) * 100, 2)

        # Save DataFrame to CSV with Bought Price field
        df.to_csv(f"{Reports_Dir}/{stock}_trades.csv", index=False)

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
                round((sum([trade['Profit Amount'] for trade in trades if trade['Profit Amount']]) / capital) * 100, 2)]
        })

        # Save summary DataFrame to CSV
        summary_df.to_csv(f"{Summary_Dir}/{stock}_summary.csv", index=False)

    except Exception as e:
        print(f"Error processing {symbol}: {e}")


if __name__ == "__main__":
    # Define parameters
    symbols_file = 'custom.txt'
    from_date = '2015-01-01'  # 5 years ago from now
    to_date = '2019-12-31'  # Current date
    capital = 1000000
    target_percentage = 15  # Example target percentage
    stop_loss_percentage = 5  # Example stop loss percentage
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

    # Initialize lists to store summarized information
    summary_data = []

    for stock in stocks:
        # Call main function with parameters
        print(f"Backtesting for Symbol : {stock}")
        main(stock+str(".NS"), from_date, to_date, capital, target_percentage, stop_loss_percentage)
    # Call function to create the Master.csv file
    create_master_file(Summary_Dir)
