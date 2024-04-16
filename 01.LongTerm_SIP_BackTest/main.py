"""
Author: Aloysius Pious
Date: 28-02-2024
Email : aloysius.pious@gmail.com

Description:
Backtest : Invest in SIP Nifty 100 Stocks whenever there is a market fall
"""
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import csv
import os
import sys
import yfinance as yf
import talib as ta

################
sys.path.append('../my_lib')
from zerodha_data import *


########################################


def create_master_sheet(from_date, to_date):
    # Read the list of stock names from the file
    with open('./symbols/' + symbols_file, 'r') as file:
        stocks = file.read().splitlines()
    # Convert the from_date and to_date strings to datetime objects
    from_date = pd.to_datetime(from_date)
    to_date = pd.to_datetime(to_date)

    # Generate a date range from from_date to to_date with a frequency of 'MS' (Month Start)
    months_years = pd.date_range(start=from_date, end=to_date, freq='MS').strftime('%m-%Y').tolist()

    # Create an empty dictionary to hold data for the master CSV
    data = {'Stock': stocks}
    for month_year in months_years:
        data[month_year] = ['' for _ in range(len(stocks))]

    # Create the DataFrame for the master CSV
    master_df = pd.DataFrame(data)

    # Save the master DataFrame to a CSV file
    master_df.to_csv(f'{Master_Dir}/master.csv', index=False)


def create_directory(symbols_type):
    directories_to_create = [f'Reports_{from_date}_to_{to_date}', f'Charts_{from_date}_to_{to_date}',
                             f'Summary_{from_date}_to_{to_date}', f'Master_{from_date}_to_{to_date}']
    # Iterate over each directory and create it if it does not exist
    for directory in directories_to_create:
        directory_name = symbols_type + "_" + directory
        # print(directory_name)
        """Create directory if it does not exist"""
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
            # print(f"Directory '{directory_name}' created successfully.")
        # else:
        #   print(f"Directory '{directory_name}' already exists.")


def calculate_macd(df):
    """
    Calculate MACD indicator and determine True or False based on MACD and Signal line crossovers.
    Returns True if MACD line crosses above the Signal line, False otherwise.
    """
    short_window = 12
    long_window = 26
    signal_window = 9
    # Calculate short-term EMA
    short_ema = df['Close'].ewm(span=short_window, min_periods=1, adjust=False).mean()
    # Calculate long-term EMA
    long_ema = df['Close'].ewm(span=long_window, min_periods=1, adjust=False).mean()
    # Calculate MACD line
    macd_line = short_ema - long_ema
    # Calculate Signal line (EMA of MACD line)
    signal_line = macd_line.ewm(span=signal_window, min_periods=1, adjust=False).mean()
    # Determine crossover (MACD line crosses above Signal line)
    crossover = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
    # Return True if crossover occurred, False otherwise
    return crossover[-1]


def draw_chart(filtered_df, stock, buy_dates, buy_prices, EMA_PERIOD):
    if not filtered_df.empty:
        # Calculate EMA
        ema_values = filtered_df['Close'].ewm(span=EMA_PERIOD, adjust=False).mean()

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(filtered_df['Date'], filtered_df['Close'], label='Close Price', color='blue')
        plt.plot(filtered_df['Date'], ema_values, label=f'EMA ({EMA_PERIOD} periods)', color='orange')  # Plot EMA line
        plt.scatter(buy_dates, buy_prices, color='red', marker='o', label='Buy Area')
        plt.title(f'Stock Price Movement of [ {stock} ] with Buy Areas and EMA ({from_date} to {to_date})')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        # Print stock name on the chart
        plt.text(filtered_df['Date'].iloc[0], filtered_df['Close'].min(), f'Stock: {stock}', fontsize=12, color='green')
        # Save the plot in the subdirectory
        plt.savefig(f'{Charts_Dir}/' + f'{stock}_plot.png')
        # plt.show()


# Define function to fetch Yahoo Finance data
def fetch_yahoo_finance_data(symbol, start_date, end_date):
    data = yf.download(symbol + str('.NS'), start=start_date, end=end_date)
    return data


def low_less_than_n_low(filtered_df):
    # Calculate the least open of the past 7 days excluding today
    filtered_df['Past_7_Days_Min_Open'] = filtered_df['Open'].shift(1).rolling(window=8, min_periods=1).apply(
        lambda x: min(x), raw=True)

    # Compare today's open with the least open of the past 7 days
    today_open_less_than_past_7_days_min = filtered_df['Open'].iloc[-1] < filtered_df['Past_7_Days_Min_Open'].iloc[-1]

    # Print the result
    print(today_open_less_than_past_7_days_min)
    return today_open_less_than_past_7_days_min
    # The column 'Current_Low_Less_8_Days_Low' will contain 'True' where the current low is less than the past 8 days low, otherwise 'False'


def get_signal(i, filtered_df):
    # Calculate EMA
    price_ema = filtered_df['Close'].ewm(span=EMA_PERIOD).mean()
    # if price_ema.iloc[i] > filtered_df['Close'].iloc[i]:
    current_clo_green = filtered_df['Close'].iloc[i] > filtered_df['Open'].iloc[i]
    current_clo_red = filtered_df['Close'].iloc[i] < filtered_df['Open'].iloc[i]
    current_clo_gt_200ema = filtered_df['Close'].iloc[i] > price_ema.iloc[i]
    current_clo_lt_200ema = filtered_df['Close'].iloc[i] < price_ema.iloc[i]
    current_opn_lt_200ema = filtered_df['Open'].iloc[i] < price_ema.iloc[i]
    #############
    prev_1_red = filtered_df['Close'].iloc[i - 1] < filtered_df['Open'].iloc[i - 1]
    prev_2_red = filtered_df['Close'].iloc[i - 2] < filtered_df['Open'].iloc[i - 2]
    #############
    three_black_crow = current_clo_red and prev_1_red and prev_2_red
    #####
    prev_1_clo_lt_200ema = price_ema.iloc[i - 1] > filtered_df['Close'].iloc[i - 1]
    prev_2_clo_lt_200ema = price_ema.iloc[i - 2] > filtered_df['Close'].iloc[i - 2]
    prev_3_clo_lt_200ema = price_ema.iloc[i - 3] > filtered_df['Close'].iloc[i - 3]
    three_black_clo_lt_200ema = three_black_crow and current_clo_lt_200ema and prev_1_clo_lt_200ema and prev_2_clo_lt_200ema
    three_prev_clo_lt_200ema = prev_1_clo_lt_200ema and prev_2_clo_lt_200ema and prev_3_clo_lt_200ema
    #############################
    current_hig_lt_200ema = price_ema.iloc[i] > filtered_df['High'].iloc[i]
    prev_1_hig_lt_200ema = price_ema.iloc[i - 1] > filtered_df['High'].iloc[i - 1]
    prev_2_hig_lt_200ema = price_ema.iloc[i - 2] > filtered_df['High'].iloc[i - 2]
    three_hig_lt_200ema = current_hig_lt_200ema and prev_1_hig_lt_200ema and prev_2_hig_lt_200ema
    #######
    prev_1_opn_lt_200ema = price_ema.iloc[i - 1] > filtered_df['Open'].iloc[i - 1]
    prev_2_opn_lt_200ema = price_ema.iloc[i - 2] > filtered_df['Open'].iloc[i - 2]
    three_black_opn_lt_200ema = current_opn_lt_200ema and prev_1_opn_lt_200ema and prev_2_opn_lt_200ema
    #############
    prev_three_black_crow = filtered_df['Close'].iloc[i - 1] < filtered_df['Open'].iloc[i - 1] \
                            and filtered_df['Close'].iloc[i - 2] < filtered_df['Open'].iloc[i - 2] \
                            and filtered_df['Close'].iloc[i - 3] < filtered_df['Open'].iloc[i - 3]
    current_three_black_crow = filtered_df['Close'].iloc[i] < filtered_df['Open'].iloc[i] \
                               and filtered_df['Close'].iloc[i - 1] < filtered_df['Open'].iloc[i - 1] \
                               and filtered_df['Close'].iloc[i - 2] < filtered_df['Open'].iloc[i - 2]
    ## Three Black Crow closed less than previous candle
    three_black_clo_below_previous = filtered_df['Close'].iloc[i] < filtered_df['Close'].iloc[i - 1] \
                                     and filtered_df['Close'].iloc[i - 1] < filtered_df['Close'].iloc[i - 2]
    condition_4 = three_black_clo_lt_200ema and three_black_opn_lt_200ema and three_black_clo_below_previous

    return condition_4


def start_calc():
    # Read stocks from EMA_Swing.txt file
    with open('./symbols/' + symbols_file, 'r') as file:
        stocks = file.read().splitlines()

    # Initialize lists to store summarized information
    summary_data = []

    for stock in stocks:
        # Retrieve data for the stock from Yahoo Finance
        # dataframe = fetch_yahoo_finance_data(stock, from_date, to_date)
        # Retrieve data for the stock from Yahoo Finance
        dataframe = fetch_yahoo_finance_data(stock, from_date,
                                             to_date) if data_source == "yahoo" else get_data_from_zerodha(stock,
                                                                                                           from_date,
                                                                                                           to_date,
                                                                                                           time_frame)

        # Check if the index contains date information
        if dataframe is not None and not dataframe.empty and isinstance(dataframe.index, pd.DatetimeIndex):
            # Convert the index to datetime
            dataframe.index = pd.to_datetime(dataframe.index)
        # If you still need a 'Date' column, you can create it from the index
        dataframe['Date'] = dataframe.index

        # Filter DataFrame based on FROM and TO dates
        filtered_df = dataframe[(dataframe['Date'] >= from_date) & (dataframe['Date'] <= to_date)]

        total_invested_amount = 0  # Initialize total invested amount
        current_value = 0  # Initialize current value
        buy_dates = []  # List to store buy dates
        buy_prices = []  # List to store buy prices
        buy_qty = []

        # Find buy points
        for i in range(EMA_PERIOD, len(filtered_df)):
            if get_signal(i, filtered_df):
                # if low_less_than_n_low(filtered_df):#volume_spike(filtered_df, i):
                # if volume_spike(filtered_df, i):
                buy_date = filtered_df['Date'].iloc[i]

                # Check if entry already exists for this month-year combination
                month_year = buy_date.strftime('%m-%Y')
                master_csv_path = f'{Master_Dir}/master.csv'
                if os.path.exists(master_csv_path):
                    master_df = pd.read_csv(master_csv_path, index_col=0)
                else:
                    master_df = pd.DataFrame()

                if stock not in master_df.index:
                    master_df.loc[stock] = ""

                # Check if the stock has already been bought in this month
                if month_year in master_df.columns and master_df.at[stock, month_year] == "BUY":
                    continue  # Skip buying for this month if already bought

                # Update master CSV with "BUY" entry
                master_df[month_year] = master_df[month_year].astype(str)
                master_df.at[stock, month_year] = "BUY"
                master_df.to_csv(master_csv_path)

                # Append buy information to lists
                buy_dates.append(filtered_df['Date'].iloc[i])  # Append datetime object for buy date
                buy_prices.append(round(filtered_df['Close'].iloc[i], 2))  # Append buy price
                num_stocks_to_buy = int(AMOUNT_TO_INVEST // filtered_df['Close'].iloc[i])
                buy_qty.append(round(num_stocks_to_buy, 2))  # Append buy quantity
                total_invested_amount += round(num_stocks_to_buy * filtered_df['Close'].iloc[i], 2)
                current_value += round(num_stocks_to_buy * filtered_df['Close'].iloc[-1], 2)

        # Calculate cumulative percentage return
        if total_invested_amount != 0:
            cumulative_percentage_return = ((current_value - total_invested_amount) / total_invested_amount) * 100
        else:
            cumulative_percentage_return = 0

        # Append summarized data to the list
        # summary_data.append([stock, total_invested_amount, current_value, cumulative_percentage_return])
        summary_data.append(
            [stock, round(total_invested_amount, 2), round(current_value, 2), round(cumulative_percentage_return, 2)])

        # Create a DataFrame for buy dates and prices
        buy_df = pd.DataFrame({'Buy Date': buy_dates, 'Buy Price': buy_prices, 'Buy Qty': buy_qty})

        # Save buy DataFrame to CSV
        buy_df.to_csv(f'{Reports_Dir}/{stock}_buy_dates_prices.csv', index=False)
        # Assuming you have this function defined
        draw_chart(filtered_df, stock, buy_dates, buy_prices, EMA_PERIOD)

    # Convert summary data to DataFrame
    summary_df = pd.DataFrame(summary_data,
                              columns=['Stocks', 'Invested Amount', 'Current Value of Today', 'Cumulative Percentage'])

    # Round numeric columns to two decimal places
    summary_df[['Invested Amount', 'Current Value of Today', 'Cumulative Percentage']] = summary_df[
        ['Invested Amount', 'Current Value of Today', 'Cumulative Percentage']].round(2)

    # Define the file path for the summary CSV
    summary_csv_path = f'{Summary_Dir}/summary_{from_date}_to_{to_date}.csv'

    # Write the summary data to the CSV file
    summary_df.to_csv(summary_csv_path, index=False)

    # Print the summary data in tabular format
    print(
        tabulate(summary_data, headers=['Stocks', 'Invested Amount', 'Current Value of Today', 'Cumulative Percentage'],
                 tablefmt='pretty'))

    # Calculate total invested amount and total cumulative percentage
    total_invested_amount = sum(row[1] for row in summary_data)
    total_current_value = sum(row[2] for row in summary_data)
    total_cumulative_percentage = ((total_current_value - total_invested_amount) / total_invested_amount) * 100

    # Print total aggregated information
    print('Total Invested Amount:', round(total_invested_amount, 2))
    print('Total Current Value as of Today:', round(total_current_value, 2))
    print('Total Cumulative Percentage Return:', f'{total_cumulative_percentage:.2f}%')

    # Open the summary CSV file in append mode
    with open(summary_csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([])  # Add an empty row for separation
        writer.writerow(['AMOUNT_TO_INVEST', round(AMOUNT_TO_INVEST, 2)])
        writer.writerow(['Total Invested Amount', round(total_invested_amount, 2)])
        writer.writerow(['Total Current Value as of Today', round(total_current_value, 2)])
        writer.writerow(['Total Cumulative Percentage Return', f'{total_cumulative_percentage:.2f}%'])
        writer.writerow(['Start_Date', f'{str(from_date):}'])
        writer.writerow(['End_Date', f'{str(to_date):}'])


# You need to define the fetch_yahoo_finance_data() and draw_chart() functions.
##################

symbols_file = 'nifty_100.txt'
# symbols_file = 'next_50.txt'
# symbols_file = 'nifty_future.txt'
# symbols_file = 'nifty_500.txt'
# symbols_file = 'equity_cash_greater_100.txt'
# symbols_file = "less_than_bookval_3x.txt"
# symbols_file = 'custom.txt'
# symbols_file = 'large_cap.txt'
# Extracting 'next_50' from symbols_file

# Define constants
EMA_PERIOD = 200
VOLUME_SPIKE_WINDOW = 20
AMOUNT_TO_INVEST = 3500
from_date = '2018-01-01'
to_date = '2024-12-31'
time_frame = 'day'
#############
data_source = "yahoo"
# data_source = "zerodha"

print(f"********** retrieving  Data Source From {data_source} *************")
##############
symbols_type = symbols_file.split('.')[0]
Reports_Dir = f'{symbols_type}_Reports_{from_date}_to_{to_date}'
Charts_Dir = f'{symbols_type}_Charts_{from_date}_to_{to_date}'
Summary_Dir = f'{symbols_type}_Summary_{from_date}_to_{to_date}'
Master_Dir = f'{symbols_type}_Master_{from_date}_to_{to_date}'
##############
create_directory(symbols_type)
create_master_sheet(from_date, to_date)
start_calc()
