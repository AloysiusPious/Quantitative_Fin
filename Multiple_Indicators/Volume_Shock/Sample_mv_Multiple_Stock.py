import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import csv
import os
import sys
sys.path.append('/Users/aloysiuspious/Personal/Algo/BackTest/my_lib')
from yfin import *

def create_master_sheet(from_date, to_date):
    # Read the list of stock names from the file
    with open('../symbols/' + symbols_file, 'r') as file:
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
    master_df.to_csv('master.csv', index=False)
def create_directory():
    directories_to_create = ["Reports", "Charts", "Summary"]
    # Iterate over each directory and create it if it does not exist
    for directory in directories_to_create:
        """Create directory if it does not exist"""
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory '{directory}' created successfully.")
        else:
            print(f"Directory '{directory}' already exists.")
def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    """
    Calculate MACD indicator and determine True or False based on MACD and Signal line crossovers.
    Returns True if MACD line crosses above the Signal line, False otherwise.
    """
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
    plt.savefig("Charts/" + f'{stock}_plot.png')
    # plt.show()

def start_calc():

    # Read stocks from EMA_Swing.txt file
    with open('../symbols/'+symbols_file, 'r') as file:
        stocks = file.read().splitlines()

    # Initialize lists to store summarized information
    summary_data = []

    for stock in stocks:
        # Retrieve data for the stock from Yahoo Finance
        dataframe = get_df_from_yf(stock, from_date, to_date)

        # Check if the index contains date information
        if isinstance(dataframe.index, pd.DatetimeIndex):
            # Convert the index to datetime
            dataframe.index = pd.to_datetime(dataframe.index)

        # If you still need a 'Date' column, you can create it from the index
        dataframe['Date'] = dataframe.index

        # Filter DataFrame based on FROM and TO dates
        filtered_df = dataframe[(dataframe['Date'] >= from_date) & (dataframe['Date'] <= to_date)]

        # Calculate EMA
        price_ema = filtered_df['Close'].ewm(span=EMA_PERIOD).mean()

        total_invested_amount = 0  # Initialize total invested amount
        current_value = 0  # Initialize current value
        buy_dates = []  # List to store buy dates
        buy_prices = []  # List to store buy prices
        buy_qty = []

        # Find buy points
        for i in range(EMA_PERIOD, len(filtered_df)):
            if price_ema.iloc[i] > filtered_df['Close'].iloc[i]:
                # Check for volume spike
                volume_avg = filtered_df['Volume'].iloc[i - VOLUME_SPIKE_WINDOW:i].mean()
                if volume_avg * 1.5 < filtered_df['Volume'].iloc[i]:
                    buy_date = filtered_df['Date'].iloc[i]

                    # Check if entry already exists for this month-year combination
                    month_year = buy_date.strftime('%m-%Y')
                    master_csv_path = f'master.csv'
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
        #summary_data.append([stock, total_invested_amount, current_value, cumulative_percentage_return])
        summary_data.append(
            [stock, round(total_invested_amount, 2), round(current_value, 2), round(cumulative_percentage_return, 2)])

        # Create a DataFrame for buy dates and prices
        buy_df = pd.DataFrame({'Buy Date': buy_dates, 'Buy Price': buy_prices, 'Buy Qty': buy_qty})

        # Save buy DataFrame to CSV
        buy_df.to_csv(f'Reports/{stock}_buy_dates_prices.csv', index=False)
        # Assuming you have this function defined
        draw_chart(filtered_df, stock, buy_dates, buy_prices, EMA_PERIOD)

    # Print summarized data in a table
    # Define the file path for the summary CSV
    summary_csv_path = "summary.csv"

    # Write the summary data to the CSV file
    with open(summary_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Stocks', 'Invested Amount', 'Current Value of Today', 'Cumulative Percentage'])
        for row in summary_data:
            writer.writerow([row[0], round(row[1], 2), round(row[2], 2), round(row[3], 2)])

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
        writer.writerow(['Total Invested Amount', round(total_invested_amount, 2)])
        writer.writerow(['Total Current Value as of Today', round(total_current_value, 2)])
        writer.writerow(['Total Cumulative Percentage Return', f'{total_cumulative_percentage:.2f}%'])

# You need to define the get_df_from_yf() and draw_chart() functions.
##################

symbols_file = 'nifty_50.txt'
#symbols_file = 'custom.txt'
# Define constants
EMA_PERIOD = 200
VOLUME_SPIKE_WINDOW = 20
AMOUNT_TO_INVEST = 20000
from_date = '2019-01-01'
to_date = '2024-12-31'
##############
create_directory()
create_master_sheet(from_date, to_date)
start_calc()