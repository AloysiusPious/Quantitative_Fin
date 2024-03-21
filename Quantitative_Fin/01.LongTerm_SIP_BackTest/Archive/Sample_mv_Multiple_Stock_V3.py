'''

The provided code appears to analyze stock data from a CSV file (presumably containing data for TCS.NS, based on the file name) using Pandas. The script calculates the Exponential Moving Average (EMA) and checks for volume spikes.

Here's a brief explanation of what the code does:

It imports the required libraries: datetime for handling dates and pandas for data manipulation.
It defines constants such as START_DATE, END_DATE, EMA_PERIOD, and VOLUME_SPIKE_WINDOW.
It reads data from a CSV file named 'TCS.NS.csv' into a Pandas DataFrame.
It calculates the Exponential Moving Average (EMA) of closing prices over a specified period (EMA_PERIOD).
It iterates through the DataFrame rows starting from the 253rd row (as the EMA calculation requires preceding data) and checks for conditions where the EMA is greater than the closing price and the volume exceeds 1.5 times the average volume within a specified window (VOLUME_SPIKE_WINDOW).
The script prints the date and closing price when a volume spike (50% higher than the average volume) occurs alongside the condition where the EMA is greater than the closing price.

This code could be useful for identifying potential trading opportunities based on volume spikes and EMA crossovers. However, it's essential to perform further analysis and risk management before executing trades based solely on these signals. Additionally, backtesting such strategies can provide insights into their historical performance and help assess their viability.
With Custom Dates
print in table Stocks, invested_Amount, current_value_of_today, Cumulative_Percentage,
##############
Total_invested_Amount, Total_Cumulative_percentage
'''
import pandas as pd
import matplotlib.pyplot as plt

EMA_PERIOD = 200
VOLUME_SPIKE_WINDOW = 20
AMOUNT_TO_INVEST = 10000

# Read stocks from EMA_Swing.txt file
with open('../../../Archive/EMA_Swing.txt', 'r') as file:
    stocks = file.read().splitlines()

# Define FROM and TO dates
from_date = '2019-01-01'
to_date = '2024-12-31'

for stock in stocks:
    # Read CSV data for the stock
    dataframe = pd.read_csv(f'CSV_Day_Data/{stock}.csv')
    dataframe['Date'] = pd.to_datetime(dataframe['Date'])  # Convert 'Date' column to datetime

    # Filter DataFrame based on FROM and TO dates
    filtered_df = dataframe[(dataframe['Date'] >= from_date) & (dataframe['Date'] <= to_date)]


    # Calculate EMA
    #price_ema = filtered_df['Close'].ewm(span=EMA_PERIOD).mean()
    price_ema = filtered_df['Close'].ewm(span=EMA_PERIOD).mean()
    if len(filtered_df) < EMA_PERIOD:
        print(f"Insufficient data for {stock}. Skipping...")
        continue

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
                buy_dates.append(filtered_df['Date'].iloc[i])  # Append datetime object for buy date
                buy_prices.append(filtered_df['Close'].iloc[i])  # Append buy price
                num_stocks_to_buy = int(AMOUNT_TO_INVEST // filtered_df['Close'].iloc[i])
                buy_qty.append(int(AMOUNT_TO_INVEST // filtered_df['Close'].iloc[i]))  # Append buy quantity
                total_invested_amount += num_stocks_to_buy * filtered_df['Close'].iloc[i]
                current_value += num_stocks_to_buy * filtered_df['Close'].iloc[-1]

    # Calculate cumulative percentage return
    cumulative_percentage_return = ((current_value - total_invested_amount) / total_invested_amount) * 100

    # Create a DataFrame for buy dates and prices
    buy_df = pd.DataFrame({'Buy Date': buy_dates, 'Buy Price': buy_prices, 'Buy Qty': buy_qty})

    # Save buy DataFrame to CSV
    buy_df.to_csv(f'Reports/{stock}_buy_dates_prices.csv', index=False)

    # Print total invested amount, current value, and cumulative percentage return
    print(f'Stock: {stock}')
    print(f'Total Invested Amount: {total_invested_amount}')
    print(f'Current Value as of Today: {current_value}')
    print(f'Cumulative Percentage Return: {cumulative_percentage_return:.2f}%')

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(filtered_df['Date'], filtered_df['Close'], label='Close Price', color='blue')
    plt.scatter(buy_dates, buy_prices, color='red', marker='o', label='Buy Area')
    plt.title(f'Stock Price Movement of [ {stock} ] with Buy Areas ({from_date} to {to_date})')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)

    # Print stock name on the chart
    plt.text(filtered_df['Date'].iloc[0], filtered_df['Close'].min(), f'Stock: {stock}', fontsize=12, color='green')

    plt.show()