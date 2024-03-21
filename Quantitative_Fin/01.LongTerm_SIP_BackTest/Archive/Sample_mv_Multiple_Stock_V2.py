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
'''
import pandas as pd
import matplotlib.pyplot as plt

EMA_PERIOD = 200
VOLUME_SPIKE_WINDOW = 20
AMOUNT_TO_INVEST = 10000

# Read stocks from EMA_Swing.txt file
with open('../../../Archive/EMA_Swing.txt', 'r') as file:
    stocks = file.read().splitlines()

for stock in stocks:
    # Read CSV data for the stock
    #stock='CSV_Day_Data/'+stock
    dataframe = pd.read_csv(str('CSV_Day_Data/')+stock+str('.csv'))
    dataframe['Date'] = pd.to_datetime(dataframe['Date'])  # Convert 'Date' column to datetime

    # Calculate EMA
    price_ema = dataframe['Close'].ewm(span=EMA_PERIOD).mean()

    total_invested_amount = 0  # Initialize total invested amount
    current_value = 0  # Initialize current value
    buy_dates = []  # List to store buy dates
    buy_prices = []  # List to store buy prices
    buy_qty = []

    # Find buy points
    for i in range(EMA_PERIOD, len(dataframe)):
        print(price_ema[i])
        if price_ema[i] > dataframe['Close'][i]:
            # Check for volume spike
            volume_avg = dataframe['Volume'].iloc[i - VOLUME_SPIKE_WINDOW:i].mean()
            if volume_avg * 1.5 < dataframe['Volume'][i]:
                buy_dates.append(dataframe['Date'].iloc[i])  # Append datetime object for buy date
                buy_prices.append(dataframe['Close'].iloc[i])  # Append buy price
                #buy_qty.append(int(AMOUNT_TO_INVEST / dataframe['Close'].iloc[i]))  # Append buy price
                # Calculate number of stocks to buy with 10000 rupees
                num_stocks_to_buy = int(AMOUNT_TO_INVEST // dataframe['Close'][i])
                buy_qty.append(int(AMOUNT_TO_INVEST // dataframe['Close'].iloc[i]))  # Append buy price
                total_invested_amount += num_stocks_to_buy * dataframe['Close'][i]  # Add the amount spent on buying stocks
                current_value += num_stocks_to_buy * dataframe['Close'].iloc[-1]  # Add the current value of the investment

    # Calculate cumulative percentage return
    cumulative_percentage_return = ((current_value - total_invested_amount) / total_invested_amount) * 100

    # Create a DataFrame for buy dates and prices
    buy_df = pd.DataFrame({'Buy Date': buy_dates, 'Buy Price': buy_prices, 'Buy Qty': buy_qty})

    # Save buy DataFrame to CSV
    buy_df.to_csv(str('Reports/')+f'{stock}_buy_dates_prices.csv', index=False)

    # Print total invested amount, current value, and cumulative percentage return
    print(f'Stock: {stock}')
    print(f'Total Invested Amount: {total_invested_amount}')
    print(f'Current Value as of Today: {current_value}')
    print(f'Cumulative Percentage Return: {cumulative_percentage_return:.2f}%')

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(dataframe['Date'], dataframe['Close'], label='Close Price', color='blue')
    plt.scatter(buy_dates, buy_prices, color='red', marker='o', label='Buy Area')
    plt.title(f'Stock Price Movement of [ {stock} ] with Buy Areas')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)

    # Print stock name on the chart
    plt.text(dataframe['Date'].iloc[0], dataframe['Close'].min(), f'Stock: {stock}', fontsize=12, color='green')

    plt.show()
