import pandas as pd

# Define the list of stock names
stocks = ['AAPL', 'TSLA', 'GOOGL']  # Example list of stock names

# Define the list of months with years
months_years = ['Jan-2023', 'Feb-2023', 'Mar-2023', 'Apr-2023', 'May-2023', 'Jun-2023',
                'Jul-2023', 'Aug-2023', 'Sep-2023', 'Oct-2023', 'Nov-2023', 'Dec-2023',
                'Jan-2024', 'Feb-2024', 'Mar-2024', 'Apr-2024', 'May-2024', 'Jun-2024',
                'Jul-2024', 'Aug-2024', 'Sep-2024', 'Oct-2024', 'Nov-2024', 'Dec-2024']

# Create an empty dictionary to hold data for the master CSV
data = {'Stock': stocks}
for month_year in months_years:
    data[month_year] = ['' for _ in range(len(stocks))]

# Create the DataFrame for the master CSV
master_df = pd.DataFrame(data)

# Save the master DataFrame to a CSV file
master_df.to_csv('master.csv', index=False)