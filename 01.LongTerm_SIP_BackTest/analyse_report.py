import pandas as pd
import os


def analyze_investments(directory):
    # Initialize an empty DataFrame to store all data
    all_data = pd.DataFrame()

    # Read all CSV files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            if 'Buy Date' in df.columns and 'Buy Price' in df.columns and 'Buy Qty' in df.columns:
                df['Buy Date'] = pd.to_datetime(df['Buy Date'])
                if not df.empty:
                    all_data = pd.concat([all_data, df], ignore_index=True)

    if all_data.empty:
        print("No valid data found in the CSV files.")
        return

    # Extract month and year from 'Buy Date'
    all_data['YearMonth'] = all_data['Buy Date'].dt.to_period('M')

    # Calculate the monthly total investment
    all_data['Investment'] = all_data['Buy Price'] * all_data['Buy Qty']
    monthly_investment = all_data.groupby('YearMonth')['Investment'].sum().reset_index()

    # Calculate the monthly average investment
    monthly_avg_investment = monthly_investment['Investment'].mean()

    # Find the month with the most and least investment
    most_invest_month = monthly_investment.loc[monthly_investment['Investment'].idxmax()]
    least_invest_month = monthly_investment.loc[monthly_investment['Investment'].idxmin()]

    # Additional analysis
    # Calculate the total number of transactions per month
    transactions_per_month = all_data.groupby('YearMonth').size().reset_index(name='Transactions')

    # Rate the months based on investment amount
    monthly_investment['Rating'] = monthly_investment['Investment'].rank(ascending=False, method='min')

    # Merge the transactions per month with the monthly investment data
    monthly_investment = pd.merge(monthly_investment, transactions_per_month, on='YearMonth')

    # Prepare the results DataFrame
    result_df = pd.DataFrame({
        'YearMonth': monthly_investment['YearMonth'].astype(str),
        'Monthly Investment': monthly_investment['Investment'],
        'Transactions': monthly_investment['Transactions'],
        'Rating': monthly_investment['Rating']
    })

    # Add the summary rows
    summary_data = {
        'YearMonth': ['Average Investment', 'Most Investment Month', 'Most Investment Amount', 'Least Investment Month',
                      'Least Investment Amount'],
        'Monthly Investment': [
            monthly_avg_investment,
            most_invest_month['YearMonth'].strftime('%Y-%m'),
            most_invest_month['Investment'],
            least_invest_month['YearMonth'].strftime('%Y-%m'),
            least_invest_month['Investment']
        ],
        'Transactions': [None, None, None, None, None],  # Placeholder for summary rows
        'Rating': [None, None, None, None, None]  # Placeholder for summary rows
    }
    summary_df = pd.DataFrame(summary_data)

    # Remove all-NA columns from each DataFrame
    result_df = result_df.dropna(axis=1, how='all')
    summary_df = summary_df.dropna(axis=1, how='all')

    # Concatenate the DataFrames
    result_df = pd.concat([result_df, summary_df], ignore_index=True)

    # Save the results to a CSV file
    result_file_path = os.path.join(directory, 'investment_analysis.csv')
    result_df.to_csv(result_file_path, index=False)

    # Print the results
    print("Monthly Average Investment: ", monthly_avg_investment)
    print("Most Investment Month: ", most_invest_month['YearMonth'].strftime('%Y-%m'), " Amount: ",
          most_invest_month['Investment'])
    print("Least Investment Month: ", least_invest_month['YearMonth'].strftime('%Y-%m'), " Amount: ",
          least_invest_month['Investment'])

#    print("\nInvestment Amount for Each Month:")
 #   for _, row in monthly_investment.iterrows():
 #       print(
 #           f"Month: {row['YearMonth']} Investment: {row['Investment']} Transactions: {row['Transactions']} Rating: {row['Rating']}")


symbols_file = 'nifty_100.txt'
# symbols_file = 'next_50.txt'
# symbols_file = 'nifty_future.txt'
# symbols_file = 'nifty_500.txt'
# symbols_file = 'equity_cash_greater_100.txt'
# symbols_file = "less_than_bookval_3x.txt"
# symbols_file = 'custom.txt'
# symbols_file = 'large_cap.txt'
# Extracting 'next_50' from symbols_file
from_date = '2014-01-01'
to_date = '2023-12-31'

##############
symbols_type = symbols_file.split('.')[0]
Reports_Dir = f'{symbols_type}_Reports_{from_date}_to_{to_date}'
# Example usage
analyze_investments(Reports_Dir)