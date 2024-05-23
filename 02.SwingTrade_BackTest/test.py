def create_master_file_old(summary_dir):
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