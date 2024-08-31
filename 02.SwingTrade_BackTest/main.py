import configparser
from matplotlib.offsetbox import AnchoredText
from swing_util import *
import time

def draw_down_chart():
    all_trades = []
    for filename in os.listdir(Reports_Dir):
        if filename.endswith(".csv"):
            filepath = os.path.join(Reports_Dir, filename)
            df = pd.read_csv(filepath, parse_dates=['Buy Date', 'Exited Date'])
            all_trades.append(df)
    all_trades = pd.concat(all_trades, ignore_index=True)
    all_trades = all_trades.sort_values(by='Buy Date')

    # Calculate cumulative profit
    all_trades['Cumulative Profit'] = all_trades['Profit Amount'].cumsum()

    # Calculate capital over time
    all_trades['Capital'] = capital + all_trades['Cumulative Profit']

    # Fetch Nifty 50 data within the specified date range
    nifty50_data = get_nifty50_data(from_date, to_date)

    # Calculate the percentage increase
    final_capital = all_trades['Capital'].iloc[-1]
    percentage_increase = ((final_capital - capital) / capital) * 100

    # Create a figure and a set of subplots
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Plotting the capital growth
    ax1.plot(all_trades['Buy Date'], all_trades['Capital'], marker='', linestyle='-', color='b',
             label='Capital Over Time')
    ax1.annotate(f'Start: ₹{capital}', xy=(all_trades['Buy Date'].iloc[0], capital),
                 xytext=(all_trades['Buy Date'].iloc[0], capital),
                 arrowprops=dict(facecolor='green', shrink=0.05))
    ax1.annotate(f'End: ₹{final_capital:.2f} ({percentage_increase:.2f}%)',
                 xy=(all_trades['Buy Date'].iloc[-1], final_capital),
                 xytext=(all_trades['Buy Date'].iloc[-1], final_capital),
                 arrowprops=dict(facecolor='red', shrink=0.05))

    # Set labels for the first y-axis
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Capital (₹)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.legend(loc='upper left')

    # Create a second y-axis to plot the Nifty 50 index
    ax2 = ax1.twinx()
    ax2.plot(nifty50_data.index, nifty50_data['Close'], linestyle='--', color='orange', label='Nifty 50')
    ax2.set_ylabel('Nifty 50 Index', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax2.legend(loc='upper center')

    # Add grid, title, and layout settings
    plt.title('Capital Growth Over Time and Nifty 50 Index')
    fig.tight_layout()
    plt.grid(True)

    # Adjust text box to fit inside the chart window
    wrapped_condition = condition.replace(" and ", " and\n")
    textstr = '\n'.join((
        f'capital = {capital}',
        f'no_of_stock_to_trade = {no_of_stock_to_trade}',
        f'compound = {compound}',
        f'target_percentage = {target_percentage}',
        f'stop_loss_percentage = {stop_loss_percentage}',
        f'trade_Logic:\n{wrapped_condition}'
    ))

    anchored_text = AnchoredText(textstr, loc='lower right', frameon=True, bbox_to_anchor=(1, 0.15),
                                 bbox_transform=ax1.transAxes, prop=dict(size=8))  # Reduce font size to 8
    anchored_text.patch.set_boxstyle("round,pad=0.5,rounding_size=0.5")
    ax1.add_artist(anchored_text)

    # Save the plot
    plt.savefig(f'{Charts_Dir}/draw_down_{symbols_type}_{from_date}_to_{to_date}.png')
    plt.show()
    plt.close()

def create_master_file(summary_dir, from_date, to_date, capital):
    # Initialize lists to store data
    stock_names = []
    total_trades = []
    total_winning_trades = []
    total_losing_trades = []
    total_winning_trade_percentage = []
    total_losing_trade_percentage = []
    total_profit = []
    total_cumulative_return_percentage = []
    total_charges_paid_list = []
    total_profit_after_charges = []
    total_profit_percentage_after_charges = []
    total_charges_percentage_list = []

    # Iterate over the summary files for each stock
    for filename in os.listdir(summary_dir):
        if filename.endswith("_summary.csv"):
            stock_name = filename.split("_")[0]  # Extract stock name from filename
            df_summary = pd.read_csv(os.path.join(summary_dir, filename))

            # Extract data from the summary DataFrame
            stock_names.append(stock_name)
            total_trades.append(df_summary['Total Trades'].values[0])
            total_winning_trades.append(df_summary['No of Winning Trades'].values[0])
            total_losing_trades.append(df_summary['No of Losing Trades'].values[0])
            total_winning_trade_percentage.append(round(df_summary['Winning Trade Percentage'].values[0], 2))
            total_losing_trade_percentage.append(round(df_summary['Losing Trade Percentage'].values[0], 2))
            total_profit.append(round(df_summary['Total Profit'].values[0], 2))
            total_cumulative_return_percentage.append(round(df_summary['Cumulative Return Percentage'].values[0], 2))
            total_charges_paid = round(df_summary['Total Charges Paid'].values[0], 2)
            total_charges_paid_list.append(total_charges_paid)

            # Calculate profit after charges and its percentage
            profit_after_charges = round(total_profit[-1] - total_charges_paid, 2)
            total_profit_after_charges.append(profit_after_charges)
            profit_percentage_after_charges = round((profit_after_charges / capital) * 100, 2)
            total_profit_percentage_after_charges.append(profit_percentage_after_charges)

            # Calculate total charges as a percentage of capital
            total_charges_percentage = round((total_charges_paid / capital) * 100, 2)
            total_charges_percentage_list.append(total_charges_percentage)

    # Create the Master DataFrame
    master_df = pd.DataFrame({
        'Stock Name': stock_names,
        'Total Trades': total_trades,
        'No of Winning Trade': total_winning_trades,
        'No of Losing Trade': total_losing_trades,
        'Winning Trade Percentage': total_winning_trade_percentage,
        'Losing Trade Percentage': total_losing_trade_percentage,
        'Total Profit': total_profit,
        'Total Cumulative Return Percentage': total_cumulative_return_percentage,
        'Total Charges Paid': total_charges_paid_list,
        'Profit After Charges': total_profit_after_charges,
        'Profit Percentage After Charges': total_profit_percentage_after_charges,
        'Total Charges Percentage': total_charges_percentage_list
    })

    # Calculate overall totals
    overall_totals = {
        'Stock Name': 'Overall',
        'Total Trades': sum(total_trades),
        'No of Winning Trade': sum(total_winning_trades),
        'No of Losing Trade': sum(total_losing_trades),
        'Winning Trade Percentage': round((sum(total_winning_trades) / sum(total_trades)) * 100, 2) if sum(total_trades) > 0 else 0,
        'Losing Trade Percentage': round((sum(total_losing_trades) / sum(total_trades)) * 100, 2) if sum(total_trades) > 0 else 0,
        'Total Profit': round(sum(total_profit), 2),
        'Total Cumulative Return Percentage': round((sum(total_profit) / capital) * 100, 2),
        'Total Charges Paid': round(sum(total_charges_paid_list), 2),
        'Profit After Charges': round(sum(total_profit_after_charges), 2),
        'Profit Percentage After Charges': round((sum(total_profit_after_charges) / capital) * 100, 2),
        'Total Charges Percentage': round((sum(total_charges_paid_list) / capital) * 100, 2)
    }

    # Append overall totals to the Master DataFrame
    master_df = pd.concat([master_df, pd.DataFrame(overall_totals, index=[0])], ignore_index=True)

    # Create the directory if it doesn't exist
    master_dir = f'{symbols_type}_Master_{from_date}_to_{to_date}'
    os.makedirs(master_dir, exist_ok=True)

    # Save Master DataFrame to CSV
    master_df.to_csv(f"{master_dir}/Master_{from_date}_to_{to_date}.csv", index=False)

    print(f"Master file created successfully at {master_dir}/Master_{from_date}_to_{to_date}.csv")



# Function to calculate summary statistics for each stock
def calculate_summary_per_stock(symbol, report_df, Summary_Dir):
    total_trades = len(report_df)
    if total_trades == 0:
        return {
            'Symbol': symbol,
            'Total Trades': 0,
            'No of Winning Trades': 0,
            'No of Losing Trades': 0,
            'Winning Trade Percentage': 0,
            'Losing Trade Percentage': 0,
            'Total Profit': 0,
            'Cumulative Return Percentage': 0,
            'Total Charges Paid': 0
        }

    winning_trades = sum(1 for trade in report_df if trade['Trade Status'] == 'Target')
    losing_trades = total_trades - winning_trades if total_trades > 0 else 0

    winning_trade_percentage = (winning_trades / total_trades) * 100
    losing_trade_percentage = (losing_trades / total_trades) * 100
    total_profit = sum(trade['Profit Amount'] for trade in report_df)
    total_charges_paid = sum(trade['Invested Amount'] * charges_percentage / 100 for trade in report_df)
    cumulative_return_percentage = (total_profit / capital) * 100

    summary = {
        'Symbol': symbol,
        'Total Trades': total_trades,
        'No of Winning Trades': winning_trades,
        'No of Losing Trades': losing_trades,
        'Winning Trade Percentage': round(winning_trade_percentage, 2),
        'Losing Trade Percentage': round(losing_trade_percentage, 2),
        'Total Profit': round(total_profit, 2),
        'Cumulative Return Percentage': round(cumulative_return_percentage, 2),
        'Total Charges Paid': round(total_charges_paid, 2)
    }

    # Create summary DataFrame
    summary_df = pd.DataFrame([summary])

    # Write summary to CSV
    os.makedirs(Summary_Dir, exist_ok=True)
    summary_df.to_csv(f"{Summary_Dir}/{symbol}_summary.csv", index=False)

    return summary
# Function to calculate the number of holding days
def calculate_holding_days(buy_date, exit_date):
    return (pd.to_datetime(exit_date) - pd.to_datetime(buy_date)).days


# Read all stock CSV files from the specified directory except 'stock_date_ref.csv'
def read_stock_files():
    stock_files = sorted(glob.glob(f'{cvs_data_dir}/*.csv'))
    stock_files = [f for f in stock_files if not f.endswith('stock_date_ref.csv')]
    return stock_files

def process_date(date):
    global active_positions, capital_per_stock, final_capital
    # Get the last date
    last_date = date_ref_df['Date'].max()
    stock_files = read_stock_files()
    for stock_file in stock_files:
        stock_df = pd.read_csv(stock_file)
        stock_data = stock_df[stock_df['Date'] == date]

        if not stock_data.empty:
            row = stock_data.iloc[0]
            symbol = os.path.basename(stock_file).replace('.csv', '')

            if symbol not in trade_report:
                trade_report[symbol] = []

            if 'Buy_Signal' in row and not pd.isna(row['Buy_Signal']):
                active_for_stock = [p for p in active_positions if p['symbol'] == symbol]

                if len(active_for_stock) == 0 and len(active_positions) < no_of_stock_to_trade:
                    buy_price = row['Buy_Signal']
                    target_price = row['Target']
                    stop_loss_price = row['StopLoss']
                    shares_to_buy = int(capital_per_stock // buy_price)
                    invested_amount = shares_to_buy * buy_price
                    active_positions.append({
                        'symbol': symbol,
                        'buy_date': date,
                        'buy_price': buy_price,
                        'target_price': target_price,
                        'stop_loss_price': stop_loss_price,
                        'shares': shares_to_buy
                    })
                    portfolio[symbol] = {
                        'Buy Date': date,
                        'Bought Price': buy_price,
                        'Quantity Bought': shares_to_buy,
                        'Invested Amount': invested_amount,
                        'Stop Loss': stop_loss_price,
                        'Target': target_price,
                        'Exited Date': None,
                        'Exited Price': None,
                        'Profit Amount': None,
                        'Trade Status': None,
                        'No of holding Days': None,
                        'Profit %': None
                    }
                    print(f"Bought {symbol} on {date} at {buy_price}")

            for position in active_positions:
                if position['symbol'] == symbol:
                    if row['High'] >= position['target_price']:
                        sell_price = position['target_price']
                        profit_amount = (sell_price - position['buy_price']) * position['shares']
                        invested_amount = position['buy_price'] * position['shares']
                        profit_percent = (profit_amount / invested_amount) * 100
                        active_positions = [p for p in active_positions if p['symbol'] != symbol]
                        trade_report[symbol].append({
                            'Buy Date': position['buy_date'],
                            'Bought Price': round_to_nearest_0_05(position['buy_price']),
                            'Quantity Bought': position['shares'],
                            'Invested Amount': round_to_nearest_0_05(invested_amount),
                            'Stop Loss': round_to_nearest_0_05(position['stop_loss_price']),
                            'Target': round_to_nearest_0_05(position['target_price']),
                            'Exited Date': date,
                            'Exited Price': round_to_nearest_0_05(sell_price),
                            'Profit Amount': round_to_nearest_0_05(profit_amount),
                            'Trade Status': 'Target',
                            'No of holding Days': calculate_holding_days(position['buy_date'], date),
                            'Profit %': round_to_nearest_0_05(profit_percent)
                        })
                        final_capital += profit_amount
                        print(f"Sold {symbol} on {date} at {sell_price} (Target hit)")

                        if compound:
                            capital_per_stock = final_capital / no_of_stock_to_trade

                    elif row['Low'] <= position['stop_loss_price']:
                        sell_price = position['stop_loss_price']
                        profit_amount = (sell_price - position['buy_price']) * position['shares']
                        invested_amount = position['buy_price'] * position['shares']
                        profit_percent = (profit_amount / invested_amount) * 100
                        active_positions = [p for p in active_positions if p['symbol'] != symbol]
                        trade_report[symbol].append({
                            'Buy Date': position['buy_date'],
                            'Bought Price': round_to_nearest_0_05(position['buy_price']),
                            'Quantity Bought': position['shares'],
                            'Invested Amount': round_to_nearest_0_05(invested_amount),
                            'Stop Loss': round_to_nearest_0_05(position['stop_loss_price']),
                            'Target': round_to_nearest_0_05(position['target_price']),
                            'Exited Date': date,
                            'Exited Price': round_to_nearest_0_05(sell_price),
                            'Profit Amount': round_to_nearest_0_05(profit_amount),
                            'Trade Status': 'StopLoss',
                            'No of holding Days': calculate_holding_days(position['buy_date'], date),
                            'Profit %': round_to_nearest_0_05(profit_percent)
                        })
                        final_capital += profit_amount
                        print(f"Sold {symbol} on {date} at {sell_price} (Stop Loss hit)")

                        if compound:
                            capital_per_stock = final_capital / no_of_stock_to_trade
        if date == last_date:
            last_day_data = stock_df[stock_df['Date'] <= to_date].tail(1)
            if not last_day_data.empty:
                last_close_price = last_day_data.iloc[-1]['Close']
                for position in active_positions:
                    if position['symbol'] == symbol:
                        profit_amount = (last_close_price - position['buy_price']) * position['shares']
                        invested_amount = position['buy_price'] * position['shares']
                        profit_percent = (profit_amount / invested_amount) * 100
                        trade_report[symbol].append({
                            'Buy Date': position['buy_date'],
                            'Bought Price': round_to_nearest_0_05(position['buy_price']),
                            'Quantity Bought': position['shares'],
                            'Invested Amount': round_to_nearest_0_05(invested_amount),
                            'Stop Loss': round_to_nearest_0_05(position['stop_loss_price']),
                            'Target': round_to_nearest_0_05(position['target_price']),
                            'Exited Date': last_day_data.iloc[-1]['Date'],
                            'Exited Price': round_to_nearest_0_05(last_close_price),
                            'Profit Amount': round_to_nearest_0_05(profit_amount),
                            'Trade Status': 'LastDayClose',
                            'No of holding Days': calculate_holding_days(position['buy_date'], last_day_data.iloc[-1]['Date']),
                            'Profit %': round_to_nearest_0_05(profit_percent)
                        })
                        final_capital += profit_amount
                        print(f"Sold {symbol} on {last_day_data.iloc[-1]['Date']} at {last_close_price} (Last day close)")

                        if compound:
                            capital_per_stock = final_capital / no_of_stock_to_trade
def mark_signals(symbol, start_date, end_date, target_percentage, stop_loss_percentage):
    if not os.path.exists(f'{cvs_raw_data}/{symbol}.csv'):
        #print(not os.path.exists(f'{cvs_raw_data}/{stock}.csv'))
        print(f"{symbol} Not found in local, downloading from online and processing it ...")
        # Fetch data from Yahoo Finance
        data = fetch_yahoo_finance_data(symbol + '.NS', start_date, end_date)
        if data.empty:
            print(f"No data found in yfinance for {symbol}. Skipping...")
            return 0, 0  # Skip this stock and return 0 charges and 0 trades
        data.to_csv(f"{cvs_raw_data}/{stock}.csv")
        data = pd.read_csv(f'{cvs_raw_data}/{symbol}.csv')
        print('---')
    else:
        print(f"{symbol} found in local and processing it ...")
        data = pd.read_csv(f'{cvs_raw_data}/{symbol}.csv')
        print('---')

    # Calculate EMA
    data = calculate_ema(data, 200)
    data = calculate_ema(data, 50)
    data = calculate_ema(data, 100)
    data = calculate_ema(data, 20)
    data = calculate_ema(data, 7)
    data = calculate_supertrend(data, period=7, multiplier=3)

    ###
    for i in range(100, len(data)):  # Start from the 100th day to have enough data for calculations
        #if not trade and data.index[i].date() >= datetime.strptime(from_date, '%Y-%m-%d').date():

        is_previous_green = (data.iloc[i - 1]['Close'] > data.iloc[i - 1]['Open'])

        is_current_red = (data.iloc[i]['Close'] < data.iloc[i]['Open'])
        is_50EMA_above_200EMA = data.iloc[i]['EMA_50'] > data.iloc[i]['EMA_200']
        is_prev_close_below_200EMA = data.iloc[i - 1]['Close'] < data.iloc[i - 1]['EMA_200']
        yday_50EMA_above_200EMA = data.iloc[i - 1]['EMA_50'] > data.iloc[i - 1]['EMA_200']
        yday_100EMA_above_200EMA = data.iloc[i - 1]['EMA_100'] > data.iloc[i - 1]['EMA_200']
        yday_7EMA_above_200EMA = data.iloc[i - 1]['EMA_7'] > data.iloc[i - 1]['EMA_200']
        yday_50EMA_above_100EMA = data.iloc[i - 1]['EMA_50'] > data.iloc[i - 1]['EMA_100']
        yday_20EMA_above_50EMA = data.iloc[i - 1]['EMA_20'] > data.iloc[i - 1]['EMA_50']
        two_prev_yday_20EMA_below_50EMA = data.iloc[i - 2]['EMA_20'] < data.iloc[i - 2]['EMA_50'] or data.iloc[i - 3]['EMA_20'] < data.iloc[i - 3]['EMA_50']
        yday_7EMA_below_20EMA = data.iloc[i - 1]['EMA_7'] < data.iloc[i - 1]['EMA_20']
        yday_20EMA_below_200EMA = data.iloc[i - 1]['EMA_20'] < data.iloc[i - 1]['EMA_200']
        yday_20EMA_above_200EMA = data.iloc[i - 1]['EMA_20'] > data.iloc[i - 1]['EMA_200']
        yday_20EMA_above_100EMA = data.iloc[i - 1]['EMA_20'] > data.iloc[i - 1]['EMA_100']
        yday_20EMA_below_100EMA = data.iloc[i - 1]['EMA_20'] < data.iloc[i - 1]['EMA_100']
        day_7EMA_below_100EMA = data.iloc[i - 1]['EMA_7'] < data.iloc[i - 1]['EMA_100']
        ###
        yday_close_above_7EMA = data.iloc[i - 1]['Close'] > data.iloc[i - 1]['EMA_7']
        two_day_b4_close_above_7EMA = data.iloc[i - 2]['Close'] > data.iloc[i - 1]['EMA_7']
        two_white_crow = data.iloc[i - 2]['Close'] > data.iloc[i - 2]['EMA_7'] and  data.iloc[i - 2]['Close'] > data.iloc[i - 2]['EMA_7']

        yday_open_below_7EMA = data.iloc[i - 1]['Open'] < data.iloc[i - 1]['EMA_7']
        yday_close_above_20EMA = data.iloc[i - 1]['Close'] > data.iloc[i - 1]['EMA_20']
        yday_close_below_20EMA = data.iloc[i - 1]['Close'] < data.iloc[i - 1]['EMA_20']
        yday_open_below_20EMA = data.iloc[i - 1]['Open'] < data.iloc[i - 1]['EMA_20']
        yday_close_below_50EMA = data.iloc[i - 1]['Close'] < data.iloc[i - 1]['EMA_50']
        ####
        yday_close_above_20EMA = data.iloc[i - 1]['Close'] > data.iloc[i - 1]['EMA_20']
        #########
        two_day_b4_close_below_7EMA = data.iloc[i - 2]['Close'] < data.iloc[i - 2]['EMA_20']
        three_day_b4_close_below_7EMA = data.iloc[i - 3]['Close'] < data.iloc[i - 3]['EMA_20']
        four_day_b4_close_below_7EMA = data.iloc[i - 4]['Close'] < data.iloc[i - 4]['EMA_20']
        five_day_b4_close_below_7EMA = data.iloc[i - 5]['Close'] < data.iloc[i - 5]['EMA_20']
        six_day_b4_close_below_7EMA = data.iloc[i - 6]['Close'] < data.iloc[i - 6]['EMA_20']
        seven_day_b4_close_below_7EMA = data.iloc[i - 7]['Close'] < data.iloc[i - 7]['EMA_20']
        ##########
        #########
        two_day_b4_ohlc_avg_below_7EMA = (data.iloc[i - 2]['Open'] + data.iloc[i - 2]['High'] + data.iloc[i - 2]['Low'] + data.iloc[i - 2]['Close']) / 4 < data.iloc[i - 2]['EMA_7']
        three_day_b4_ohlc_avg_below_7EMA = (data.iloc[i - 3]['Open'] + data.iloc[i - 3]['High'] + data.iloc[i - 3]['Low'] + data.iloc[i - 3]['Close']) / 4 < data.iloc[i - 3]['EMA_7']
        four_day_b4_ohlc_avg_below_7EMA = (data.iloc[i - 4]['Open'] + data.iloc[i - 4]['High'] + data.iloc[i - 4]['Low'] + data.iloc[i - 4]['Close']) / 4 < data.iloc[i - 4]['EMA_7']
        five_day_b4_ohlc_avg_below_7EMA = (data.iloc[i - 5]['Open'] + data.iloc[i - 5]['High'] + data.iloc[i - 5]['Low'] + data.iloc[i - 5]['Close']) / 4 < data.iloc[i - 5]['EMA_7']
        six_day_b4_ohlc_avg_below_7EMA = (data.iloc[i - 6]['Open'] + data.iloc[i - 6]['High'] + data.iloc[i - 6]['Low'] + data.iloc[i - 6]['Close']) / 4 < data.iloc[i - 6]['EMA_7']
        seven_day_b4_ohlc_avg_below_7EMA = (data.iloc[i - 7]['Open'] + data.iloc[i - 7]['High'] + data.iloc[i - 7]['Low'] + data.iloc[i - 7]['Close']) / 4 < data.iloc[i - 7]['EMA_7']
        ##########
        is_open_below_yday_close = data.iloc[i]['Open'] < data.iloc[i - 1]['Close']
        is_tday_high_break_yday_high = data.iloc[i]['High'] > data.iloc[i - 1]['High']
        #####
        six_candle_below_EMA_7 = two_day_b4_close_below_7EMA and three_day_b4_close_below_7EMA and four_day_b4_close_below_7EMA and five_day_b4_close_below_7EMA \
                                 and six_day_b4_close_below_7EMA and seven_day_b4_close_below_7EMA
        six_candle_avg_below_EMA_7 = two_day_b4_ohlc_avg_below_7EMA and three_day_b4_ohlc_avg_below_7EMA and four_day_b4_ohlc_avg_below_7EMA and five_day_b4_ohlc_avg_below_7EMA \
        and six_day_b4_ohlc_avg_below_7EMA and seven_day_b4_ohlc_avg_below_7EMA

        four_candle_avg_below_EMA_7 = two_day_b4_ohlc_avg_below_7EMA and three_day_b4_ohlc_avg_below_7EMA and four_day_b4_ohlc_avg_below_7EMA and five_day_b4_ohlc_avg_below_7EMA
        two_candle_avg_below_EMA_7 = two_day_b4_ohlc_avg_below_7EMA and three_day_b4_ohlc_avg_below_7EMA and four_day_b4_ohlc_avg_below_7EMA

        ######
        yday_20EMA_gt_50EMA_gt_100EMA_gt_200EMA =  data.iloc[i - 1]['EMA_20'] > data.iloc[i - 1]['EMA_7'] and data.iloc[i - 1]['EMA_50'] > data.iloc[i - 1]['EMA_100'] \
                and data.iloc[i - 1]['EMA_100'] > data.iloc[i - 1]['EMA_200'] and data.iloc[i - 2]['EMA_7'] < data.iloc[i - 2]['EMA_50'] and data.iloc[i - 3]['EMA_7'] < data.iloc[i - 3]['EMA_50']
        ######
        lowest_low = min(data.iloc[i - 1]['Low'],data.iloc[i - 2]['Low'],data.iloc[i - 3]['Low'],data.iloc[i - 4]['Low'],data.iloc[i - 5]['Low'],
                         data.iloc[i - 6]['Low'],data.iloc[i - 7]['Low'],data.iloc[i - 8]['Low'])
        highest_close = max(data.iloc[i - 1]['Close'],data.iloc[i - 2]['Close'],data.iloc[i - 3]['Close'],data.iloc[i - 4]['Close'],data.iloc[i - 5]['Close'],
                            data.iloc[i - 6]['Close'],data.iloc[i - 7]['Close'])
        buy_today_cond = is_tday_high_break_yday_high and is_open_below_yday_close
        EMA_50_100_200_UPTREND = data.iloc[i - 1]['EMA_200'] < data.iloc[i - 1]['EMA_100'] < data.iloc[i - 1]['EMA_50']
        if buy_today_cond and condition:
            bought_price = round_to_nearest_0_05(data.iloc[i - 1]['High'])
            stop_loss = round_to_nearest_0_05(bought_price * (1 - stop_loss_percentage / 100))
            target = round_to_nearest_0_05(bought_price * (1 + target_percentage / 100))
            #stop_loss = lowest_low - (bought_price - lowest_low)
            #target = bought_price + (bought_price - lowest_low) * 2
            data.loc[data.index[i], 'Buy_Signal'] = round_to_nearest_0_05(data.iloc[i - 1]['High'])
            data.loc[data.index[i], 'Target'] = target
            data.loc[data.index[i], 'StopLoss'] = stop_loss
            #print(f'Bought Price :{bought_price},Lowest Low :{lowest_low}, Stop Loss :{stop_loss}, Target :{target}')
    #data = data.loc[from_date:]
    #print(data['Date'])
    data = data.loc[data['Date'] >= from_date]
    data = convert_all_col_digit(data)
    # Columns to keep
    columns_to_keep = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
                       'EMA_200', 'EMA_50', 'EMA_100', 'EMA_20', 'EMA_7', 'Buy_Signal', 'Target', 'StopLoss']
    # Keep only the specified columns
    data = data.filter(columns_to_keep)
    data.to_csv(f"{cvs_data_dir}/{stock}.csv", index=False)
# Read the configuration file
config = configparser.ConfigParser()
config.read('config.cfg')

# Extract variables from the configuration file
symbols_file = config['trade_symbol']['symbols_file']
create_chart = config.getboolean('trade_symbol', 'create_chart')

from_date = config['time_management']['from_date']
to_date = config['time_management']['to_date']

capital = float(config['risk_management']['capital'])
no_of_stock_to_trade = int(config['risk_management']['no_of_stock_to_trade'])
compound = config.getboolean('risk_management', 'compound')
target_percentage = float(config['risk_management']['target_percentage'])
stop_loss_percentage = float(config['risk_management']['stop_loss_percentage'])
charges_percentage = float(config['risk_management']['charges_percentage'])
cleanup_logs = config.getboolean('house_keeping', 'cleanup_logs')
condition = config['trade_logic']['condition']
##############
symbols_type = symbols_file.split('.')[0]
Reports_Dir = f'{symbols_type}_Reports_{from_date}_to_{to_date}'
Charts_Dir = f'{symbols_type}_Charts_{from_date}_to_{to_date}'
Summary_Dir = f'{symbols_type}_Summary_{from_date}_to_{to_date}'
Master_Dir = f'{symbols_type}_Master_{from_date}_to_{to_date}'
cvs_data_dir = f'{symbols_type}_Cvs_Data_{from_date}_to_{to_date}'
cvs_raw_data = f'{symbols_type}_Raw_Data_{from_date}_to_{to_date}'
##############

create_directory(symbols_type, from_date, to_date)

###
with open('./symbols/' + symbols_file, 'r') as file:
    stocks = [line.split('#')[0].strip() for line in file if not line.lstrip().startswith('#')]
total_stocks = len(stocks)
print(f"Total number of stocks: {total_stocks}")
for count, stock in enumerate(stocks, 1):
    print(f"Processing stock {count}/{total_stocks}: {stock}")
    mark_signals(stock, from_date, to_date, target_percentage, stop_loss_percentage)
######################### MARK SIGNAL END ############################
print("Processed All Data, now will start Trade...")
time.sleep(10)
######################### ACTUAL TRADE BEGINS #############################
#############

# Read the date reference file
get_stock_for_date_refrence(cvs_raw_data, from_date, to_date)
file_list = [f'{cvs_raw_data}/stock_date_ref.csv']
copy_specific_files(file_list, cvs_data_dir)
date_ref_df = pd.read_csv(f'{cvs_data_dir}/stock_date_ref.csv')
dates = date_ref_df['Date'].tolist()

# Initialize variables
active_positions = []
capital_per_stock = capital / no_of_stock_to_trade
portfolio = {}
trade_report = {}
final_capital = capital

# Process each date in the date reference file
for date in dates:
    process_date(date)

# Save each stock's trade report to a separate CSV file
for symbol, trades in trade_report.items():
    report_df = pd.DataFrame(trades)
    if not report_df.empty:
        report_df.to_csv(f'{Reports_Dir}/{symbol}_trade_report.csv', index=False)

# Print the final capital
print(f"Final capital: {final_capital}")

# Initialize summary list
all_summaries = []

# Iterate over each stock's trade report and calculate summary
for report_file in glob.glob(f"{Reports_Dir}/*.csv"):
    symbol = os.path.basename(report_file).replace('_trade_report.csv', '')
    df = pd.read_csv(report_file)
    summary = calculate_summary_per_stock(symbol, df.to_dict('records'), Summary_Dir)
    all_summaries.append(summary)

# Print confirmation
print("Stock summaries created successfully.")

# Call function to create the Master_no_Compound_sce_5.csv file
create_master_file(Summary_Dir,  from_date, to_date, capital)
# Clean up logs if necessary
if cleanup_logs:
    for log_file in glob.glob('*.log'):
        os.remove(log_file)

draw_down_chart()
