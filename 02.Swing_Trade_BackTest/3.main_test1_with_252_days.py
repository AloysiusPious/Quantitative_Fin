import configparser
from matplotlib.offsetbox import AnchoredText
from swing_util import *
import concurrent.futures
import time
import talib
from patterns import morning_star
def get_india_vix_data():

    # ---------------- Load or Download ----------------
    if not os.path.exists(vix_file_path):
        print((enctoken, "INDIA VIX", from_date, to_date, time_frame))
        # ---- Fetch Nifty 50 data ----i have set this function to return "from_date -1"
        vix_data = fetch_kite_data(enctoken, "INDIA VIX", from_date, to_date, interval=time_frame)
        # 🔥 HARD GUARD (THIS FIXES YOUR ERROR)
        if vix_data is None or vix_data.empty:
            raise RuntimeError("❌ INDIA VIX data fetch failed")
        # Ensure Date column is datetime
        vix_data['Date'] = pd.to_datetime(vix_data['Date'])

        # Set as index
        vix_data.set_index('Date', inplace=True)

        # Filter between dates
        start = pd.to_datetime(from_date)
        end = pd.to_datetime(to_date)

        vix_data = vix_data.loc[start:end]
        vix_data.to_csv(vix_file_path, index=True)
def get_vix_by_date(date_str):

    """
    vix_file_path: path to your india vix CSV
    date_str: string date like "2024-05-15"
    """
    # Read CSV with date parsing
    df = pd.read_csv(vix_file_path, parse_dates=['Date'])

    # Convert input date to datetime
    target_date = pd.to_datetime(date_str)

    # Filter row
    row = df[df['Date'] == target_date]

    if row.empty:
        return None  # or raise Exception("Date not found")

    # If your vix column is named "VIX"
    return float(row.iloc[0]['Close'])
def draw_down_chart(total_trades, winning_trades_per, net_profit ):
    all_trades = []
    # ---- Load all CSV files and merge ----
    if os.path.isdir(Reports_Dir) and os.listdir(Reports_Dir):
        for filename in os.listdir(Reports_Dir):
            if filename.endswith(".csv"):
                filepath = os.path.join(Reports_Dir, filename)
                df = pd.read_csv(filepath, parse_dates=['Buy Date', 'Exited Date'])
                all_trades.append(df)

        all_trades = pd.concat(all_trades, ignore_index=True)
        all_trades = all_trades.sort_values(by='Buy Date')

        # ---- Capital Computation ----
        all_trades['Cumulative Profit'] = all_trades['Profit Amount'].cumsum()
        all_trades['Capital'] = capital + all_trades['Cumulative Profit']

        # Ensure dates are datetime
        all_trades['Buy Date'] = pd.to_datetime(all_trades['Buy Date'])

        file_path = f'{cvs_raw_data}/nifty_50.csv'

        # ---------------- Load or Download ----------------
        if not os.path.exists(file_path):
            # ---- Fetch Nifty 50 data ----i have set this function to return "from_date -1"
            nifty50_data = fetch_kite_data(enctoken, "NIFTY 50", from_date, to_date, interval=time_frame)
            # Ensure Date column is datetime
            nifty50_data['Date'] = pd.to_datetime(nifty50_data['Date'])

            # Set as index
            nifty50_data.set_index('Date', inplace=True)

            # Filter between dates
            start = pd.to_datetime(from_date)
            end = pd.to_datetime(to_date)

            nifty50_data = nifty50_data.loc[start:end]
            nifty50_data.to_csv(file_path, index=True)
        nifty50_data = pd.read_csv(file_path)
        # Fix Nifty date column and set index
        nifty50_data['Date'] = pd.to_datetime(nifty50_data['Date'])
        nifty50_data.set_index('Date', inplace=True)

        # ---- Calculate returns ----
        final_capital = all_trades['Capital'].iloc[-1]
        percentage_increase = ((final_capital - capital) / capital) * 100
        # ---- Capital Computation ----
        all_trades['Cumulative Profit'] = all_trades['Profit Amount'].cumsum()
        all_trades['Capital'] = capital + all_trades['Cumulative Profit']

        # ---- Drawdown Calculation ----
        all_trades['Peak'] = all_trades['Capital'].cummax()
        all_trades['Drawdown'] = all_trades['Capital'] - all_trades['Peak']
        all_trades['Drawdown_Pct'] = (all_trades['Drawdown'] / all_trades['Peak']) * 100

        # ---- Top 5 Drawdowns ----
        dd_points = all_trades.nsmallest(5, 'Drawdown')

        # ---- Create Plot ----
        fig, ax1 = plt.subplots(figsize=(14, 7))

        # Capital growth line
        ax1.plot(
            all_trades['Buy Date'],
            all_trades['Capital'],
            linestyle='-',
            color='b'
            #label='Capital Over Time'
        )

        # Start annotation
        ax1.annotate(
            f'Start: ₹{capital}',
            xy=(all_trades['Buy Date'].iloc[0], capital),
            xytext=(all_trades['Buy Date'].iloc[0], capital * 1.01),
            arrowprops=dict(facecolor='green', shrink=0.05)
        )

        # End annotation
        ax1.annotate(
            f'End: ₹{final_capital:.2f} ({percentage_increase:.2f}%)',
            xy=(all_trades['Buy Date'].iloc[-1], final_capital),
            xytext=(all_trades['Buy Date'].iloc[-1], final_capital * 1.01),
            arrowprops=dict(facecolor='red', shrink=0.05)
        )

        # Axis labels
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Capital (₹)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')

        # --- FIX: only show legend if there are labels ---
        handles, labels = ax1.get_legend_handles_labels()
        if labels:  # avoid "No artists with labels" warning
            ax1.legend(loc='upper left')

        # ---- Nifty 50 plot on second axis ----
        ax2 = ax1.twinx()
        ax2.plot(
            nifty50_data.index,
            nifty50_data['Close'],
            linestyle='--',
            color='orange',
            label='Nifty 50'
        )
        ax2.set_ylabel('Nifty 50 Index', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        ax2.legend(loc='upper center')

        # ---- Chart styling ----
        plt.title('Capital Growth Over Time and Nifty 50 Index')
        fig.tight_layout()
        plt.grid(True)
        # ---- Mark Top 5 Drawdowns (Improved, No Overlap) ----
        offsets = [1.10, 0.90, 1.15, 0.85, 1.20]  # Different Y offsets for labels

        for i, (_, row) in enumerate(dd_points.iterrows()):
            dd_date = row['Buy Date']
            dd_cap = row['Capital']
            dd_pct = abs(row['Drawdown_Pct'])

            # Scatter the DD point
            ax1.scatter(dd_date, dd_cap, color='red', s=60, zorder=5)

            # Compute offset position
            y_offset_position = dd_cap * offsets[i]  # Multiply capital to move label away

            # Draw vertical dashed helper line
            ax1.plot(
                [dd_date, dd_date],
                [dd_cap, y_offset_position],
                color='red',
                linestyle='dashed',
                linewidth=1,
                alpha=0.7
            )

            # Draw label
            ax1.annotate(
                f"-{dd_pct:.2f}%",
                xy=(dd_date, y_offset_position),
                xytext=(dd_date, y_offset_position),
                color="red",
                fontsize=10,
                ha="center",
                bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.3')
            )
        # Step 1 → profit before tax
        profit_before_tax = final_capital - capital
        # Step 2 → profit after 20% tax
        net_profit_after_tax = profit_before_tax * 0.80
        # Step 3 → final capital after tax
        final_after_tax = capital + net_profit_after_tax
        # Step 4 → CAGR (annualized return)
        cagr = (final_after_tax / capital) ** (1 / num_years) - 1
        # Step 5 → prepare text box
        textstr = '\n'.join((
            f'max_position = {no_of_stock_to_trade}',
            f'total_trades = {total_trades}',
            f'winning_trade_% = {winning_trades_per}',
            # corrected net % after tax from your existing net_profit variable
            f'net_profit_%_aft_20%_tax_and_charges = {round(net_profit * 0.80, 2)}',
            f'profit_before_tax = {round(profit_before_tax, 2)}',
            f'net_profit_aft_20%_tax_and_charges = {round(net_profit_after_tax, 2)}',
            # round CAGR to percentage and readable format
            f'CAGR = {round(cagr * 100, 2)} %',
            #f'logic = {cond_buy_text}'
            f'logic = {"none"}'

        ))

        # anchored_text = AnchoredText(
        #     textstr,
        #     loc='lower right',
        #     frameon=True,
        #     bbox_to_anchor=(1, 0.15),
        #     bbox_transform=ax1.transAxes,
        #     prop=dict(size=8)
        # )
        anchored_text = AnchoredText(
            textstr,
            loc='upper left',
            frameon=True,
            bbox_to_anchor=(0.02, 0.98),  # 👈 near top-left
            bbox_transform=ax1.transAxes,
            prop=dict(size=8)
        )
        anchored_text.patch.set_boxstyle("round,pad=0.5,rounding_size=0.5")
        ax1.add_artist(anchored_text)
        # ---- Save and show ----
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
        'Winning Trade Percentage': round((sum(total_winning_trades) / sum(total_trades)) * 100, 2) if sum(
            total_trades) > 0 else 0,
        'Losing Trade Percentage': round((sum(total_losing_trades) / sum(total_trades)) * 100, 2) if sum(
            total_trades) > 0 else 0,
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
    ###

    net_profit = round((sum(total_profit_after_charges) / capital) * 100, 2)
    draw_down_chart(sum(total_trades), round((sum(total_winning_trades) / sum(total_trades)) * 100, 2) if sum(
            total_trades) > 0 else 0, round((sum(total_profit_after_charges) / capital) * 100, 2) ,)
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


def process_date(date, date_ref_df, trade_report):
    global active_positions, capital_per_stock, final_capital, last_buy_month

    last_date = date_ref_df['Date'].max()
    stock_files = read_stock_files()
    current_month = pd.to_datetime(date).to_period('M')

    for stock_file in stock_files:
        stock_df = pd.read_csv(stock_file)
        stock_data = stock_df[stock_df['Date'] == date]
        symbol = os.path.basename(stock_file).replace('.csv', '')

        if symbol not in trade_report:
            trade_report[symbol] = []

        if not stock_data.empty:
            row = stock_data.iloc[0]

            # ✅ Already bought this stock this month?
            already_bought_this_month = (
                symbol in last_buy_month and last_buy_month[symbol] == current_month
            )

            # ================= BUY LOGIC =================
            if 'Buy_Signal' in row and not pd.isna(row['Buy_Signal']):

                active_for_stock = [p for p in active_positions if p['symbol'] == symbol]

                if (len(active_for_stock) == 0
                        and len(active_positions) < no_of_stock_to_trade
                        and not already_bought_this_month):

                    buy_price = row['Buy_Signal']
                    shares_to_buy = int(capital_per_stock // buy_price)
                    invested_amount = shares_to_buy * buy_price

                    active_positions.append({
                        'symbol': symbol,
                        'buy_date': date,
                        'buy_price': buy_price,
                        'target_price': None,       # ✅ No target
                        'stop_loss_price': None,    # ✅ No stop loss
                        'shares': shares_to_buy
                    })

                    portfolio[symbol] = {
                        'Buy Date': date,
                        'Bought Price': buy_price,
                        'Quantity Bought': shares_to_buy,
                        'Invested Amount': invested_amount,
                        'Stop Loss': None,
                        'Target': None,
                        'Exited Date': None,
                        'Exited Price': None,
                        'Profit Amount': None,
                        'Trade Status': None,
                        'No of holding Days': None,
                        'Profit %': None
                    }

                    # ✅ Stamp this month so we skip further signals this month
                    last_buy_month[symbol] = current_month
                    print(f"Bought {symbol} on {date} at {buy_price}")

        # ================= LAST DAY EXIT =================
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
                            'Stop Loss': None,
                            'Target': None,
                            'Exited Date': last_day_data.iloc[-1]['Date'],
                            'Exited Price': round_to_nearest_0_05(last_close_price),
                            'Profit Amount': round_to_nearest_0_05(profit_amount),
                            'Trade Status': 'LastDayClose',
                            'No of holding Days': calculate_holding_days(
                                position['buy_date'],
                                last_day_data.iloc[-1]['Date']
                            ),
                            'Profit %': round_to_nearest_0_05(profit_percent)
                        })

                        final_capital += profit_amount

                        if compound:
                            capital_per_stock = final_capital / no_of_stock_to_trade

                        print(f"Sold {symbol} on {last_day_data.iloc[-1]['Date']} at {last_close_price} (Last day close)")

                # ✅ Clear all positions for this symbol after last day exit
                active_positions = [p for p in active_positions if p['symbol'] != symbol]
######################### ACTUAL TRADE BEGINS #############################

def get_stock_for_reference_date(enctoken, cvs_data_dir, cvs_raw_data, start_date, end_date):
    file_path = f'{cvs_raw_data}/stock_date_ref.csv'

    # Load or Download
    if not os.path.exists(file_path):
        # Read the date reference file
        nifty50_data = fetch_kite_data(enctoken, "TCS", start_date, end_date, interval=time_frame)
        # Extract only the Date column & ensure date format
        date_ref = pd.DataFrame({
            'Date': pd.to_datetime(nifty50_data['Date']).dt.date
        })
        # Drop empty rows (usually none)
        date_ref.dropna(inplace=True)
        # Save to CSV
        date_ref.to_csv(f"{cvs_raw_data}/stock_date_ref.csv", index=False)
    date_ref_df = pd.read_csv(f'{cvs_raw_data}/stock_date_ref.csv')
    dates = date_ref_df['Date'].tolist()
    # Process each date in the date reference file
    for date in dates:
        process_date(date, date_ref_df, trade_report)


def procee_final_report(Reports_Dir):
    ###### Process Final Report

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
    create_master_file(Summary_Dir, from_date, to_date, capital)
    # Clean up logs if necessary
    if cleanup_logs:
        for log_file in glob.glob('*.log'):
            os.remove(log_file)


# Globals: cvs_raw_data, cvs_data_dir, time_frame='daily', fetch_kite_data

def mark_signals(enctoken, symbol, start_date, end_date):
    file_path = f'{cvs_raw_data}/{symbol}.csv'

    # Load or Download
    if not os.path.exists(file_path):
        print(f"{symbol} Not found locally — downloading ...")
        data = fetch_kite_data(enctoken, symbol, start_date, end_date, interval=time_frame)
        if data is None or data.empty:
            print(f"No data for {symbol}")
            return
        data.to_csv(file_path, index=False)
    else:
        print(f"{symbol} found in local and processing it ...")
        data = pd.read_csv(file_path)
    # Prep
    data = data.sort_values('Date').reset_index(drop=True)
    data['Date'] = pd.to_datetime(data['Date'])
    # ← Monthly Chart  Begin HERE →#########################################
    # Monthly analysis
    monthly = data.resample('ME', on='Date').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'
    }).dropna()
    monthly['Prev_Month_Red'] = (monthly['Close'] < monthly['Open']).shift(1)
    data = data.sort_values('Date').set_index('Date')
    data['Month_Start'] = data.index.to_period('M').start_time
    # --- FIX: handle the map result BEFORE fillna ---
    mapped = data['Month_Start'].map(
        monthly.set_index(monthly.index.to_period('M').start_time)['Prev_Month_Red']
    )
    mapped = mapped.astype('boolean')  # This ensures it's a proper boolean dtype
    data['Prev_Month_Was_Red'] = (
        mapped.infer_objects()
        .fillna(False)
        .astype(bool)
    )
    data['Trading_Day_of_Month'] = data.groupby(data.index.to_period('M')).cumcount() + 1
    data = data.reset_index()
    block_first_5 = data['Prev_Month_Was_Red'] & (data['Trading_Day_of_Month'] <= 5)
    # ← Monthly Chart END HERE →#########################################

    # Indicators
    data['EMA7'] = data['Close'].ewm(span=7, adjust=False).mean()
    data['EMA20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['EMA50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['EMA100'] = data['Close'].ewm(span=100, adjust=False).mean()
    data['EMA200'] = data['Close'].ewm(span=200, adjust=False).mean()
    data['Pct_Below_7EMA'] = (data['Close'] - data['EMA7']) / data['EMA7'] * 100
    data['Pct_Below_20EMA'] = (data['Close'] - data['EMA20']) / data['EMA20'] * 100
    data['Pct_Below_50EMA'] = (data['Close'] - data['EMA50']) / data['EMA50'] * 100
    data['Pct_Above_200EMA'] = (data['Close'] - data['EMA200']) / data['EMA200'] * 100
    data['RSI14'] = talib.RSI(data['Close'], timeperiod=14)

    # Load VIX + map to rows
    vix_df = pd.read_csv(vix_file_path, parse_dates=['Date']).set_index('Date')
    data['VIX_Close'] = data['Date'].map(vix_df['Close'])
    # --- MACD ---
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD_Line'] = ema12 - ema26
    data['MACD_Signal'] = data['MACD_Line'].ewm(span=9, adjust=False).mean()
    data['MACD_Slope'] = data['MACD_Line'].diff()
    cond_macd_turn = (data['MACD_Line'] < 0) & (data['MACD_Line'].diff() > 0)
    # Columns
    data['Buy_Signal'] = np.nan
    data['StopLoss'] = np.nan
    data['Target'] = np.nan
    today_green = (data['Close'] > data['Open'])
    yday_red = (data['Close'].shift(1) < data['Open'].shift(1))
    db4_red = (data['Close'].shift(2) < data['Open'].shift(2))
    global cond_buy_text
    # Buy logic per row
    #data['Morning_Star'] = morning_star(data)
    #cond_buy_text = "((data['Pct_Below_20EMA'] < -2) & (data['VIX_Close'] < 30) & (data['EMA20'] > data['EMA50']))"
    # data['Morning_Star'] = morning_star(data)
    #
    # past_3_red= (data['Close'].shift(1) < data['Open'].shift(1)) & (data['Close'].shift(2) < data['Open'].shift(2)) & (data['Close'].shift(3) < data['Open'].shift(3))
    # cond_buy_text = "((data['VIX_Close'] < 30)  & (data['EMA20'] > data['EMA50']) & past_3_red  & today_green)"
    #cond_buy_text = "((data['Pct_Below_50EMA'] < -2) & (data['VIX_Close'] < 30) & (~block_first_5))"
    #cond_buy_text = "((data['Pct_Below_50EMA'] < -2) & (data['VIX_Close'] < 30) & (~block_first_5) & (data['EMA50'] > data['EMA100']))"
    #cond_buy_text = "(data['EMA20'] > data['EMA50']) & (data['EMA20'].shift(1) <= data['EMA50'].shift(1)) & (data['VIX_Close'] < 30)"
    #cond_buy_text = "((data['EMA20'] > data['EMA200']) & (data['Pct_Below_20EMA'] < -2) & (data['VIX_Close'] < 30))"
    #cond_buy_text = "(data['Pct_Below_20EMA'] < -2) & (data['VIX_Close'] < 30) #& (data['RSI14'] < 40)"
    #cond_buy_text = "((data['Pct_Below_20EMA'] < -2) & (data['VIX_Close'] < 30)) & (~block_first_5)"  # ← No trades in  first 5 days after red month
    #three_candle_dwn = (data['Low'].shift(3) > data['Low'].shift(2)) & (data['Low'].shift(2) > data['Low'].shift(1)) & (data['Low'].shift(1) > data['Low'])
    #cond_buy_text = "(three_candle_dwn & (data['VIX_Close'] < 30) & (data['EMA20'] > data['EMA50']))"
    # Detect swing low: Current low is minimum in a 5-bar window (2 left, current, 2 right)
    # ####################################################
    # window = 5
    # data['Swing_Low_Price'] = data['Low'].rolling(window, center=True).min()
    # data['Is_Swing_Low'] = data['Low'] == data['Swing_Low_Price']
    # data['High_Not_Over_2pct_From_SwingLow'] = (
    #         (data['High'] - data['Swing_Low_Price']) / data['Swing_Low_Price'] <= 0.02)
    # cond_buy_text = "(data['High_Not_Over_2pct_From_SwingLow']) & (data['VIX_Close'] < 30)  & (data['EMA20'] > data['EMA200'])"
    # ####################################################
    ####################################################
    window = 21

    data['Swing_Low_Price'] = data['Low'].rolling(window).min()
    data['Swing_High_Price'] = data['High'].rolling(window).max()
    data['Valid_Swing'] = (
            (data['Low'] == data['Swing_Low_Price']) &
            ((data['Swing_High_Price'] - data['Swing_Low_Price']) / data['Swing_Low_Price'] > 0.10)
    )
    data['Today_Green'] = data['Close'] > data['Open']
    #cond_buy_text = "(data['Today_Green']) & (data['Valid_Swing']) & (data['VIX_Close'] < 30)  & (data['EMA20'] > data['EMA200'])"
    ######## 252 Days Low
    cond_buy_text = "((data['Low'].rolling(5).min() <= data['Low'].rolling(252).min() * 1.03) \
    & (data['Close'] > data['Close'].shift(1)) \
    & (data['RSI14'] > 35) \
    & (data['RSI14'] < 55) \
    & ((data['MACD_Line'] - data['MACD_Signal']) < 0) \
    & (data['Volume'] > data['Volume'].rolling(20).mean() * 1.2))"
    ##########################
    ####################################################
    # data['Pullback'] = (
    #         (data['Close'] < data['EMA20']) &
    #         (data['Close'] > data['EMA50'])
    # )
    # data['Momentum_Turn'] = data['Close'] > data['Close'].shift(1)
    # cond_buy_text = "(data['Pullback'] & data['Momentum_Turn'] &(data['EMA20'] > data['EMA50']) &(data['VIX_Close'] < 30))"
    ####################################################
    # data['Fractal_Low'] = (
    #         (data['Low'].shift(2) < data['Low'].shift(3)) &
    #         (data['Low'].shift(2) < data['Low'].shift(4)) &
    #         (data['Low'].shift(2) < data['Low'].shift(1)) &
    #         (data['Low'].shift(2) < data['Low'])
    # )
    #
    # # Signal comes 2 candles later
    # data['Fractal_Low'] = data['Fractal_Low'].shift(1)
    # cond_buy_text = "(data['Fractal_Low'] &(data['EMA20'] > data['EMA50']) &(data['VIX_Close'] < 30))"
    ######################################################
    # window = 5
    # # 1️⃣ Causal swing low (no future candles)
    # data['Swing_Low_Causal'] = (data['Low'] == data['Low'].rolling(window).min())
    # data['Swing_Low_Causal_1'] = (data['Low'].shift(1) == data['Low'].shift(1).rolling(window).min())
    # swing_low = data['Swing_Low_Causal'] | data['Swing_Low_Causal_1']
    # After calculating EMAs and VIX
    # Uptrend = (data['EMA50'] > data['EMA200']) # Uptrend check
    # # Buy condition text (early detection)
    # data['yday_Pct_Below_20EMA'] = (data['Low'].shift(1) - data['EMA20'].shift(1)) / data['EMA20'].shift(1) * 100
    # data['tday_Pct_Below_20EMA'] = (data['Low'] - data['EMA20']) / data['EMA20'] * 100
    # yday_or_tday_pct_Below_20EMA = (data['yday_Pct_Below_20EMA'] < -2) | (data['tday_Pct_Below_20EMA'] < -2)
    # cond_buy_text = "((data['Swing_Low_Causal']) & (yday_or_tday_pct_Below_20EMA) & (Uptrend) & (data['VIX_Close'] < 30) & today_green & yday_red & db4_red)"
    ##############################################)#######
    ######################################################
    cond_buy = eval(cond_buy_text)
    ################
    #data.loc[cond_buy, 'Buy_Signal'] = data.loc[cond_buy, 'Close']
    #data.loc[cond_buy, 'StopLoss'] = data.loc[cond_buy, 'Buy_Signal'] * 0.90
    #data.loc[cond_buy, 'Target'] = data.loc[cond_buy, 'Buy_Signal'] * 1.08 #1.08
    # Open target/stop loss
    data.loc[cond_buy, 'Buy_Signal'] = data.loc[cond_buy, 'Close']
    data.loc[cond_buy, 'StopLoss'] = np.nan
    data.loc[cond_buy, 'Target'] = np.nan
    # Cleanup
    data = data.dropna(subset=['EMA20', 'RSI14']).reset_index(drop=True)
    data = data[data['Date'] >= pd.to_datetime(start_date)]

    # Save output
    data.to_csv(f"{cvs_data_dir}/{symbol}.csv", index=False)

    latest_pct = data['Pct_Below_20EMA'].iloc[-1] if not data.empty else np.nan
    num_signals = data['Buy_Signal'].notna().sum()
    print(f"✅ Processed {symbol} — SHOP w/ RSI: Latest % Below EMA20: {latest_pct:.2f}%. Signals: {num_signals}")
    return int(num_signals) if num_signals else 0



##################################################################################################
def start_processing_symbols(enctoken, symbols_file, from_date, to_date):

    # ---------- Load symbols ----------
    with open('./symbols/' + symbols_file, 'r') as file:
        stocks = [
            line.split('#')[0].strip()
            for line in file
            if line.strip() and not line.lstrip().startswith('#')
        ]

    total_stocks = len(stocks)
    print(f"Total number of stocks: {total_stocks}")

    # ---------- Worker ----------
    def process_stock(stock):
        try:
            num_signals = mark_signals(enctoken, stock, from_date, to_date)
            return stock, int(num_signals or 0), None
        except Exception as e:
            return stock, 0, e

    # ---------- Parallel execution ----------
    start_time = time.time()
    total_signals = 0
    max_threads = min(10, total_stocks)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        results = list(executor.map(process_stock, stocks))

    # ---------- Aggregate results ----------
    for stock, num_signals, err in results:
        if err:
            print(f"❌ {stock} failed: {err}")
        else:
            print(f"✅ {stock} done | Signals: {num_signals}")
            total_signals += num_signals

    # ---------- Summary ----------
    print(f"\n✅ Completed {total_stocks} stocks in {time.time() - start_time:.2f}s")
    print(f"📊 Total Signals Captured : {total_signals}")
    print("Now will start Trade...")
    time.sleep(0.25)


#################################
# Read the configuration file
config = configparser.ConfigParser()
config.read('config.cfg')

# Extract variables from the configuration file
symbols_file = config['trade_symbol']['symbols_file']
create_chart = config.getboolean('trade_symbol', 'create_chart')
enctoken = config['trade_symbol']['enc_token']
time_frame = config['trade_symbol']['time_frame']
from_date = config['time_management']['from_date']
to_date = config['time_management']['to_date']
# Convert to date objects
d1 = datetime.strptime(from_date, "%Y-%m-%d")
d2 = datetime.strptime(to_date, "%Y-%m-%d")
num_years = round((d2 - d1).days / 365.25)
capital = float(config['risk_management']['capital'])
no_of_stock_to_trade = int(config['risk_management']['no_of_stock_to_trade'])
compound = config.getboolean('risk_management', 'compound')
target_percentage = float(config['risk_management']['target_percentage'])
stop_loss_percentage = float(config['risk_management']['stop_loss_percentage'])
charges_percentage = float(config['risk_management']['charges_percentage'])
risk_per_trade = float(config['risk_management']['risk_per_trade'])
cleanup_logs = config.getboolean('house_keeping', 'cleanup_logs')
# condition = config['trade_logic']['condition']
##############
symbols_type = symbols_file.split('.')[0]
Reports_Dir = f'{symbols_type}_Reports_{from_date}_to_{to_date}'
Charts_Dir = f'{symbols_type}_Charts_{from_date}_to_{to_date}'
Summary_Dir = f'{symbols_type}_Summary_{from_date}_to_{to_date}'
Master_Dir = f'{symbols_type}_Master_{from_date}_to_{to_date}'
cvs_data_dir = f'{symbols_type}_Cvs_Data_{from_date}_to_{to_date}'
cvs_raw_data = f'{symbols_type}_Raw_Data_{from_date}_to_{to_date}'
vix_file_path = f'{cvs_raw_data}/vix.csv'
# Initialize variables
active_positions = []
capital_per_stock = capital / no_of_stock_to_trade
portfolio = {}
trade_report = {}
final_capital = capital
##############
remove_directory()
create_directory(symbols_type, from_date, to_date)
get_india_vix_data()
start_processing_symbols(enctoken, symbols_file, from_date, to_date)
get_stock_for_reference_date(enctoken, cvs_data_dir, cvs_raw_data, from_date, to_date)
procee_final_report(Reports_Dir)