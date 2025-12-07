import configparser
import os
import glob
import time
import concurrent.futures
import pandas as pd
import numpy as np
import talib
from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt
from swing_util import *
from stratesgies import *

# =============================================================================
#                       FII-GRADE SWING LOW + MULTIPLE TARGET STRATEGY
# =============================================================================

def mark_signals(enctoken, symbol, start_date, end_date):
    import os
    import numpy as np
    import pandas as pd
    import talib

    file_path = f'{cvs_raw_data}/{symbol}.csv'

    if not os.path.exists(file_path):
        print(f"{symbol} Not found locally — downloading ...")
        data = fetch_kite_data(enctoken, symbol, start_date, end_date, interval='day')
        if data is None or data.empty:
            print(f"No data for {symbol}")
            return
        data.to_csv(file_path, index=False)
    else:
        print(f"{symbol} found in local and processing it ...")
        data = pd.read_csv(file_path)

    data = data.sort_values('Date').reset_index(drop=True)
    data['Date'] = pd.to_datetime(data['Date'])

    # Indicators
    data['EMA20']  = data['Close'].ewm(span=20,  adjust=False).mean()
    data['EMA50']  = data['Close'].ewm(span=50,  adjust=False).mean()
    data['EMA100'] = data['Close'].ewm(span=100, adjust=False).mean()
    data['EMA200'] = data['Close'].ewm(span=200, adjust=False).mean()

    # EMA Stack Bullish in last 60 days
    stack_bull = (data['EMA20'] > data['EMA50']) & (data['EMA50'] > data['EMA100']) & (data['EMA100'] > data['EMA200'])
    recent_stack = stack_bull.rolling(60, min_periods=1).max().astype(bool)

    # Swing Low Detection
    data['SwingLow'] = data['Low'].rolling(40, min_periods=20).min()
    is_swing_low = (data['Low'] == data['SwingLow'])

    # Pullback to EMA50 zone
    pullback_ok = data['Low'] <= data['EMA50'] * 1.04

    # Reversal candle + volume + close above prev high
    strong_candle = (data['Close'] > data['Open']) & ((data['Close'] - data['Open']) > 0.7 * (data['High'] - data['Low']))
    breakout_high = data['Close'] > data['High'].shift(1)
    volume_ok = data['Volume'] > data['Volume'].rolling(20).mean()

    # FINAL FII ENTRY AT SWING LOW
    cond_buy = recent_stack & pullback_ok & is_swing_low & strong_candle #& breakout_high & volume_ok

    # Stop Loss & Multiple Targets
    data.loc[cond_buy, 'StopLoss'] = (data['SwingLow'] * 0.994).round(2)
    data.loc[cond_buy, 'Buy_Signal'] = data['Close'].round(2)

    risk = data['Buy_Signal'] - data['StopLoss']
    data.loc[cond_buy, 'Target_1R'] = (data['Buy_Signal'] + risk).round(2)
    data.loc[cond_buy, 'Target_3R'] = (data['Buy_Signal'] + 3 * risk).round(2)
    data.loc[cond_buy, 'Target_5R'] = (data['Buy_Signal'] + 5 * risk).round(2)
    data.loc[cond_buy, 'Target_7R'] = (data['Buy_Signal'] + 7 * risk).round(2)
    data.loc[cond_buy, 'Risk_%'] = ((risk / data['Buy_Signal']) * 100).round(2)

    data.loc[cond_buy, 'Signal_Note'] = "FII_SWING_LOW_MULTITARGET"

    result = data[data['Date'] >= start_date].copy()
    result.to_csv(f"{cvs_data_dir}/{symbol}.csv", index=False)

    trades = cond_buy.sum()
    if trades > 0:
        print(f"Processed {symbol} → {trades} FII Swing Low Entries | Avg Risk: {result.loc[cond_buy, 'Risk_%'].mean():.2f}%")
    else:
        print(f"Processed {symbol} → Waiting for FII entry...")


# =============================================================================
#                       MULTIPLE TARGET + TRAILING EXIT ENGINE
# =============================================================================

def log_trade(pos, exit_date, exit_price, status, shares_sold, profit, trade_report):
    invested = pos['buy_price'] * shares_sold
    profit_pct = (profit / invested * 100) if invested > 0 else 0

    trade_entry = {
        'Buy Date': pos['buy_date'],
        'Bought Price': round(pos['buy_price'], 2),
        'Quantity': shares_sold,
        'Invested Amount': round(invested, 2),
        'Stop Loss': round(pos['initial_sl'], 2),
        'Exited Date': exit_date,
        'Exited Price': round(exit_price, 2),
        'Profit Amount': round(profit, 2),
        'Trade Status': status,
        'Profit %': round(profit_pct, 2)
    }
    trade_report[pos['symbol']].append(trade_entry)
    print(f"   → {status}: {pos['symbol']} | {shares_sold} shares @ {exit_price} | ₹{profit:,.0f} ({profit_pct:+.2f}%)")


def process_date(date, date_ref_df, trade_report):
    global active_positions, capital_per_stock, final_capital
    last_date = date_ref_df['Date'].max()
    stock_files = read_stock_files()

    for stock_file in stock_files:
        stock_df = pd.read_csv(stock_file)
        day_data = stock_df[stock_df['Date'] == date]
        if day_data.empty: continue
        row = day_data.iloc[0]
        symbol = os.path.basename(stock_file).replace('.csv', '')

        if symbol not in trade_report:
            trade_report[symbol] = []

        # === NEW BUY ===
        if 'Buy_Signal' in row and pd.notna(row['Buy_Signal']):
            if any(p['symbol'] == symbol for p in active_positions): continue
            if len(active_positions) >= no_of_stock_to_trade: continue

            buy_price = row['Buy_Signal']
            sl = row['StopLoss']
            risk = buy_price - sl
            shares = int(capital_per_stock // buy_price)
            if shares < 10: continue

            active_positions.append({
                'symbol': symbol,
                'buy_date': date,
                'buy_price': buy_price,
                'initial_sl': sl,
                'sl': sl,
                'shares': shares,
                'remaining_shares': shares,
                'target_1r': buy_price + risk,
                'target_3r': buy_price + 3 * risk,
                'target_5r': buy_price + 5 * risk,
                'booked': {'1R': False, '3R': False, '5R': False}
            })
            print(f"BUY → {symbol} @ {buy_price} | SL: {sl:.2f} | Risk: {risk/buy_price*100:.2f}%")

        # === CHECK ACTIVE POSITIONS ===
        for pos in active_positions[:]:
            if pos['symbol'] != symbol: continue
            high, low, close = row['High'], row['Low'], row['Close']

            # Stop Loss
            if low <= pos['sl']:
                profit = (pos['sl'] - pos['buy_price']) * pos['remaining_shares']
                log_trade(pos, date, pos['sl'], 'StopLoss', pos['remaining_shares'], profit, trade_report)
                final_capital += profit
                active_positions.remove(pos)
                if compound: capital_per_stock = final_capital / no_of_stock_to_trade
                continue

            booked = False

            # 5R (30% of position)
            if not pos['booked']['5R'] and high >= pos['target_5r']:
                sell_shares = int(pos['remaining_shares'] * 0.3)
                profit = (pos['target_5r'] - pos['buy_price']) * sell_shares
                log_trade(pos, date, pos['target_5r'], '5R_Hit', sell_shares, profit, trade_report)
                final_capital += profit
                pos['remaining_shares'] -= sell_shares
                pos['booked']['5R'] = True
                booked = True

            # 3R (40% of remaining)
            elif not pos['booked']['3R'] and high >= pos['target_3r']:
                sell_shares = int(pos['remaining_shares'] * 0.4)
                profit = (pos['target_3r'] - pos['buy_price']) * sell_shares
                log_trade(pos, date, pos['target_3r'], '3R_Hit', sell_shares, profit, trade_report)
                final_capital += profit
                pos['remaining_shares'] -= sell_shares
                pos['booked']['3R'] = True
                booked = True

            # 1R (40% of remaining)
            elif not pos['booked']['1R'] and high >= pos['target_1r']:
                sell_shares = int(pos['remaining_shares'] * 0.4)
                profit = (pos['target_1r'] - pos['buy_price']) * sell_shares
                log_trade(pos, date, pos['target_1r'], '1R_Hit', sell_shares, profit, trade_report)
                final_capital += profit
                pos['remaining_shares'] -= sell_shares
                pos['booked']['1R'] = True
                booked = True

            # Trail remaining with EMA20
            if booked and pos['remaining_shares'] > 0:
                pos['sl'] = max(pos['sl'], row['EMA20'] * 0.98)

            # Final exit on last day
            if date == last_date and pos['remaining_shares'] > 0:
                profit = (close - pos['buy_price']) * pos['remaining_shares']
                log_trade(pos, date, close, 'Final_Trailed', pos['remaining_shares'], profit, trade_report)
                final_capital += profit
                active_positions.remove(pos)


# =============================================================================
#                       REST OF YOUR FRAMEWORK (UNCHANGED + CLEANED)
# =============================================================================

def draw_down_chart():
    all_trades = []
    if os.path.isdir(Reports_Dir) and os.listdir(Reports_Dir):
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
        nifty50_data = get_nifty50_data(from_date, from_date)
        # nifty50_data = fetch_kite_data(enctoken, "NIFTY 50", from_date, to_date, interval='day')

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
        # wrapped_condition = condition_chart.replace(" and ", " and\n")
        textstr = '\n'.join((
            f'capital = {capital}',
            f'no_of_stock_to_trade = {no_of_stock_to_trade}',
            f'compound = {compound}',
            f'target_percentage = {target_percentage}',
            f'stop_loss_percentage = {stop_loss_percentage}',
            # f'trade_Logic:\n{wrapped_condition}'
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

def calculate_holding_days(buy_date, exit_date):
    return (pd.to_datetime(exit_date) - pd.to_datetime(buy_date)).days

def read_stock_files():
    return [f for f in sorted(glob.glob(f'{cvs_data_dir}/*.csv')) if not f.endswith('stock_date_ref.csv')]

def get_stock_for_reference_date(enctoken, cvs_data_dir, cvs_raw_data, start_date, end_date):
    get_stock_for_date_refrence(cvs_raw_data, from_date, to_date)
    file_list = [f'{cvs_raw_data}/stock_date_ref.csv']
    copy_specific_files(file_list, cvs_data_dir)
    date_ref_df = pd.read_csv(f'{cvs_data_dir}/stock_date_ref.csv')
    for date in date_ref_df['Date'].tolist():
        process_date(date, date_ref_df, trade_report)

def procee_final_report(Reports_Dir):
    for symbol, trades in trade_report.items():
        if trades:
            pd.DataFrame(trades).to_csv(f'{Reports_Dir}/{symbol}_trade_report.csv', index=False)
    print(f"\nFINAL CAPITAL: ₹{final_capital:,.2f} | Return: {((final_capital-capital)/capital)*100:.2f}%")
    for report_file in glob.glob(f"{Reports_Dir}/*.csv"):
        symbol = os.path.basename(report_file).replace('_trade_report.csv', '')
        df = pd.read_csv(report_file)
        calculate_summary_per_stock(symbol, df.to_dict('records'), Summary_Dir)
    create_master_file(Summary_Dir, from_date, to_date, capital)

def start_processing_symbols(enctoken, symbols_file, from_date, to_date):
    with open(f'./symbols/{symbols_file}') as f:
        stocks = [line.split('#')[0].strip() for line in f if line.strip() and not line.startswith('#')]
    print(f"Processing {len(stocks)} stocks...")

    def process(stock):
        try:
            mark_signals(enctoken, stock, from_date, to_date)
            return f"Done {stock}"
        except: return f"Failed {stock}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        for r in executor.map(process, stocks): print(r)

    print("Signal generation complete. Starting backtest...")


# =============================================================================
#                                   MAIN
# =============================================================================

enctoken = "your_token_here"
config = configparser.ConfigParser()
config.read('config.cfg')

symbols_file = config['trade_symbol']['symbols_file']
from_date = config['time_management']['from_date']
to_date = config['time_management']['to_date']
capital = float(config['risk_management']['capital'])
no_of_stock_to_trade = int(config['risk_management']['no_of_stock_to_trade'])
compound = config.getboolean('risk_management', 'compound')
charges_percentage = float(config['risk_management']['charges_percentage'])
cleanup_logs = config.getboolean('house_keeping', 'cleanup_logs')
target_percentage = float(config['risk_management']['target_percentage'])
stop_loss_percentage = float(config['risk_management']['stop_loss_percentage'])
symbols_type = symbols_file.split('.')[0]
Reports_Dir = f'{symbols_type}_Reports_{from_date}_to_{to_date}'
Charts_Dir = f'{symbols_type}_Charts_{from_date}_to_{to_date}'
Summary_Dir = f'{symbols_type}_Summary_{from_date}_to_{to_date}'
cvs_data_dir = f'{symbols_type}_Cvs_Data_{from_date}_to_{to_date}'
cvs_raw_data = f'{symbols_type}_Raw_Data_{from_date}_to_{to_date}'

active_positions = []
capital_per_stock = capital / no_of_stock_to_trade
trade_report = {}
final_capital = capital

create_directory(symbols_type, from_date, to_date)
start_processing_symbols(enctoken, symbols_file, from_date, to_date)
get_stock_for_reference_date(enctoken, cvs_data_dir, cvs_raw_data, from_date, to_date)
procee_final_report(Reports_Dir)

if config.getboolean('trade_symbol', 'create_chart', fallback=True):
    draw_down_chart()

print("\nFII MULTI-TARGET SWING LOW STRATEGY BACKTEST COMPLETE")
print("You are now trading like a ₹1000 Cr prop desk.")