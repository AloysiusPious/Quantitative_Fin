import pandas as pd
import yfinance as yf
#import talib as ta
import os
import numpy as np
import configparser
import shutil
import glob
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, time
import re
from swing_util import *


def process_files(directory):
    final_data = []

    for filename in os.listdir(directory):
        if filename.startswith("Master_") and filename.endswith(".csv"):
            start_date, end_date = extract_date_range_from_filename(filename)
            if start_date and end_date:
                filepath = os.path.join(directory, filename)
                df = pd.read_csv(filepath)
                overall_row = df[df['Stock Name'] == 'Overall'].iloc[0]
                final_data.append([start_date, end_date] + overall_row.tolist())

    final_df = pd.DataFrame(final_data, columns=[
        'Start Year', 'End Year', 'Stock Name', 'Total Trades', 'No of Winning Trade', 'No of Losing Trade',
        'Winning Trade Percentage', 'Losing Trade Percentage', 'Total Profit', 'Total Cumulative Return Percentage',
        'Total Charges Paid', 'Profit After Charges', 'Profit Percentage After Charges', 'Total Charges Percentage'
    ])

    # Calculate grand totals and averages
    grand_totals = {
        'Start Year': 'Grand Total',
        'End Year': '',
        'Stock Name': 'Overall',
        'Total Trades': final_df['Total Trades'].sum(),
        'No of Winning Trade': final_df['No of Winning Trade'].sum(),
        'No of Losing Trade': final_df['No of Losing Trade'].sum(),
        'Winning Trade Percentage': round((final_df['No of Winning Trade'].sum() / final_df['Total Trades'].sum()) * 100, 2) if final_df['Total Trades'].sum() > 0 else 0,
        'Losing Trade Percentage': round((final_df['No of Losing Trade'].sum() / final_df['Total Trades'].sum()) * 100, 2) if final_df['Total Trades'].sum() > 0 else 0,
        'Total Profit': final_df['Total Profit'].sum(),
        'Total Cumulative Return Percentage': round_to_nearest_five_cents(final_df['Total Cumulative Return Percentage'].mean()),
        'Total Charges Paid': final_df['Total Charges Paid'].sum(),
        'Profit After Charges': round_to_nearest_five_cents(final_df['Profit After Charges'].sum()),
        'Profit Percentage After Charges': round_to_nearest_five_cents(final_df['Profit Percentage After Charges'].mean()),
        'Total Charges Percentage': round_to_nearest_five_cents(final_df['Total Charges Percentage'].mean())
    }

    overall_df = pd.DataFrame([grand_totals])
    final_df = pd.concat([final_df, overall_df], ignore_index=True)

    # Save the consolidated DataFrame to CSV
    final_df.to_csv(f"{directory}/Final_Consolidated.csv", index=False)
    print("Final_Consolidated.csv created successfully.")

def create_directory(symbols_type):
    directories_to_create = [f'Reports_{from_date}_to_{to_date}', f'Charts_{from_date}_to_{to_date}',
                             f'Summary_{from_date}_to_{to_date}', f'Master_{from_date}_to_{to_date}', f'Cvs_Data_{from_date}_to_{to_date}']
    # Iterate over each directory and create it if it does not exist
    for directory in directories_to_create:
        directory_name = symbols_type + "_" + directory
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
def create_master_file(summary_dir):
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
        if filename.endswith(f"_summary_{from_date}_to_{to_date}.csv"):
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

    # Save Master DataFrame to CSV
    master_df.to_csv(f"{Master_Dir}/Master_{from_date}_to_{to_date}.csv", index=False)

def analyze_csv_files(provided_date):
    open_positions = 0
    csv_files = [f for f in os.listdir(Reports_Dir) if f.endswith('.csv')]

    if len(csv_files) < no_of_stock_to_trade:
        return 0, True

    for csv_file in csv_files:
        file_path = os.path.join(Reports_Dir, csv_file)
        data = pd.read_csv(file_path)

        # Convert dates to datetime for comparison
        data['Buy Date'] = pd.to_datetime(data['Buy Date'])
        data['Exited Date'] = pd.to_datetime(data['Exited Date'])
        provided_date = pd.to_datetime(provided_date)

        # Check for open positions
        for i, row in data.iterrows():
            if row['Buy Date'] <= provided_date and (pd.isna(row['Exited Date']) or row['Exited Date'] > provided_date):
                open_positions += 1
    if open_positions > no_of_stock_to_trade:
        print(f'Current Open Position more then {no_of_stock_to_trade} : {open_positions}')
    return open_positions, open_positions < no_of_stock_to_trade

def check_target_stop_loss_trades(data, capital_per_stock, target_percentage, stop_loss_percentage, ):
    #total_charges_percentage = 0.2
    trade = None
    trades = []
    total_charges_paid = 0
    ###
    for i in range(100, len(data)):  # Start from the 100th day to have enough data for calculations
        if not trade and data.index[i].date() >= datetime.strptime(from_date, '%Y-%m-%d').date():
            is_previous_green = (data.iloc[i - 1]['Close'] > data.iloc[i - 1]['Open'])

            is_current_red = (data.iloc[i]['Close'] < data.iloc[i]['Open'])
            is_50EMA_above_200EMA = data.iloc[i]['EMA_50'] > data.iloc[i]['EMA_200']
            is_prev_close_below_200EMA = data.iloc[i - 1]['Close'] < data.iloc[i - 1]['EMA_200']
            yday_50EMA_above_200EMA = data.iloc[i - 1]['EMA_50'] > data.iloc[i - 1]['EMA_200']
            yday_20EMA_above_50EMA = data.iloc[i - 1]['EMA_20'] > data.iloc[i - 1]['EMA_50']
            two_prev_yday_20EMA_below_50EMA = data.iloc[i - 2]['EMA_20'] < data.iloc[i - 2]['EMA_50'] or data.iloc[i - 3]['EMA_20'] < data.iloc[i - 3]['EMA_50']
            yday_7EMA_below_20EMA = data.iloc[i - 1]['EMA_7'] < data.iloc[i - 1]['EMA_20']
            ###
            yday_close_above_7EMA = data.iloc[i - 1]['Close'] > data.iloc[i - 1]['EMA_7']
            yday_close_below_7EMA = data.iloc[i - 1]['Close'] < data.iloc[i - 1]['EMA_7']
            ####
            yday_close_above_20EMA = data.iloc[i - 1]['Close'] > data.iloc[i - 1]['EMA_20']
            two_day_b4_close_below_7EMA = data.iloc[i - 2]['Close'] < data.iloc[i - 2]['EMA_7']
            three_day_b4_close_below_7EMA = data.iloc[i - 3]['Close'] < data.iloc[i - 3]['EMA_7']
            four_day_b4_close_below_7EMA = data.iloc[i - 4]['Close'] < data.iloc[i - 4]['EMA_7']
            five_day_b4_close_below_7EMA = data.iloc[i - 5]['Close'] < data.iloc[i - 5]['EMA_7']
            is_open_below_yday_close = data.iloc[i]['Open'] < data.iloc[i - 1]['Close']
            is_tday_high_break_yday_high = data.iloc[i]['High'] > data.iloc[i - 1]['High']
            ema7_break_for_first_time = yday_close_above_7EMA and (two_day_b4_close_below_7EMA or three_day_b4_close_below_7EMA or four_day_b4_close_below_7EMA or five_day_b4_close_below_7EMA)
            ######
            buy_today_cond = is_tday_high_break_yday_high and is_open_below_yday_close
            sce_1 = buy_today_cond and is_previous_green and yday_50EMA_above_200EMA and yday_close_above_7EMA and yday_unusual_volume(data, i)
            sce_1_1 = buy_today_cond and is_previous_green and yday_20EMA_above_50EMA and ema7_break_for_first_time
            sce_1_2 = buy_today_cond and is_previous_green and yday_20EMA_above_50EMA and ema7_break_for_first_time and yday_unusual_volume(data, i)

            if sce_1_1:
                #open_positions, open_pos = analyze_csv_files(data.index[i].date())
                #if open_pos:
                #    print(f'Current Open Position : {open_positions}')
                buy_date = data.index[i].date()
                bought_price = round_to_nearest_five_cents(data.iloc[i - 1]['High'])
                quantity_bought = int(capital_per_stock / bought_price)
                stop_loss = round_to_nearest_five_cents(bought_price * (1 - stop_loss_percentage / 100))
                target = round_to_nearest_five_cents(bought_price * (1 + target_percentage / 100))
                # Calculate charges for buying
                buy_turnover = bought_price * quantity_bought
                buy_charges = (total_charges_percentage / 100) * buy_turnover
                total_charges_paid += round_to_nearest_five_cents(buy_charges)
                trade = {
                    'Buy Date': buy_date,
                    'Bought Price': bought_price,
                    'Quantity Bought': quantity_bought,
                    'Invested Amount': capital_per_stock,
                    'Stop Loss': stop_loss,
                    'Target': target,
                    'Exited Date': None,
                    'Exited Price': None,
                    'Profit Amount': None,
                    'Trade Status': None
                }
                data.loc[data.index[i], 'Buy Signal'] = data.iloc[i]['Low'].astype(float)
                data.loc[data.index[i], 'Target Level'] = target
                data.loc[data.index[i], 'Stop Loss Level'] = stop_loss
        elif trade and (trade['Stop Loss'] >= data.iloc[i]['Low'] or trade['Target'] <= data.iloc[i]['High']):
            if trade['Target'] <= data.iloc[i]['High']:
                #data.loc[data.index[i], 'Trade Status'] = 'Target'
                profit_amount = round_to_nearest_five_cents((trade['Target'] - trade['Bought Price']) * trade['Quantity Bought'])
                trade['Exited Price'] = trade['Target']
                trade['Trade Status'] = 'Target'
                trade['Exited Date'] = data.index[i].date()
            elif trade['Stop Loss'] >= data.iloc[i]['Low']:
                #data.loc[data.index[i], 'Trade Status'] = 'StopLoss'
                profit_amount = round_to_nearest_five_cents((trade['Stop Loss'] - trade['Bought Price']) * trade['Quantity Bought'])
                trade['Exited Price'] = trade['Stop Loss']
                trade['Trade Status'] = 'StopLoss'
                trade['Exited Date'] = data.index[i].date()
            sell_turnover = bought_price * trade['Quantity Bought']
            # Calculate charges for selling
            sell_charges = (total_charges_percentage / 100) * sell_turnover
            total_charges_paid += round_to_nearest_five_cents(sell_charges)
            trade['Profit Amount'] = round_to_nearest_five_cents(profit_amount - buy_charges - sell_charges)
            trades.append(trade)
            #print(trade)
            trade = None
            if compound:
                capital_per_stock += int(profit_amount)
    if create_chart and len(trades) != 0:
        visualize(data, 'Target Level', 'Stop Loss Level', stock, Charts_Dir)
    num_buy_signals = data["Buy Signal"].notna().sum()
    #print(len(trades))
    #print(num_buy_signals)
    data.to_csv(f"{cvs_data_dir}/{stock}_data.csv")
    if len(trades) != num_buy_signals:
        data, trades, total_charges_paid_cl = get_last_unclosed_trade(data, trades, capital_per_stock, total_charges_paid)
        total_charges_paid = total_charges_paid_cl + total_charges_paid
    return trades, total_charges_paid
def mark_signals(symbol, start_date, end_date, target_percentage, stop_loss_percentage, ):
    # Fetch data from Yahoo Finance
    data = fetch_yahoo_finance_data(symbol, start_date, end_date)
    if data.empty:
        print(f"No data found for {symbol}. Skipping...")
        return 0, 0  # Skip this stock and return 0 charges and 0 trades
    # Calculate EMA
    data = calculate_ema(data, 200)
    data = calculate_ema(data, 50)
    data = calculate_ema(data, 100)
    data = calculate_ema(data, 20)
    data = calculate_ema(data, 7)
    #total_charges_percentage = 0.2
    trade = None
    trades = []
    total_charges_paid = 0
    ###
    for i in range(100, len(data)):  # Start from the 100th day to have enough data for calculations
        if not trade and data.index[i].date() >= datetime.strptime(from_date, '%Y-%m-%d').date():
            data = convert_all_col_digit(data)
            is_previous_green = (data.iloc[i - 1]['Close'] > data.iloc[i - 1]['Open'])

            is_current_red = (data.iloc[i]['Close'] < data.iloc[i]['Open'])
            is_50EMA_above_200EMA = data.iloc[i]['EMA_50'] > data.iloc[i]['EMA_200']
            is_prev_close_below_200EMA = data.iloc[i - 1]['Close'] < data.iloc[i - 1]['EMA_200']
            yday_50EMA_above_200EMA = data.iloc[i - 1]['EMA_50'] > data.iloc[i - 1]['EMA_200']
            yday_20EMA_above_50EMA = data.iloc[i - 1]['EMA_20'] > data.iloc[i - 1]['EMA_50']
            two_prev_yday_20EMA_below_50EMA = data.iloc[i - 2]['EMA_20'] < data.iloc[i - 2]['EMA_50'] or data.iloc[i - 3]['EMA_20'] < data.iloc[i - 3]['EMA_50']
            yday_7EMA_below_20EMA = data.iloc[i - 1]['EMA_7'] < data.iloc[i - 1]['EMA_20']
            ###
            yday_close_above_7EMA = data.iloc[i - 1]['Close'] > data.iloc[i - 1]['EMA_7']
            yday_close_below_7EMA = data.iloc[i - 1]['Close'] < data.iloc[i - 1]['EMA_7']
            ####
            yday_close_above_20EMA = data.iloc[i - 1]['Close'] > data.iloc[i - 1]['EMA_20']
            two_day_b4_close_below_7EMA = data.iloc[i - 2]['Close'] < data.iloc[i - 2]['EMA_7']
            three_day_b4_close_below_7EMA = data.iloc[i - 3]['Close'] < data.iloc[i - 3]['EMA_7']
            four_day_b4_close_below_7EMA = data.iloc[i - 4]['Close'] < data.iloc[i - 4]['EMA_7']
            five_day_b4_close_below_7EMA = data.iloc[i - 5]['Close'] < data.iloc[i - 5]['EMA_7']
            is_open_below_yday_close = data.iloc[i]['Open'] < data.iloc[i - 1]['Close']
            is_tday_high_break_yday_high = data.iloc[i]['High'] > data.iloc[i - 1]['High']
            ema7_break_for_first_time = yday_close_above_7EMA and (two_day_b4_close_below_7EMA or three_day_b4_close_below_7EMA or four_day_b4_close_below_7EMA or five_day_b4_close_below_7EMA)
            ######
            buy_today_cond = is_tday_high_break_yday_high and is_open_below_yday_close
            sce_1 = buy_today_cond and is_previous_green and yday_50EMA_above_200EMA and yday_close_above_7EMA and yday_unusual_volume(data, i)
            sce_1_1 = buy_today_cond and is_previous_green and yday_20EMA_above_50EMA and ema7_break_for_first_time
            sce_1_2 = buy_today_cond and is_previous_green and yday_20EMA_above_50EMA and ema7_break_for_first_time and yday_unusual_volume(data, i)
            if sce_1_1:
                bought_price = round_to_nearest_five_cents(data.iloc[i - 1]['High'])
                stop_loss = round_to_nearest_five_cents(bought_price * (1 - stop_loss_percentage / 100))
                target = round_to_nearest_five_cents(bought_price * (1 + target_percentage / 100))
                data.loc[data.index[i], 'Buy Signal'] = data.iloc[i]['Low'].astype(float)
                data.loc[data.index[i], 'Target Level'] = target
                data.loc[data.index[i], 'Stop Loss Level'] = stop_loss
    data = data.loc[from_date:]
    data.to_csv(f"{cvs_data_dir}/{stock}_data.csv")
def get_last_unclosed_trade(data, trades, capital_per_stock, total_charges_paid):
        ########
        non_empty_indices = data[data["Buy Signal"].notna()].index
        last_buy_signal_index = non_empty_indices[len(non_empty_indices) - 1]
        buy_date = last_buy_signal_index.strftime("%Y-%m-%d")
        #print(buy_date)
        #print(non_empty_indices)
        #
        prev_buy_signal_index = data[data["Buy Signal"].notna()].index[-1]
        #print(prev_buy_signal_index)
        #previous_day_data = data.loc[prev_buy_signal_index - data.Timedelta(days=1)]
        previous_trading_day_index = data.index[data.index.get_loc(prev_buy_signal_index) - 1].strftime("%Y-%m-%d")
        #print(previous_trading_day_index)
        previous_day_data = data.loc[previous_trading_day_index]
        bought_price = previous_day_data['High']
        #print(bought_price)
        quantity_bought = int(capital_per_stock / bought_price)
        stop_loss = round_to_nearest_five_cents(bought_price * (1 - stop_loss_percentage / 100))
        target = round_to_nearest_five_cents(bought_price * (1 + target_percentage / 100))
        # Calculate charges for buying
        buy_turnover = bought_price * quantity_bought
        buy_charges = (total_charges_percentage / 100) * buy_turnover
        total_charges_paid += round_to_nearest_five_cents(buy_charges)
        trade = {
            'Buy Date': buy_date,
            'Bought Price': bought_price,
            'Quantity Bought': quantity_bought,
            'Invested Amount': capital_per_stock,
            'Stop Loss': stop_loss,
            'Target': target,
            'Exited Date': None,
            'Exited Price': None,
            'Profit Amount': None,
            'Trade Status': None
        }
        profit_amount = round_to_nearest_five_cents((data.iloc[-1]['Close'] - trade['Bought Price']) * trade['Quantity Bought'])
        sell_turnover = bought_price * trade['Quantity Bought']
        trade['Exited Date'] = data.index[-1].strftime("%Y-%m-%d")
        trade['Exited Price'] = data.iloc[-1]['Close']
        trade['Trade Status'] = 'Cl_Open'
        # Calculate charges for selling
        sell_charges = (total_charges_percentage / 100) * sell_turnover
        total_charges_paid += round_to_nearest_five_cents(sell_charges)
        trade['Profit Amount'] = round_to_nearest_five_cents(profit_amount - buy_charges - sell_charges)
        trades.append(trade)
        return data, trades, total_charges_paid
def check_for_stock_buy(capital_per_stock, target_percentage, stop_loss_percentage, ):
    trade = None
    trades = []
    total_charges_paid = 0
    # Load the stock_date_ref.csv file
    stock_date_ref = pd.read_csv(f'{cvs_data_dir}/stock_date_ref.csv')
    # Get a list of all the other CSV files in the directory
    csv_files = [f for f in os.listdir(f'{cvs_data_dir}') if f.endswith('.csv') and f != 'stock_date_ref.csv']
    # Sort the list of CSV files alphabetically
    csv_files.sort()
    # Iterate over each date in the stock_date_ref file
    for index, row in stock_date_ref.iterrows():
        date = row['Date']
        # Iterate over each CSV file
        for file in csv_files:
            # Load the CSV file
            print((cvs_data_dir+"/"+file))
            #df = pd.read_csv(cvs_data_dir+"/"+file)
            # Filter the data for the current date
            #data = df[df['Date'] == date]
def main(symbol, start_date, end_date, capital, target_percentage, stop_loss_percentage):
    try:
        capital_per_stock = round(capital / no_of_stock_to_trade, 2)
        # trades, total_charges_paid = check_target_stop_loss_trades(data, capital_per_stock, target_percentage, stop_loss_percentage)
        check_for_stock_buy(capital_per_stock, target_percentage, stop_loss_percentage)
        '''
        if not trades:
            print(f"No trades found for {symbol}. Skipping...")
            return total_charges_paid, trades  # Return charges paid, even if no trades found
        # Calculate No of holding Days
        df = pd.DataFrame(trades)
        df['Buy Date'] = pd.to_datetime(df['Buy Date'])  # Convert Buy Date to datetime format
        df['Exited Date'] = pd.to_datetime(df['Exited Date'])  # Convert Exited Date to datetime format
        df['No of holding Days'] = round((df['Exited Date'] - df['Buy Date']).dt.days, 2)  # Calculate holding days
        # Calculate Profit %
        df['Profit %'] = round((df['Profit Amount'] / df['Invested Amount']) * 100, 2)
        # Save DataFrame to CSV with Bought Price field
        df.to_csv(f"{Reports_Dir}/{symbol}_trades_{from_date}_to_{to_date}.csv", index=False)
        print("Creating Draw-Down Chart.....")
        draw_down_chart()
        # Create summary DataFrame
        summary_df = pd.DataFrame({
            'Total Trades': [len(trades)],
            'No of Winning Trade': [
                len([trade for trade in trades if trade['Profit Amount'] and trade['Profit Amount'] > 0])],
            'No of Losing Trade': [
                len([trade for trade in trades if trade['Profit Amount'] and trade['Profit Amount'] < 0])],
            'Winning Trade Percentage': [
                round(len([trade for trade in trades if trade['Profit Amount'] and trade['Profit Amount'] > 0]) / len(
                    trades) * 100, 2)],
            'Losing Trade Percentage': [
                round(len([trade for trade in trades if trade['Profit Amount'] and trade['Profit Amount'] < 0]) / len(
                    trades) * 100, 2)],
            'Total Profit': [round(sum([trade['Profit Amount'] for trade in trades if trade['Profit Amount']]), 2)],
            'Cumulative Return Percentage': [
                round((sum([trade['Profit Amount'] for trade in trades if trade['Profit Amount']]) / capital) * 100, 2)],
            'Total Charges Paid': [total_charges_paid]
        })

        # Save summary DataFrame to CSV
        summary_df.to_csv(f"{Summary_Dir}/{symbol}_summary_{from_date}_to_{to_date}.csv", index=False)
        return total_charges_paid, trades  # Return the total charges paid and trades for this stock
    '''
    except Exception as e:
        print(f"Error processing {symbol}: {e}")
        return 0, []  # Return 0 charges and an empty list of trades in case of error
def get_stock_for_date_refrence():
    print(f'Downloading stock for Date reference..')
    nifty50_data = get_nifty50_data(from_date, to_date)
    nifty50_data.reset_index(inplace=True)
    nifty50_data.rename(columns={'index': 'Date'})
    nifty50_data = nifty50_data[['Date']]
    nifty50_data.to_csv(f"{cvs_data_dir}/stock_date_ref.csv", index=False, date_format='%Y-%m-%d')
    print(f'Ok.')
if __name__ == "__main__":
    # Create a ConfigParser object
    config = configparser.ConfigParser()
    # Read the cfg file
    config.read('config.cfg')
    symbols_file = str(config['trade_symbol']['symbols_file'])
    create_chart = str(config['trade_symbol']['create_chart'])
    create_chart = True if create_chart == 'true' else False
    # Access sections and keys
    from_date = str(config['time_management']['from_date'])
    to_date = str(config['time_management']['to_date'])
    #from_date = datetime.strptime(config['time_management']['from_date'], '%Y-%m-%d').date()
    #to_date = datetime.strptime(config['time_management']['to_date'], '%Y-%m-%d').date()
    year_wise = str(config['time_management']['year_wise'])
    year_wise = True if year_wise == 'true' else False
    year_split = int(config['time_management']['year_split'])
    capital = float(config['risk_management']['capital'])
    no_of_stock_to_trade = int(config['risk_management']['no_of_stock_to_trade'])
    compound = str(config['risk_management']['compound'])
    compound = True if compound == 'true' else False
    target_percentage = float(config['risk_management']['target_percentage'])
    stop_loss_percentage = float(config['risk_management']['stop_loss_percentage'])
    total_charges_percentage = float(config['risk_management']['charges_percentage'])

    ###
    cleanup_logs = str(config['house_keeping']['cleanup_logs'])
    cleanup_logs = True if cleanup_logs == 'true' else False
    if cleanup_logs:
        remove_directory()
    ##############
    symbols_type = symbols_file.split('.')[0]
    Reports_Dir = f'{symbols_type}_Reports_{from_date}_to_{to_date}'
    Charts_Dir = f'{symbols_type}_Charts_{from_date}_to_{to_date}'
    Summary_Dir = f'{symbols_type}_Summary_{from_date}_to_{to_date}'
    Master_Dir = f'{symbols_type}_Master_{from_date}_to_{to_date}'
    cvs_data_dir = f'{symbols_type}_Cvs_Data_{from_date}_to_{to_date}'

    ##############
    create_directory(symbols_type)
    get_stock_for_date_refrence()
    with open('./symbols/' + symbols_file, 'r') as file:
        stocks = [line.split('#')[0].strip() for line in file if not line.lstrip().startswith('#')]
    total_stocks = len(stocks)
    print(f"Total number of stocks: {total_stocks}")
    # Initialize lists to store summarized information
    summary_data = []
    end_year_c = 0
    # Initialize total charges variable
    total_charges_for_all_stocks = 0
    for count, stock in enumerate(stocks, 1):
        #print(".")
        print(f"Processing stock {count}/{total_stocks}: {stock}")
        # Call main function with parameters and aggregate total charges
        mark_signals(stock + ".NS", from_date, to_date, target_percentage, stop_loss_percentage)
        #charges_paid, trades = main(stock + ".NS", from_date, to_date, capital, target_percentage, stop_loss_percentage)
        #total_charges_for_all_stocks += charges_paid
    # Call function to create the Master_no_Compound_sce_5.csv file
    #create_master_file(Summary_Dir)

    # Print total charges paid for all stocks
    print(f"Total charges paid for all stocks: ₹{total_charges_for_all_stocks:.2f}")
    print("All stocks processed.")