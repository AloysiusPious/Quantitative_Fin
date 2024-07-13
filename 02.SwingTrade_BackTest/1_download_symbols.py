import configparser
import pandas as pd
import glob
import os
from swing_util import *


def download_symbols(symbol, start_date, end_date, target_percentage, stop_loss_percentage, ):
    # Fetch data from Yahoo Finance
    data = fetch_yahoo_finance_data(symbol, start_date, end_date)
    if data.empty:
        print(f"No data found for {symbol}. Skipping...")
        return 0, 0  # Skip this stock and return 0 charges and 0 trades
    data.to_csv(f"{cvs_raw_data}/{stock}.csv")

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
##############
symbols_type = symbols_file.split('.')[0]
Reports_Dir = f'{symbols_type}_Reports_{from_date}_to_{to_date}'
Charts_Dir = f'{symbols_type}_Charts_{from_date}_to_{to_date}'
Summary_Dir = f'{symbols_type}_Summary_{from_date}_to_{to_date}'
Master_Dir = f'{symbols_type}_Master_{from_date}_to_{to_date}'
cvs_data_dir = f'{symbols_type}_Cvs_Data_{from_date}_to_{to_date}'
cvs_raw_data = f'{symbols_type}_Raw_Data_{from_date}_to_{to_date}'
##############

if cleanup_logs:
    remove_directory()
create_directory(symbols_type, from_date, to_date)
get_stock_for_date_refrence(cvs_raw_data, from_date, to_date)

with open('./symbols/' + symbols_file, 'r') as file:
    stocks = [line.split('#')[0].strip() for line in file if not line.lstrip().startswith('#')]
total_stocks = len(stocks)
print(f"Total number of stocks: {total_stocks}")
for count, stock in enumerate(stocks, 1):
    print(f"Processing stock {count}/{total_stocks}: {stock}")
    download_symbols(stock + ".NS", from_date, to_date, target_percentage, stop_loss_percentage)