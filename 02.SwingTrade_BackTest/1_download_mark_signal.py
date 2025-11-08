import configparser
import pandas as pd
import glob
import os
from swing_util import *


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
        sce_1 = buy_today_cond and is_previous_green and yday_50EMA_above_200EMA and yday_close_above_7EMA and pd_rsi_below_n(data, i-2, 14, 40)# and pd_rsi_above_n(data, i-2, 14, 40)
        sce_2 = buy_today_cond and is_previous_green and yday_100EMA_above_200EMA and six_candle_below_EMA_7 and yday_close_above_7EMA
        sce_3 = buy_today_cond and is_previous_green and yday_100EMA_above_200EMA and yday_20EMA_below_200EMA and six_candle_avg_below_EMA_7 and yday_close_above_7EMA and pd_rsi_below_n(data, i-2, 14, 40)
        sce_4 = buy_today_cond and is_previous_green and yday_50EMA_above_200EMA and two_candle_avg_below_EMA_7 and yday_close_above_7EMA and pd_rsi_below_n(data, i-2, 14, 50)
        sce_7 = buy_today_cond and is_previous_green and EMA_50_100_200_UPTREND and yday_20EMA_above_50EMA and yday_close_above_7EMA and macd_buy(data, i-1)
        sce_8 = buy_today_cond and is_previous_green  and (lowest_low < data.iloc[i - 8]['EMA_200'])  \
                and (highest_close > data.iloc[i - 1]['EMA_200']) and yday_close_above_7EMA #and EMA_50_100_200_UPTREND
        #sce_3 = buy_today_cond and is_previous_green and yday_50EMA_above_200EMA and yday_open_below_7EMA and yday_close_above_7EMA
        #sce_4 = buy_today_cond and is_previous_green and yday_50EMA_above_200EMA and yday_open_below_20EMA and yday_close_above_20EMA
        #sce_5 = buy_today_cond and is_previous_green and yday_20EMA_gt_50EMA_gt_100EMA_gt_200EMA and yday_close_above_7EMA
        ##sce_2 and pd_rsi_below_n(data, i-2, 14, 50) and pd_rsi_below_n(data, i-5, 14, 35):
        if sce_8:
            bought_price = round_to_nearest_0_05(data.iloc[i - 1]['High'])
            #stop_loss = round_to_nearest_0_05(bought_price * (1 - stop_loss_percentage / 100))
            stop_loss = lowest_low - (bought_price - lowest_low)
            target = round_to_nearest_0_05(bought_price * (1 + target_percentage / 100))
            data.loc[data.index[i], 'Buy_Signal'] = round_to_nearest_0_05(data.iloc[i - 1]['High'])
            #data.loc[data.index[i], 'Target'] = target
            data.loc[data.index[i], 'Target'] = bought_price + (bought_price - lowest_low) * 2
            data.loc[data.index[i], 'StopLoss'] = stop_loss
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
