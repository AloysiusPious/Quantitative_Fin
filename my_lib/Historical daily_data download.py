# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 19:49:29 2021

@author: TAMIL SELVAN
"""

from kiteconnect import KiteConnect, KiteTicker
import pandas as pd
import datetime as dt
import numpy as np
import os
import time
import json
from kite_trade import *
'''
##############
api_key = 'your_api_key'
api_secret = 'your_api_secret'
kite = KiteConnect(api_key)
url = kite.login_url()
request_token = 'your_request_token'
session = kite.generate_session(request_token, api_secret)
access_token = session['access_token']
#######################
'''
enctoken='amIaOy4A956ziT7GfaHEsmvTIXwZEc9GKskuvnQTF0AnH5fSCcL2PwCSrDh+pMWMhpBEdYPi4UyIhELi2AR7G1DwXP58E3B1TLx9Nfq5y+aNw4FGP6sthQ=='
kite = KiteApp(enctoken=enctoken)
###############
#kite.set_access_token(access_token)
###############################
print(kite.margins())

instruments = pd.DataFrame(kite.instruments('NSE'))


def lookup(df, symbol):
    try:
        return df[df.tradingsymbol == symbol].instrument_token.values[0]
    except:
        return -1


def fetchandstoreOHLC(symbol, duration, tf):
    nodata = []
    try:
        if not os.path.isfile("../daily_data/"+symbol + '.csv'):  # checks whether the file is already present or not
            data = pd.DataFrame()
            instrument_token = lookup(instruments,
                                      symbol)  # gets the instrument token of the symbol, where instruments is the dataframe of instruments fetched from kiteconnect
            if duration > 2000:
                no_of_two_thousands = duration // 2000  # gets the no of 2000s : for eg: if duration is 4350, it returns 2 since there are 2 thousands....this is due to the 2000 days limit set by kite
                start = 2000
                end = 0
                for i in range(no_of_two_thousands):
                    df = pd.DataFrame(
                        kite.historical_data(instrument_token, from_date=datetime.today() - timedelta(start),
                                             to_date=datetime.today() - timedelta(end), interval=tf))
                    # print(df)
                    data = data._append(df)
                    start += 2000
                    end += 2000
                dur1 = duration % 2000
                df1 = pd.DataFrame(
                    kite.historical_data(instrument_token,
                                         from_date=datetime.datetime.today() - datetime.timedelta(duration),
                                         to_date=datetime.datetime.today() - datetime.timedelta(duration - dur1),
                                         interval=tf))
                data = data._append(df1)
            else:
                df2 = pd.DataFrame(
                    kite.historical_data(instrument_token,
                                         from_date=datetime.datetime.today() - datetime.timedelta(duration),
                                         to_date=datetime.datetime.today(), interval=tf))
                data = data._append(df2)
            data.set_index("date", inplace=True)
            data.sort_index(ascending=True, inplace=True)
            print('Storing {} days Day_data for {}'.format(duration, symbol))
            data.to_csv("../daily_data/"+symbol + '.csv')
        else:
            print('Data already available for{}'.format(symbol))
            return
    except:
        print("skipping for {}".format(symbol))
        nodata.append(symbol)


'''
#tickers = ['HDFCBANK', 'ICICIBANK', 'KOTAKBANK', 'AXISBANK', 'SBIN', 'RELIANCE', 'TCS', 'INFY', 'HINDUNILVR', 'HDFC',
           'BAJFINANCE', 'WIPRO', 'BHARTIARTL', 'HCLTECH', 'ASIANPAINT', 'ITC', 'LT', 'ULTRACEMCO',
           'MARUTI', 'SUNPHARMA', 'TATASTEEL', 'JSWSTEEL', 'TITAN', 'ADANIPORTS', 'ONGC', 'HDFCLIFE', 'TECHM',
           'DIVISLAB', 'POWERGRID', 'SBILIFE', 'NTPC', 'BAJAJ-AUTO', 'BPCL', 'IOC', 'M&M', 'SHREECEM', 'HINDALCO',
           'GRASIM', 'BRITANNIA', 'TATAMOTORS', 'COALINDIA', 'TATACONSUM', 'INDUSINDBK', 'DRREDDY', 'CIPLA',
           'EICHERMOT', 'UPL', 'NESTLEIND', 'HEROMOTOCO', 'NIFTY 50', 'NIFTY BANK']
           '''
my_file = open("../symbols/custom.txt", "r")
# reading the file
data = my_file.read()
# replacing end splitting the text
# when newline ('\n') is seen.
tickers = data.split("\n")
#print(tickers)
my_file.close()

for ticker in tickers:
    fetchandstoreOHLC(ticker, 10000, 'day')

