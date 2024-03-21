'''
"minute": 30,
"hour": 365,
"day": 2000,

"3minute": 90,
"5minute": 90,
"10minute": 90,
"15minute": 180,
"30minute": 180,
"60minute": 365
'''
from kiteconnect import KiteConnect, KiteTicker
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os
import time
import json
from kite_trade import *



def lookup(df, symbol):
    try:
        return df[df.tradingsymbol == symbol].instrument_token.values[0]
    except:
        return -1


def get_data_from_zerodha(symbol, from_date_str, to_date_str, tf):

    nodata = []
    ###############################################
    enctoken = 'amIaOy4A956ziT7GfaHEsmvTIXwZEc9GKskuvnQTF0AnH5fSCcL2PwCSrDh+pMWMhpBEdYPi4UyIhELi2AR7G1DwXP58E3B1TLx9Nfq5y+aNw4FGP6sthQ=='
    kite = KiteApp(enctoken=enctoken)
    ###############################
    # print(kite.margins())
    instruments = pd.DataFrame(kite.instruments('NSE'))
    #########################################

    try:
        # Convert strings to datetime objects
        from_date = datetime.strptime(from_date_str, '%Y-%m-%d')
        to_date = datetime.strptime(to_date_str, '%Y-%m-%d')
        # Calculate the difference in days
        duration = (to_date - from_date).days
        #print(duration)
        nodata = []
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
                data = data._append(df)
                start += 2000
                end += 2000
            dur1 = duration % 2000
            df1 = pd.DataFrame(
                kite.historical_data(instrument_token, from_date=datetime.today() - timedelta(duration),
                                     to_date=datetime.today() - timedelta(duration - dur1), interval=tf))
            data = data._append(df1)
        else:
            df2 = pd.DataFrame(
                kite.historical_data(instrument_token, from_date=datetime.datetime.today() - datetime.timedelta(duration),
                                     to_date=datetime.datetime.today(), interval=tf))
            data = data._append(df2)
        data.set_index("date", inplace=True)
        data.sort_index(ascending=True, inplace=True)
        # Convert index to datetime
        data.index = pd.to_datetime(data.index)

        # Remove time from index
        data.index = data.index.date
        # Rename index name
        data.index.name = 'date'
        #data.index = data.index(data.index.date) if tf == 'day' else data
        # Capitalize the first letter of each column name
        data.columns = data.columns.str.capitalize()
        # Capitalize the first letter of the index name
        data = data.rename_axis(index=lambda x: x.capitalize())
        data = data.astype(str)
        data['Volume'] = data['Volume'].astype(int)  # Convert 'Volume' to integer
        data = data.astype({'Open': float, 'High': float, 'Low': float, 'Close': float})
        data.index = pd.to_datetime(data.index)

        return data
    except:
        print("skipping for {}".format(symbol))
        nodata.append(symbol)


'''
#########################################
stock = "TCS"
from_date = '2018-01-01'
to_date = '2024-12-31'
time_frame = 'day'

print(get_data_from_zerodha(stock, from_date, to_date, time_frame))
##########################################
'''
