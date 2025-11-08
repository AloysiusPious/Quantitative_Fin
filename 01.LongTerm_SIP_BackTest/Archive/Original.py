import datetime

import pandas as pd
import sys
sys.path.append('/zerodha_lib')
from yfin import *

START_DATE = datetime.date(2019, 1, 1)
END_DATE = datetime.date(2020, 6, 10)
EMA_PERIOD = 200
VOLUME_SPIKE_WINDOW = 20
from_date = '2021-01-01'
to_date = '2024-12-31'

#dataframe = pd.read_csv('SPY.csv')
dataframe = get_df_from_yf("TCS", from_date , to_date )
dataframe = dataframe.to_csv('TCS.csv', index=True)
dataframe = pd.read_csv('TCS.csv')
price_ema = dataframe.loc[:, ['Close']].ewm(span=EMA_PERIOD).mean()

for row in dataframe.iloc[252:].itertuples():
    if price_ema.Close[row.Index] > row.Close:
        # Volume is 7th column
        volume_avg = dataframe.iloc[row.Index - VOLUME_SPIKE_WINDOW:row.Index, [6]].mean()
        # 50% higher volume
        if volume_avg.Volume * 1.5 < row.Volume:
            print(f'{row.Date}')