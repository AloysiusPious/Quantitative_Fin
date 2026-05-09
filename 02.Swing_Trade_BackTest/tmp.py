def mark_signals_1(symbol, start_date, end_date):
    file_path = f'{cvs_raw_data}/{symbol}.csv'

    if not os.path.exists(file_path):
        print(f"{symbol} Not found locally — downloading ...")
        ENCTOKEN = "mFm/EceEQnwbTy5IWDufMWv/jCZQryYGd9UI6SdYZoakQPRbgQ0WC35PMtygKxcUXr28ZOemMDlCRZC/oIG+MLuaDmldt6kBo9s8ImD0F9xJq3QG0CQenA=="
        data = fetch_kite_data(ENCTOKEN, symbol, start_date, end_date, interval='day')
        if data is None or data.empty:
            print(f"No data for {symbol}")
            return
        data.to_csv(file_path, index=False)
    else:
        print(f"{symbol} found in local and processing it ...")
        data = pd.read_csv(file_path)

    # --- Indicators ---
    data['Low_252min'] = data['Low'].shift(1).rolling(window=252).min()
    data['Vol_20MA'] = data['Volume'].rolling(window=20).mean()
    data['Cond_VolumeConfirm'] = data['Volume'].shift(-1) > data['Vol_20MA']

    # --- MACD ---
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD_Line'] = ema12 - ema26
    data['MACD_Signal'] = data['MACD_Line'].ewm(span=9, adjust=False).mean()
    data['MACD_Cross'] = (data['MACD_Line'] > data['MACD_Signal']) & (
        data['MACD_Line'].shift(1) <= data['MACD_Signal'].shift(1)
    )
    data['MACD_Line'] = data['MACD_Line'].round(2)
    data['MACD_Signal'] = data['MACD_Signal'].round(2)

    # --- RSI (14-period) ---
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data['RSI'] = data['RSI'].round(2)

    # --- Conditions ---
    cond_low = (data['Low'].shift(1) >= data['Low_252min'] * 1.02) & \
               (data['Low'].shift(1) <= data['Low_252min'] * 1.05)
    cond_green = data['Close'] > data['Close'].shift(1)
    cond_volume = data['Cond_VolumeConfirm']
    cond_macd_signal_low = data['MACD_Signal'] < -10
    cond_rsi_range = (data['RSI'] > 30) & (data['RSI'] < 50)
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    cond_trend = data['Close'] > data['SMA_5']  # only buy if above 200DMA
    data['Vol_EMA5'] = data['Volume'].ewm(span=5).mean()
    cond_vol_accel = data['Vol_EMA5'] > data['Vol_EMA5'].shift(5)
    # --- Combine Buy Signal ---
    cond_buy = cond_low & cond_macd_signal_low & cond_vol_accel

    # --- Targets and Stoploss ---
    data.loc[cond_buy, 'Buy_Signal'] = data.loc[cond_buy, 'Close'].round(2)
    data.loc[cond_buy, 'StopLoss'] = data.loc[cond_buy, 'Low_252min'].round(2)
    risk = data['Buy_Signal'] - data['StopLoss']
    data.loc[cond_buy, 'Target'] = data['Buy_Signal'] + 3 * risk

    # --- Filter & Save ---
    data = data.loc[data['Date'] >= start_date]
    data.to_csv(f"{cvs_data_dir}/{symbol}.csv", index=False)
    print(f"✅ Processed {symbol} (MACD < -2, RSI 30–50)")


def mark_signals_3(symbol, start_date, end_date):
    file_path = f'{cvs_raw_data}/{symbol}.csv'

    if not os.path.exists(file_path):
        print(f"{symbol} Not found locally — downloading ...")
        ENCTOKEN = "ciejrdDMeGXHRLX0y2jRLjEsPndYUV2uzo5jOufr4yZuFfKabpDTfB70ieFFAXRUlowwoLN+Fj7UVs9XjyR8QP1fA0baVX9plepBzCL5Axef+fQ98xzJCA=="
        data = fetch_kite_data(ENCTOKEN, symbol, start_date, end_date, interval='day')
        if data is None or data.empty:
            print(f"No data for {symbol}")
            return
        data.to_csv(file_path, index=False)
    else:
        print(f"{symbol} found locally and processing ...")
        data = pd.read_csv(file_path)

    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values('Date', inplace=True)
    data.reset_index(drop=True, inplace=True)

    # --- Indicators ---
    data['Low_126min'] = data['Low'].shift(1).rolling(window=252, min_periods=50).min()

    # --- MACD ---
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD_Line'] = ema12 - ema26
    data['MACD_Signal'] = data['MACD_Line'].ewm(span=9, adjust=False).mean()
    data['MACD_Slope'] = data['MACD_Line'].diff()
    cond_macd_turn = (data['MACD_Slope'] > 0) & (data['MACD_Line'] < 0)  # relaxed

    # --- RSI ---
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    data['RSI'] = 100 - (100 / (1 + rs))
    cond_rsi = (data['RSI'] > 30) & (data['RSI'] < 60)  # widened slightly

    # --- Volume ---
    data['Vol_20MA'] = data['Volume'].rolling(20).mean()
    cond_vol = data['Volume'] > 1.1 * data['Vol_20MA']  # relaxed threshold

    # --- Trend ---
    data['EMA_10'] = data['Close'].ewm(span=10).mean()
    data['EMA_20'] = data['Close'].ewm(span=20).mean()
    data['EMA_50'] = data['Close'].ewm(span=50).mean()
    cond_trend = (data['EMA_10'] > data['EMA_20'] * 0.99) & (data['Close'] > data['EMA_50'] * 0.97)

    # --- ATR ---
    tr = pd.concat([
        data['High'] - data['Low'],
        (data['High'] - data['Close'].shift()).abs(),
        (data['Low'] - data['Close'].shift()).abs()
    ], axis=1).max(axis=1)
    data['ATR_14'] = tr.rolling(14).mean()
    atr_ratio = data['ATR_14'] / data['Close']
    cond_volatility_ok = (atr_ratio > 0.008) & (atr_ratio < 0.08)  # widened

    # --- Adaptive rebound ---
    cond_low = (data['Low'].shift(1) >= data['Low_126min'] * 1.01) & \
               (data['Low'].shift(1) <= data['Low_126min'] * 1.12)  # widened band

    # --- Combine ---
    cond_buy = cond_low & cond_macd_turn & cond_rsi & cond_vol & cond_trend & cond_volatility_ok

    # --- Target & SL ---
    data['Buy_Signal'] = np.where(cond_buy, data['Close'].round(2), np.nan)
    data['StopLoss'] = 0 #data['Low_126min']
    #data['StopLoss'] = np.where(cond_buy, (data['Close'] - 1.8 * data['ATR_14']).round(2), np.nan)
    risk = data['Close'] - data['StopLoss']
    data['Target'] = np.where(cond_buy, (data['Close'] + 3 * risk).round(2), np.nan)
    data['Signal'] = cond_buy

    out_df = data[data['Date'] >= pd.to_datetime(start_date)]
    out_df.to_csv(f"{cvs_data_dir}/{symbol}.csv", index=False)
    print(f"✅ {symbol}: {int(out_df['Signal'].sum())} signals generated (balanced)")
def mark_signals_08NOV(symbol, start_date, end_date):
    file_path = f'{cvs_raw_data}/{symbol}.csv'
    if not os.path.exists(file_path):
        print(f"{symbol} Not found locally — downloading ...")
        ENCTOKEN = "mFm/EceEQnwbTy5IWDufMWv/jCZQryYGd9UI6SdYZoakQPRbgQ0WC35PMtygKxcUXr28ZOemMDlCRZC/oIG+MLuaDmldt6kBo9s8ImD0F9xJq3QG0CQenA=="
        data = fetch_kite_data(ENCTOKEN, symbol, start_date, end_date, interval='day')
        if data is None or data.empty:
            print(f"No data for {symbol}")
            return
        data.to_csv(file_path, index=False)
    else:
        print(f"{symbol} found locally and processing ...")
        data = pd.read_csv(file_path)

    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values('Date', inplace=True)
    data.reset_index(drop=True, inplace=True)

    # --- Indicators ---
    #data['Low_126min'] = data['Low'].shift(1).rolling(window=126, min_periods=50).min()
    data['Low_252min'] = data['Low'].shift(1).rolling(window=252, min_periods=50).min()

    # --- MACD ---
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD_Line'] = ema12 - ema26
    data['MACD_Signal'] = data['MACD_Line'].ewm(span=9, adjust=False).mean()
    data['MACD_Slope'] = data['MACD_Line'].diff()
    cond_macd_turn = (data['MACD_Slope'] > 0) & (data['MACD_Line'] < 0)

    # --- RSI ---
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    data['RSI'] = 100 - (100 / (1 + rs))
    cond_rsi = (data['RSI'] > 30) & (data['RSI'] < 60)

    # --- Volume ---
    data['Vol_20MA'] = data['Volume'].rolling(20).mean()
    cond_vol = data['Volume'] > 1.1 * data['Vol_20MA']

    # --- Trend ---
    data['EMA_10'] = data['Close'].ewm(span=10).mean()
    data['EMA_20'] = data['Close'].ewm(span=20).mean()
    data['EMA_50'] = data['Close'].ewm(span=50).mean()
    cond_trend = (data['EMA_10'] > data['EMA_20'] * 0.99) & (data['Close'] > data['EMA_50'] * 0.97)

    # --- ATR ---
    tr = pd.concat([
        data['High'] - data['Low'],
        (data['High'] - data['Close'].shift()).abs(),
        (data['Low'] - data['Close'].shift()).abs()
    ], axis=1).max(axis=1)
    data['ATR_14'] = tr.rolling(14).mean()
    atr_ratio = data['ATR_14'] / data['Close']
    cond_volatility_ok = (atr_ratio > 0.008) & (atr_ratio < 0.08)

    # --- Adaptive rebound ---
    cond_low = (data['Low'].shift(1) >= data['Low_252min'] * 1.01) & \
               (data['Low'].shift(1) <= data['Low_252min'] * 1.12)

    # --- Combine all buy conditions ---
    cond_buy = cond_low & cond_macd_turn & cond_rsi & cond_vol #& cond_trend & cond_volatility_ok

    # --- Buy, Target (no StopLoss) ---
    data['Buy_Signal'] = np.where(cond_buy, data['Close'].round(2), np.nan)
    data['StopLoss'] = 0.0  # no stop loss
    data['Target'] = (data['Close'] * 5).round(2) #np.where(cond_buy, (data['Close'] * 1.25).round(2), np.nan)  # +25% from buy price
    #data['Signal'] = cond_buy
    yesterday_low = data["Low"].shift(1)
    low_252 = data["Low"].shift(1).rolling(252, min_periods=1).min()
    tolerance = 0.05
    cond_close = data["Close"] > data["Low"].shift(1)
    cond_near_low = (yesterday_low >= low_252) & (yesterday_low <= low_252 * (1 + tolerance))

    data["Signal"] = cond_near_low & cond_close


    out_df = data[data['Date'] >= pd.to_datetime(start_date)]
    out_df.to_csv(f"{cvs_data_dir}/{symbol}.csv", index=False)
    print(f"✅ {symbol}: {int(out_df['Signal'].sum())} signals generated (Target +25%, no SL)")
def mark_signals(symbol, start_date, end_date):
    file_path = f'{cvs_raw_data}/{symbol}.csv'
    if not os.path.exists(file_path):
        print(f"{symbol} Not found locally — downloading ...")
        ENCTOKEN = "mFm/EceEQnwbTy5IWDufMWv/jCZQryYGd9UI6SdYZoakQPRbgQ0WC35PMtygKxcUXr28ZOemMDlCRZC/oIG+MLuaDmldt6kBo9s8ImD0F9xJq3QG0CQenA=="
        data = fetch_kite_data(ENCTOKEN, symbol, start_date, end_date, interval='day')
        if data is None or data.empty:
            print(f"No data for {symbol}")
            return
        data.to_csv(file_path, index=False)
    else:
        print(f"{symbol} found locally and processing ...")
        data = pd.read_csv(file_path)

    # --- Ensure enough candles ---
    if len(data) < 252:
        print(f"⚠️ {symbol}: Not enough data ({len(data)} candles). Skipping...")
        return

    # --- Preprocess ---
    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values('Date', inplace=True)
    data.reset_index(drop=True, inplace=True)

    # --- Indicators ---
    #data['Low_252min'] = data['Low'].shift(1).rolling(window=252, min_periods=252).min()
    data['Low_252min'] = data['Low'].shift(1).rolling(window=252, min_periods=252).min()

    # Define cutoff date

    # --- MACD ---
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD_Line'] = ema12 - ema26
    data['MACD_Signal'] = data['MACD_Line'].ewm(span=9, adjust=False).mean()
    data['MACD_Slope'] = data['MACD_Line'].diff()
    cond_macd_turn = (data['MACD_Slope'] > 0) & (data['MACD_Line'] < 0)

    data['StopLoss'] = 0.0
    data['Target'] = (data['Close'] * 5).round(2)

    yesterday_low = data["Low"].shift(1)
    low_252 = data["Low"].shift(1).rolling(252, min_periods=252).min()
    cond_close = data["Close"] > data["Low"].shift(1)
    # Calculate the lowest low of the past 5 days
    data['Low_5'] = data['Low'].rolling(window=5).min()
    # Condition: past 5 days lowest low near 252-day low
    tolerance = 0.01  # example: 1%
    cond_near_low = (data['Low_5'] >= data['Low_252min']) & (data['Low_5'] <= data['Low_252min'] * (1 + tolerance))
    #cond_near_low = (yesterday_low >= low_252) & (yesterday_low <= low_252 * (1 + tolerance))

    data["Signal"] = cond_near_low & cond_close & cond_macd_turn

    out_df = data[data['Date'] >= pd.to_datetime(start_date)]
    out_df.to_csv(f"{cvs_data_dir}/{symbol}.csv", index=False)
    print(f"✅ {symbol}: {int(out_df['Signal'].sum())} signals generated (Target +25%, no SL)")

def mark_signals_old(symbol, start_date, end_date):
    if not os.path.exists(f'{cvs_raw_data}/{symbol}.csv'):
        # print(not os.path.exists(f'{cvs_raw_data}/{stock}.csv'))
        #print(f"{symbol} Not found in local, downloading from online and processing it ...")
        # Fetch data from Yahoo Finance
        #data = fetch_yahoo_finance_data(symbol + '.NS', start_date, end_date)
        print(f"{symbol} Not found locally — downloading ...")
        ENCTOKEN = "ciejrdDMeGXHRLX0y2jRLjEsPndYUV2uzo5jOufr4yZuFfKabpDTfB70ieFFAXRUlowwoLN+Fj7UVs9XjyR8QP1fA0baVX9plepBzCL5Axef+fQ98xzJCA=="
        data = fetch_kite_data(ENCTOKEN, symbol, start_date, end_date, interval='day')
        # fetch_weekly_data(ticker, start_date, end_date)
        if data.empty:
            print(f"No data found in yfinance for {symbol}. Skipping...")
            return 0, 0  # Skip this stock and return 0 charges and 0 trades
        data.to_csv(f"{cvs_raw_data}/{symbol}.csv")
        data = pd.read_csv(f'{cvs_raw_data}/{symbol}.csv')
        print('---')
    else:
        print(f"{symbol} found in local and processing it ...")
        data = pd.read_csv(f'{cvs_raw_data}/{symbol}.csv')
        print('---')
    ###
    for i in range(100, len(data)):  # Start from the 100th day to have enough data for calculations
        is_previous_green = (data.iloc[i - 1]['Close'] > data.iloc[i - 1]['Open'])
        is_green = (data.iloc[i]['Close'] > data.iloc[i]['Open'])
        ##########
        is_open_below_yday_close = data.iloc[i]['Open'] < data.iloc[i - 1]['Close']
        is_tday_high_break_yday_high = data.iloc[i]['High'] > data.iloc[i - 1]['High']
        #####
        buy_today_cond = is_tday_high_break_yday_high and is_open_below_yday_close
        condition = is_previous_green and check_cci_condition_close(data, i - 2)

        rsi_bull = pd_rsi_above_n(data, i, 14, 55) and pd_rsi_below_n(data, i, 14, 65)
        # if yday_44EMA_above_100EMA and past_2_below_100EMA and is_green:#check_cci_condition_close(data, i - 1):
        data = calculate_macd(data)

        # Compute indicators outside the loop
        data['Low_252min'] = data['Low'].shift(1).rolling(window=252).min()

        # 20-day average volume
        data['Vol_20MA'] = data['Volume'].rolling(window=20).mean()

        # Volume confirmation column (next day’s volume > 20-day average)
        data['Cond_VolumeConfirm'] = data['Volume'].shift(-1) > data['Vol_20MA']

        # --- MACD (12, 26, 9) Calculation ---
        data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD_Line'] = data['EMA12'] - data['EMA26']
        data['MACD_Signal'] = data['MACD_Line'].ewm(span=9, adjust=False).mean()

        # --- MACD crossover condition (Series-based) ---
        data['MACD_Cross'] = (data['MACD_Line'] > data['MACD_Signal']) & (
                data['MACD_Line'].shift(1) <= data['MACD_Signal'].shift(1)
        )

        condition = (
                (data.iloc[i - 1]['Low'] >= data.iloc[i]['Low_252min'] * 1.02) &
                (data.iloc[i - 1]['Low'] <= data.iloc[i]['Low_252min'] * 1.05)
                #(data.iloc[i]['Close'] > data.iloc[i - 1]['Close']) &
                #(data.iloc[i]['Cond_VolumeConfirm'])
                #& (data.iloc[i]['MACD_Cross'])  # <- Clean and aligned
        )
        #
        yesterday_low = data["Low"].shift(1)
        low_252 = data["Low"].shift(1).rolling(252, min_periods=1).min()
        tolerance = 0.05
        cond_close = data["Close"] > data["Low"].shift(1)
        cond_near_low = (yesterday_low >= low_252) & (yesterday_low <= low_252 * (1 + tolerance))

        data["Signal"] = cond_near_low & cond_close

        if cond_near_low.iloc[i] and cond_close.iloc[i]:
            bought_price = round_to_nearest_0_05(data.iloc[i]['Close'])
            # stop loss: 5% below 252-day low
            stop_loss = 10 #round_to_nearest_0_05(data.iloc[i]['Low_252min'])
            # risk per share
            risk = bought_price - stop_loss
            # target: 3x reward-to-risk ratio
            #target = bought_price + (3 * risk)
            target = bought_price + (0.25 * bought_price)
            data.loc[data.index[i], 'Buy_Signal'] = bought_price
            data.loc[data.index[i], 'Target'] = target
            data.loc[data.index[i], 'StopLoss'] = stop_loss

    data = data.loc[data['Date'] >= from_date]
    data = convert_all_col_digit(data)
    data.to_csv(f"{cvs_data_dir}/{symbol}.csv", index=False)





def calculate_macd(df, fast=12, slow=26, signal=9):
    df['EMA_fast'] = df['Close'].ewm(span=fast, adjust=False).mean()
    df['EMA_slow'] = df['Close'].ewm(span=slow, adjust=False).mean()
    df['MACD_Line'] = df['EMA_fast'] - df['EMA_slow']
    df['MACD_Signal'] = df['MACD_Line'].ewm(span=signal, adjust=False).mean()
    return df