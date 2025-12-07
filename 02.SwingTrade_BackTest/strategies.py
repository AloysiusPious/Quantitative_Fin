import os
import pandas as pd
import numpy as np
import talib

def fibonacci(enctoken, symbol, start_date, end_date):
    file_path = f'{cvs_raw_data}/{symbol}.csv'

    # ---------------- Load or Download ----------------
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

    # ---------------- Indicators ----------------
    # EMAs (trend)
    data['EMA20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['EMA50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['Trend_Up'] = data['EMA20'] > data['EMA50']
    data['Trend_Down'] = data['EMA20'] < data['EMA50']

    # Swing high / low (for dynamic fib levels)
    swing_lookback = 20
    data['Swing_High'] = data['High'].rolling(swing_lookback).max()
    data['Swing_Low'] = data['Low'].rolling(swing_lookback).min()

    # Fibonacci levels (based on last swing high/low)
    diff = data['Swing_High'] - data['Swing_Low']
    data['Fib_382'] = data['Swing_Low'] + diff * 0.382
    data['Fib_500'] = data['Swing_Low'] + diff * 0.500
    data['Fib_618'] = data['Swing_Low'] + diff * 0.618

    # Bollinger bands (optional visual / squeeze use if needed later)
    data['BB_Mid'] = data['Close'].rolling(20).mean()
    data['BB_Std'] = data['Close'].rolling(20).std()
    data['BB_Upper'] = data['BB_Mid'] + 2 * data['BB_Std']
    data['BB_Lower'] = data['BB_Mid'] - 2 * data['BB_Std']

    # Candle definitions
    data['BullCandle'] = data['Close'] > data['Open']
    data['BearCandle'] = data['Close'] < data['Open']

    # Volume SMA and ATR
    data['Vol20'] = data['Volume'].rolling(20).mean()
    data['ATR'] = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)

    # ---------------- Fibonacci Strategy Conditions ----------------
    # BUY: in uptrend, price pulls to 38.2-61.8 zone and shows bullish confirmation
    cond_fib_pull = (data['Low'] <= data['Fib_618']) & (data['High'] >= data['Fib_382'])
    cond_fib_zone = (data['Low'] >= data['Fib_382']) & (data['Low'] <= data['Fib_618'])  # price touches within zone
    cond_bull_conf = data['BullCandle']
    cond_vol_confirm = data['Volume'] > (1.0 * data['Vol20'])  # require at least normal volume; change multiplier if needed
    cond_trend_up = data['Trend_Up']

    cond_fib_buy = cond_trend_up & cond_fib_zone & cond_bull_conf & cond_vol_confirm

    # SELL (short / exit long): inverse - in downtrend, retrace up into fib zone and bearish confirmation
    cond_trend_down = data['Trend_Down']
    cond_fib_zone_up = (data['High'] <= data['Fib_618']) & (data['High'] >= data['Fib_382'])
    cond_bear_conf = data['BearCandle']
    cond_fib_sell = cond_trend_down & cond_fib_zone_up & cond_bear_conf & cond_vol_confirm

    # ---------------- Signals, StopLoss and Targets ----------------
    # Initialize columns
    data['Buy_Signal'] = np.nan
    data['Sell_Signal'] = np.nan
    data['StopLoss'] = np.nan
    data['Target'] = np.nan

    # BUY handling
    # StopLoss: swing low minus buffer (use ATR). SL = Swing_Low - 1.0 * ATR (rounded)
    # Target: 2R (reward = 2 * risk)
    buy_idx = cond_fib_buy[cond_fib_buy].index
    for idx in buy_idx:
        buy_price = data.at[idx, 'Close']
        swing_low = data.at[idx, 'Swing_Low']
        atr = data.at[idx, 'ATR'] if not np.isnan(data.at[idx, 'ATR']) else 0.0

        sl = (swing_low - 1.0 * atr) if (not np.isnan(swing_low)) else (buy_price - 1.5 * atr)
        # ensure SL is below buy price
        if sl >= buy_price:
            sl = buy_price - 1.5 * atr

        risk = buy_price - sl
        tp = buy_price + 2.0 * risk  # 2R

        data.at[idx, 'Buy_Signal'] = round(buy_price, 2)
        data.at[idx, 'StopLoss'] = round(sl, 2)
        data.at[idx, 'Target'] = round(tp, 2)

    # SELL handling (optional short signals or long-exit signals)
    sell_idx = cond_fib_sell[cond_fib_sell].index
    for idx in sell_idx:
        sell_price = data.at[idx, 'Close']
        swing_high = data.at[idx, 'Swing_High']
        atr = data.at[idx, 'ATR'] if not np.isnan(data.at[idx, 'ATR']) else 0.0

        sl = (swing_high + 1.0 * atr) if (not np.isnan(swing_high)) else (sell_price + 1.5 * atr)
        # ensure SL is above sell price
        if sl <= sell_price:
            sl = sell_price + 1.5 * atr

        risk = sl - sell_price
        tp = sell_price - 2.0 * risk  # 2R for shorts

        data.at[idx, 'Sell_Signal'] = round(sell_price, 2)
        # For consistency store StopLoss/Target for sell rows too (StopLoss is the protective level above price)
        data.at[idx, 'StopLoss'] = round(sl, 2)
        data.at[idx, 'Target'] = round(tp, 2)

    # ---------------- Save ----------------
    data = data.loc[data['Date'] >= start_date]
    data.to_csv(f"{cvs_data_dir}/{symbol}.csv", index=False)

    print(f"✅ Processed {symbol} with Fibonacci strategy (EMA20/50 trend + Fib 0.382-0.618 entries + ATR SL + 2R TP)")

def bollinger_bands(enctoken, symbol, start_date, end_date):
    file_path = f'{cvs_raw_data}/{symbol}.csv'

    # ---------------- Load or Download ----------------
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

    # ---------------- Indicators ----------------
    # Trend filter
    data['EMA200'] = data['Close'].ewm(span=200, adjust=False).mean()
    cond_trend = data['Close'] > data['EMA200']

    # RSI > 50
    data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
    cond_rsi = data['RSI'] > 50

    # Bollinger Bands (20, 2)
    data['BB_Mid'] = data['Close'].rolling(20).mean()
    data['BB_Std'] = data['Close'].rolling(20).std()
    data['BB_Upper'] = data['BB_Mid'] + 2 * data['BB_Std']
    data['BB_Lower'] = data['BB_Mid'] - 2 * data['BB_Std']

    # BB width = (Upper-Lower)/Middle
    data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Mid']

    # Squeeze: BB width < 20-bar volatility quantile
    bb_quantile = data['BB_Width'].rolling(20).quantile(0.20)
    cond_squeeze = data['BB_Width'] < bb_quantile

    # Strong candle (Body > 50% of Range)
    data['Body'] = (data['Close'] - data['Open']).abs()
    data['Range'] = (data['High'] - data['Low'])
    cond_body = data['Body'] > (0.50 * data['Range'])

    # Volume spike
    data['Vol20'] = data['Volume'].rolling(20).mean()
    cond_volume = data['Volume'] > (1.20 * data['Vol20'])

    # Breakout above Bollinger Upper band
    cond_breakout = data['Close'] > data['BB_Upper']

    # ---------------- Final Buy Condition ----------------
    cond_buy = cond_trend & cond_rsi & cond_squeeze & cond_breakout & cond_volume & cond_body

    data.loc[cond_buy, 'Buy_Signal'] = data['Close']

    # ---------------- ATR StopLoss + Target ----------------
    data['ATR'] = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)

    # SL = Close - 1.5 × ATR
    data.loc[cond_buy, 'StopLoss'] = (data['Close'] - 1.5 * data['ATR']).round(2)

    # TP = SL + 2 × (Close - SL)   →   2R reward
    data.loc[cond_buy, 'Target'] = (
        data['StopLoss'] + 2 * (data['Close'] - data['StopLoss'])
    ).round(2)

    # ---------------- Save ----------------
    data = data.loc[data['Date'] >= start_date]
    data.to_csv(f"{cvs_data_dir}/{symbol}.csv", index=False)

    print(f"✅ Processed {symbol} with Squeeze Breakout Strategy (BB + RSI + EMA200 + Volume + ATR)")

def uptrend_swing_low(enctoken, symbol, start_date, end_date):
    file_path = f'{cvs_raw_data}/{symbol}.csv'

    # ---------------- Load or Download ----------------
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

    # ---------------- Indicators ----------------
    # EMAs
    data['EMA200'] = data['Close'].ewm(span=200, adjust=False).mean()
    data['EMA100'] = data['Close'].ewm(span=100, adjust=False).mean()
    data['EMA50']  = data['Close'].ewm(span=50,  adjust=False).mean()

    # 3+ Month Low → 63 days minimum low
    data['Low_63'] = data['Low'].rolling(63).min()
    cond_3month_low = data['Close'] <= data['Low_63'].shift(1)

    # Green candle
    cond_green = data['Close'] > data['Open']

    # Close above EMA200
    cond_above_200 = data['Close'] > data['EMA200']

    # Trend confirmation
    cond_trend = (data['EMA50'] > data['EMA200']) & (data['EMA100'] > data['EMA200'])

    # Swing low for SL (default 20 bar)
    swing_lookback = 20
    data['Swing_Low'] = data['Low'].rolling(swing_lookback).min()

    # ---------------- Final BUY Condition ----------------
    cond_buy = cond_3month_low & cond_green & cond_above_200 & cond_trend

    # ---------------- Buy, StopLoss, Target ----------------
    data['Buy_Signal'] = np.nan
    data['StopLoss'] = np.nan
    data['Target'] = np.nan

    buy_idx = data[cond_buy].index
    for idx in buy_idx:
        buy_price = data.at[idx, 'Close']
        sl = data.at[idx, 'Swing_Low']

        # fallback safety in case swing low = NaN
        if np.isnan(sl):
            sl = buy_price * 0.97  # default 3% SL fallback

        # Ensure stoploss is below buy price
        if sl >= buy_price:
            sl = buy_price * 0.97

        risk = buy_price - sl
        tp = buy_price + 3 * risk  # 3R target

        data.at[idx, 'Buy_Signal'] = round(buy_price, 2)
        data.at[idx, 'StopLoss'] = round(sl, 2)
        data.at[idx, 'Target'] = round(tp, 2)

    # ---------------- Save ----------------
    data = data.loc[data['Date'] >= start_date]
    data.to_csv(f"{cvs_data_dir}/{symbol}.csv", index=False)

    print(f"✅ Processed {symbol} with 3-month-low + EMA trend + green candle + 3R target strategy")

def mark_signals(enctoken, symbol, start_date, end_date):
    import numpy as np
    import pandas as pd
    import talib
    import os

    file_path = f'{cvs_raw_data}/{symbol}.csv'

    # ---------------- Load or Download ----------------
    if not os.path.exists(file_path):
        print(f"{symbol} Not found locally — downloading ...")
        data = fetch_kite_data(enctoken, symbol, start_date, end_date, interval='day')
        if data is None or data.empty:
            return
        data.to_csv(file_path, index=False)
    else:
        data = pd.read_csv(file_path)

    # ---------------- Indicators ----------------
    data['EMA200'] = data['Close'].ewm(span=200, adjust=False).mean()
    data['EMA100'] = data['Close'].ewm(span=100, adjust=False).mean()
    data['EMA50']  = data['Close'].ewm(span=50, adjust=False).mean()
    data['ATR'] = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)
    data['RSI'] = talib.RSI(data['Close'], 14)
    data['MACD'], data['MACD_signal'], _ = talib.MACD(data['Close'], 12, 26, 9)

    # ---------------- Trend + Momentum Filter ----------------
    cond_trend = (data['EMA50'] > data['EMA200']) & (data['EMA100'] > data['EMA200'])
    cond_rsi = (data['RSI'] > 50) & (data['RSI'] < 70)
    cond_macd = data['MACD'] > data['MACD_signal']

    # ---------------- Pullback Filter ----------------
    data['Low_42'] = data['Low'].rolling(42).min()  # last 2 months ~ 42 trading days
    cond_pullback = (data['Close'] <= data['Low_42']*1.02)  # within 2% above recent low

    # ---------------- Candle Confirmation ----------------
    cond_bullish_candle = (data['Close'] > data['Open'])
    data['Vol20'] = data['Volume'].rolling(20).mean()
    cond_vol = data['Volume'] > (1.2 * data['Vol20'])
    cond_candle_confirm = (data['Close'] > data['High'].shift(1)) | \
                          ((data['Close'] - data['Open']) > (data['Open'].shift(1) - data['Close'].shift(1)))

    # ---------------- Combine Entry Conditions ----------------
    cond_buy = cond_trend & cond_rsi & cond_macd & cond_pullback & cond_bullish_candle & cond_vol & cond_candle_confirm

    # ---------------- Initialize Columns ----------------
    data['Buy_Signal'] = np.nan
    data['StopLoss'] = np.nan
    data['Target1'] = np.nan
    data['Target'] = np.nan
    data['RiskPct'] = np.nan

    # ---------------- Calculate SL & Targets ----------------
    swing_lookback = 20
    data['Swing_Low'] = data['Low'].rolling(swing_lookback).min()

    for idx in data[cond_buy].index:
        buy = data.at[idx, 'Close']
        swing_low = data.at[idx, 'Swing_Low']
        atr = data.at[idx, 'ATR']

        # SL slightly below swing low or 1.5*ATR
        if not np.isnan(swing_low):
            sl = swing_low - 0.25*atr
        else:
            sl = buy - 1.5*atr

        if sl >= buy:
            sl = buy - 1.5*atr

        risk = buy - sl
        risk_pct = (risk / buy)*100 if buy>0 else np.nan
        if risk_pct > 8:  # skip trades with excessive risk
            continue

        # Targets
        tp1 = buy + 1.5*risk
        tp = buy + 3*risk

        # Assign
        data.at[idx, 'Buy_Signal'] = round(buy,2)
        data.at[idx, 'StopLoss'] = round(sl,2)
        data.at[idx, 'Target1'] = round(tp1,2)
        data.at[idx, 'Target'] = round(tp,2)
        data.at[idx, 'RiskPct'] = round(risk_pct,2)

    # ---------------- Save CSV ----------------
    data = data.loc[data['Date'] >= start_date]
    data.to_csv(f"{cvs_data_dir}/{symbol}.csv", index=False)

    print(f"✅ Processed {symbol} — refined high-probability swing strategy")

def mark_signals(enctoken, symbol, start_date, end_date):
    file_path = f'{cvs_raw_data}/{symbol}.csv'

    # ---------------- Load or Download ----------------
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

    # ---------------- Indicators ----------------
    # EMAs
    data['EMA20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['EMA50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['EMA100'] = data['Close'].ewm(span=100, adjust=False).mean()
    data['EMA200'] = data['Close'].ewm(span=200, adjust=False).mean()

    # ATR
    data['H-L'] = data['High'] - data['Low']
    data['H-PC'] = (data['High'] - data['Close'].shift(1)).abs()
    data['L-PC'] = (data['Low'] - data['Close'].shift(1)).abs()
    data['TR'] = data[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    data['ATR'] = data['TR'].rolling(14).mean()

    # RSI
    data['RSI'] = talib.RSI(data['Close'], timeperiod=14)

    # ---------------- Conditions ----------------

    # A) Current EMA bullish alignment
    cond_ema_now = (
        (data['Close'] > data['EMA20']) &
        (data['EMA20'] > data['EMA50']) &
        (data['EMA50'] > data['EMA100']) &
        (data['EMA100'] > data['EMA200'])
    )
    # Ensure boolean dtype
    #cond_ema_now = cond_ema_now.astype(bool)
    # 2. Recent crossover event (switch from False → True)
    crossover_event = cond_ema_now & (~cond_ema_now.shift(1).fillna(False))
    # Check that no crossover happened in the last 40 bars
    no_recent_cross = ~cond_ema_now.shift(40).fillna(False)

    # B) Fresh EMA alignment — became true in last 40 days
    cond_ema_recent = (
        no_recent_cross &
        cond_ema_now
    )

    # C) First pullback into EMA20 zone (within 1%)
    cond_pullback = data['Low'] <= data['EMA20'] * 1.01

    # D) Bullish candle confirmation
    cond_bullish = data['Close'] > data['Open']

    # E) Volume confirmation (optional but improves win-rate)
    data['Vol20'] = data['Volume'].rolling(20).mean()
    cond_volume = data['Volume'] > 1.2 * data['Vol20']

    # ---------------- Final Buy Condition ----------------
    cond_buy = cond_ema_recent & cond_pullback & cond_bullish & cond_volume

    # ---------------- Buy, StopLoss, Target ----------------
    data.loc[cond_buy, 'Buy_Signal'] = data['Close']

    # Stop-loss = Swing low minus 1× ATR
    data['Swing_Low'] = data['Low'].rolling(10).min()
    data.loc[cond_buy, 'StopLoss'] = (data['Swing_Low'] - data['ATR']).round(2)

    # Target = 3R (3 times reward of risk)
    data.loc[cond_buy, 'Target'] = (
        data['Buy_Signal'] + 3 * (data['Buy_Signal'] - data['StopLoss'])
    ).round(2)

    # ---------------- Save ----------------
    data = data.loc[data['Date'] >= start_date]
    data.to_csv(f"{cvs_data_dir}/{symbol}.csv", index=False)

    print(f"✅ Processed {symbol} with Fresh EMA Alignment + EMA20 Pullback Strategy")

