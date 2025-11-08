# strategy_backtest.py
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt

# -------------------------
# 1) Indicators (modular)
# -------------------------
def compute_indicators(df):
    """
    Input: df with columns Date, Open, High, Low, Close, Volume
    Output: df augmented with indicators
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)

    # Moving averages
    df['SMA_200'] = df['Close'].rolling(200).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()

    # Your 252-day low (shifted 1 day)
    df['Low_252min'] = df['Low'].shift(1).rolling(window=252, min_periods=50).min()

    # Volume dynamics
    df['Vol_EMA5'] = df['Volume'].ewm(span=5, adjust=False).mean()
    df['Vol_EMA20'] = df['Volume'].ewm(span=20, adjust=False).mean()
    df['Vol_Spike'] = (df['Volume'] > df['Vol_EMA20'] * 1.5)  # spike
    df['Vol_Accel'] = df['Vol_EMA5'] > df['Vol_EMA5'].shift(5)  # accelerating volume

    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Line'] = (ema12 - ema26)
    df['MACD_Signal'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
    df['MACD_Cross_Up'] = (df['MACD_Line'] > df['MACD_Signal']) & (df['MACD_Line'].shift(1) <= df['MACD_Signal'].shift(1))

    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14, min_periods=14).mean()
    avg_loss = loss.rolling(14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # ATR (volatility)
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(14, min_periods=7).mean()

    # Relative strength vs index (optional, user must pass index df externally)
    # df['RS'] = df['Close'] / index_df['Close']  # compute outside if needed

    return df.dropna()

# -------------------------
# 2) Signal generator
# -------------------------
def generate_signals(df,
                     macd_signal_thresh=-10,
                     rsi_low=30, rsi_high=50):
    """
    Returns a DataFrame with boolean 'entry_signal' column.
    Logic:
      - cond_low: 1d prior low within 1.02–1.05 of 252-day low (your base)
      - cond_macd: MACD_Signal <= macd_signal_thresh OR MACD bullish crossover
      - cond_rsi: RSI in (rsi_low, rsi_high)
      - cond_vol: volume acceleration OR volume spike
      - trend filter: price > SMA_200 OR EMA_10 > EMA_50 breakout (momentum)
    """
    df = df.copy()
    cond_low = (df['Low'].shift(1) >= df['Low_252min'] * 1.02) & (df['Low'].shift(1) <= df['Low_252min'] * 1.05)
    cond_macd_deep = df['MACD_Signal'] <= macd_signal_thresh
    cond_macd_cross = df['MACD_Cross_Up']
    cond_macd = cond_macd_deep | cond_macd_cross

    cond_rsi = (df['RSI'] > rsi_low) & (df['RSI'] < rsi_high)
    cond_vol = df['Vol_Accel'] | df['Vol_Spike']
    cond_trend = (df['Close'] > df['SMA_200']) | (df['EMA_10'] > df['EMA_50'])

    # Final entry: all must be true; MACD flexible (deep negative OR crossover)
    df['entry_signal'] = cond_low & cond_macd & cond_rsi & cond_vol & cond_trend

    # Optional: penalize very-low-volume days (to avoid illiquid)
    df['entry_signal'] = df['entry_signal'] & (df['Volume'] > 5000)  # tweak threshold

    return df

# -------------------------
# 3) Position sizing & backtest engine (daily)
# -------------------------
def backtest_strategy(df, initial_capital=100000.0, risk_per_trade=0.01,
                      tp_multiplier=2.0, partial_exit_pct=0.5, atr_trail_mult=2.0):
    """
    Simulates orders:
      - Enter at next day's Open after signal.
      - Stoploss = entry_price - ATR_14 * 2
      - Partial exit at entry + tp_multiplier * risk (take partial_exit_pct)
      - Trail remaining with ATR-based trailing stop (atr_trail_mult * ATR)
    Returns: trades list, equity curve, metrics
    """
    df = df.copy()
    equity = initial_capital
    cash = initial_capital
    pos = 0.0  # number of shares
    entry_price = None
    entry_idx = None
    stop_price = None
    target_price = None
    remaining_shares = 0.0
    trade_log = []

    equity_curve = []

    for i in range(len(df)-1):
        today = df.index[i]
        tomorrow = df.index[i+1]  # we will enter at tomorrow's open if signal today

        # Record equity
        market_value = pos * df['Close'].iloc[i]
        equity_curve.append({'Date': today, 'Equity': cash + market_value})

        # If no position and entry_signal today -> enter tomorrow open
        if pos == 0 and df['entry_signal'].iloc[i]:
            open_price = df['Open'].iloc[i+1]  # enter at next day open
            atr = df['ATR_14'].iloc[i+1] if not pd.isna(df['ATR_14'].iloc[i+1]) else df['ATR_14'].iloc[i]
            if atr <= 0 or np.isnan(atr):
                continue

            # define stop and risk per share
            stop_price = open_price - atr_trail_mult * atr
            if stop_price <= 0 or stop_price >= open_price:
                continue
            risk_per_share = open_price - stop_price
            # position sizing
            position_risk_amount = equity * risk_per_trade
            shares = int(position_risk_amount / risk_per_share)
            if shares <= 0:
                continue

            # allocate
            cost = shares * open_price
            if cost > cash:
                # fractional reduce
                shares = int(cash / open_price)
                if shares <= 0:
                    continue
                cost = shares * open_price

            pos = shares
            remaining_shares = shares
            cash -= cost
            entry_price = open_price
            entry_idx = i+1
            target_price = entry_price + tp_multiplier * risk_per_share

            trade_log.append({
                'entry_date': df.index[entry_idx],
                'entry_price': entry_price,
                'shares': shares,
                'initial_stop': stop_price,
                'target_price': target_price
            })

        # If position exists, evaluate exits at today's OHLC
        if pos > 0:
            high = df['High'].iloc[i]
            low = df['Low'].iloc[i]
            close = df['Close'].iloc[i]

            # 1) Check if target (partial) hit during the day
            if remaining_shares > 0 and high >= target_price:
                # sell partial at target_price
                sell_shares = int(np.ceil(partial_exit_pct * remaining_shares))
                proceeds = sell_shares * target_price
                cash += proceeds
                pos -= sell_shares
                remaining_shares -= sell_shares
                # update trade log
                trade_log[-1].update({'partial_exit_date': df.index[i], 'partial_exit_price': target_price})

            # 2) Check stoploss hit
            # Use intra-day low to check stop
            if low <= stop_price:
                # stop executed at stop_price
                proceeds = pos * stop_price
                cash += proceeds
                trade_log[-1].update({'exit_date': df.index[i], 'exit_price': stop_price, 'exit_reason': 'stop'})
                pos = 0
                remaining_shares = 0
                entry_price = None
                stop_price = None
                target_price = None
                continue

            # 3) Update trailing stop (based on current ATR)
            current_atr = df['ATR_14'].iloc[i]
            new_trail = close - atr_trail_mult * current_atr
            # Only move stop up, never down
            if new_trail > stop_price:
                stop_price = new_trail

            # 4) If last day and still in position, close at close (end of data)
            if i == len(df)-2 and pos > 0:
                proceeds = pos * close
                cash += proceeds
                trade_log[-1].update({'exit_date': df.index[i], 'exit_price': close, 'exit_reason': 'eod'})
                pos = 0
                remaining_shares = 0
                entry_price = None
                stop_price = None
                target_price = None

    # final equity snapshot
    last_price = df['Close'].iloc[-1]
    equity_curve.append({'Date': df.index[-1], 'Equity': cash + pos * last_price})
    eqdf = pd.DataFrame(equity_curve).set_index('Date').ffill()

    # compute daily returns from equity curve
    eqdf['returns'] = eqdf['Equity'].pct_change().fillna(0)
    # metrics
    total_days = (eqdf.index[-1] - eqdf.index[0]).days
    years = total_days / 365.25 if total_days > 0 else 1.0
    total_return = (eqdf['Equity'].iloc[-1] / eqdf['Equity'].iloc[0]) - 1
    ann_return = (1 + total_return) ** (1/years) - 1
    daily_ret = eqdf['returns']
    sharpe = (daily_ret.mean() / (daily_ret.std() + 1e-9)) * np.sqrt(252)
    # max drawdown
    peak = eqdf['Equity'].cummax()
    dd = (eqdf['Equity'] / peak) - 1
    max_dd = dd.min()

    # trades per year
    n_trades = len([t for t in trade_log if 'entry_date' in t])
    trades_per_year = n_trades / years if years>0 else n_trades

    metrics = {
        'initial_capital': initial_capital,
        'ending_capital': eqdf['Equity'].iloc[-1],
        'total_return': total_return,
        'annual_return': ann_return,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'n_trades': n_trades,
        'trades_per_year': trades_per_year
    }

    return trade_log, eqdf, metrics

# -------------------------
# 4) Plotting / diagnostics
# -------------------------
def plot_equity(eqdf):
    plt.figure(figsize=(10,5))
    plt.plot(eqdf.index, eqdf['Equity'], label='Equity Curve')
    plt.title('Equity Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_signals_price(df):
    plt.figure(figsize=(12,6))
    plt.plot(df.index, df['Close'], label='Close')
    entries = df[df['entry_signal']]
    plt.scatter(entries.index, entries['Close'], marker='^', color='green', s=60, label='Entry')
    plt.plot(df.index, df['SMA_200'], label='SMA200', alpha=0.6)
    plt.plot(df.index, df['EMA_50'], label='EMA50', alpha=0.6)
    plt.title('Price + Entry Signals')
    plt.legend()
    plt.grid(True)
    plt.show()

# -------------------------
# 5) Usage example
# -------------------------
if __name__ == '__main__':
    # load csv created by fetch_kite_data or any OHLCV csv
    df = pd.read_csv('AXISBANK.csv')  # replace with your file
    df = compute_indicators(df)
    df = generate_signals(df)
    trades, eqdf, metrics = backtest_strategy(df)
    print("Backtest metrics:", metrics)
    plot_equity(eqdf)
    plot_signals_price(df)
