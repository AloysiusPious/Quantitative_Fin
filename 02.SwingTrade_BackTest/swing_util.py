import pandas as pd

def calculate_ichimoku(df, period9=9, period26=26, period52=52):
    high_9 = df['High'].rolling(window=period9).max()
    low_9 = df['Low'].rolling(window=period9).min()
    high_26 = df['High'].rolling(window=period26).max()
    low_26 = df['Low'].rolling(window=period26).min()
    high_52 = df['High'].rolling(window=period52).max()
    low_52 = df['Low'].rolling(window=period52).min()

    df['Tenkan-sen'] = (high_9 + low_9) / 2
    df['Kijun-sen'] = (high_26 + low_26) / 2
    df['Senkou Span A'] = ((df['Tenkan-sen'] + df['Kijun-sen']) / 2).shift(period26)
    df['Senkou Span B'] = ((high_52 + low_52) / 2).shift(period26)

    return df

def calculate_atr(df, window=14):
    df['High-Low'] = df['High'] - df['Low']
    df['High-Close-Prev'] = (df['High'] - df['Close'].shift()).abs()
    df['Low-Close-Prev'] = (df['Low'] - df['Close'].shift()).abs()
    df['True Range'] = df[['High-Low', 'High-Close-Prev', 'Low-Close-Prev']].max(axis=1)
    df['ATR'] = df['True Range'].rolling(window=window).mean()
    return df

def check_conditions(data, i):
    if i < 2 or i >= len(data):
        return False

    data = calculate_ichimoku(data)
    data = calculate_atr(data)

    # Conditions
    cond1 = data['Close'].iloc[i-2] <= max(250, data['High'].iloc[i-1]) * 0.9
    cond2 = data['Close'].iloc[i-1] > data['Senkou Span A'].iloc[i-1]
    cond3 = data['Volume'].rolling(window=3).mean().iloc[i-1] * data['Close'].iloc[i-1] > 100000000
    cond4 = data['Close'].iloc[i-1] >= max(data['High'].iloc[i-2:i+1].max(), 7)
    cond5 = data['Close'].iloc[i-2] <= min(data['Low'].iloc[i-2:i+1].min(), 7) * 1.1
    cond6 = data['Close'].iloc[i-2] * (data['Close'].iloc[i-1] - data['Close'].iloc[i-2]) / data['Close'].iloc[i-2] / 100 > data['ATR'].iloc[i-1]
    cond7 = data['Open'].iloc[i] < data['Close'].iloc[i-1]
    cond8 = data['High'].iloc[i] > data['High'].iloc[i-1]

    return cond1 and cond2 and cond3 and cond4 and cond5 and cond6 and cond7 and cond8

# Example usage
data = pd.read_csv('path/to/your/data.csv')
# Ensure your DataFrame has 'Open', 'High', 'Low', 'Close', and 'Volume' columns

# Example: Check conditions for a specific index
result = check_conditions(data, len(data) - 1)
print(result)