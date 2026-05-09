###############################################################################################
# Buy price
data.loc[cond_buy, 'Buy_Signal'] = data.loc[cond_buy, 'Close']
# Structure-based StopLoss
data['Recent_Low'] = data['Low'].rolling(5).min().shift(1)
data.loc[cond_buy, 'StopLoss'] = data.loc[cond_buy, 'Recent_Low']
# Risk calculation
data.loc[cond_buy, 'Risk'] = (data.loc[cond_buy, 'Buy_Signal'] - data.loc[cond_buy, 'StopLoss'])
# Risk-Reward based Target
RR = 1.8
data.loc[cond_buy, 'Target'] = (data.loc[cond_buy, 'Buy_Signal'] + RR * data.loc[cond_buy, 'Risk'])
# SL too wide
data.loc[data['Risk'] / data['Buy_Signal'] > 0.06, 'Buy_Signal'] = np.nan
# SL above entry (invalid)
data.loc[data['StopLoss'] >= data['Buy_Signal'], ['Buy_Signal', 'StopLoss', 'Target']] = np.nan
###############################################################################################