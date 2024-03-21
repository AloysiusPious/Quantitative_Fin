import yfinance as yf
def get_df_from_yf(symbol, start_date, end_date):
    #df = yf.download(symbol, start=start_date, end=end_date)
    yf_data = yf.download(symbol + '.NS', start='2019-01-01', end='2024-01-31')
    # Convert to pandas DataFrame
    df = yf_data.copy()
    return df