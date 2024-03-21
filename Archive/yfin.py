import yfinance as yf
def get_df_from_yf(symbol, start_date, end_date):
    df = yf.download(symbol, start=start_date, end=end_date)
    return df