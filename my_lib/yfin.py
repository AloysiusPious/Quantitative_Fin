import yfinance as yf
def get_df_from_yf_1(symbol, start_date, end_date):
    #df = yf.download(symbol, start=start_date, end=end_date)
    yf_data = yf.download(symbol + '.NS', start_date, end_date)
    # Convert to pandas DataFrame
    df = yf_data.copy()
    return df
def get_df_from_yf(symbol, start_date, end_date):
    try:
        yf_data = yf.download(symbol + '.NS', start=start_date, end=end_date)
        df = yf_data.copy()
        return df
    except Exception as e:
        print(f"Error downloading data for symbol {symbol}: {str(e)}")
        return None