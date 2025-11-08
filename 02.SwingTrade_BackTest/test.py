import yfinance as yf
from datetime import datetime


def fetch_yahoo_finance_data(symbol, start_date, end_date):
    try:
        from_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
        try:
            adjusted_from_date_obj = from_date_obj.replace(year=from_date_obj.year - 1)
        except ValueError:
            adjusted_from_date_obj = from_date_obj.replace(month=2, day=28, year=from_date_obj.year - 1)
        start_date = adjusted_from_date_obj.strftime('%Y-%m-%d')

        data = yf.download(symbol, start=start_date, end=end_date)
        if data.empty:
            print(f"No data found for {symbol}")
            return None

        col = ['Open', 'High', 'Low', 'Close']
        return data[col]
        print(data[col])
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None
fetch_yahoo_finance_data("INFY.NS", "2024-01-01", "2025-10-22")
# import yfinance as yf
# nsei = yf.download("INFY.NS", period="1y")
# print(nsei)