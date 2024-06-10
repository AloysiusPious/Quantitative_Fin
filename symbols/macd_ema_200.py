import matplotlib.pyplot as plt
import yfinance as yf

# Download historical data from Yahoo Finance
stock_data = yf.download('TCS.NS', start='2020-01-01', end='2024-06-01')

# Calculate EMA 200
ema_200 = stock_data['Adj Close'].ewm(span=200, adjust=False).mean()

# Calculate MACD
short_ema = stock_data['Adj Close'].ewm(span=12, adjust=False).mean()
long_ema = stock_data['Adj Close'].ewm(span=26, adjust=False).mean()
macd_line = short_ema - long_ema
signal_line = macd_line.ewm(span=9, adjust=False).mean()
macd_histogram = macd_line - signal_line

# Calculate buy and sell signals
buy_signal = (macd_line > signal_line) & (macd_line < 0) & (ema_200.diff() > 0)
sell_signal = (macd_line < signal_line) & (macd_line > 0) & (ema_200.diff() < 0)

# Plotting
plt.figure(figsize=(10, 8))

# Main chart
plt.subplot(2, 1, 1)
plt.plot(stock_data['Adj Close'], label='MSFT', color='blue')
plt.plot(ema_200, label='EMA 200', color='red')  # Adding EMA 200 to the plot
plt.scatter(stock_data.index[buy_signal], stock_data['Adj Close'][buy_signal], marker='^', color='green', label='Buy Signal')
plt.scatter(stock_data.index[sell_signal], stock_data['Adj Close'][sell_signal], marker='v', color='red', label='Sell Signal')
plt.title('Microsoft Stock Price with Buy/Sell Signals')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)

# MACD chart
plt.subplot(2, 1, 2)
plt.plot(macd_line, label='MACD Line', color='orange')
plt.plot(signal_line, label='Signal Line', color='green')
plt.bar(macd_histogram.index, macd_histogram, width=0.5, color='gray', alpha=0.5)
plt.axhline(0, linestyle='--', color='gray')
plt.title('MACD')
plt.xlabel('Date')
plt.ylabel('MACD')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()