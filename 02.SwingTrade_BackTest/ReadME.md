Sure, here's a detailed `README.md` file for the provided Python trading script:

```markdown
# Trading Strategy Backtesting

This repository contains a Python script for backtesting a trading strategy on historical stock data using various technical indicators. The script fetches data from Yahoo Finance, calculates indicators such as EMA and RSI, and simulates trading based on defined conditions. It also calculates and tracks the total charges paid for trades.

## Features

- Fetches historical stock data from Yahoo Finance.
- Calculates technical indicators such as EMA, RSI, and MACD.
- Simulates trading based on predefined conditions.
- Calculates and tracks total charges paid for trades.
- Compounds profits/losses into the next position's capital allocation.
- Generates detailed reports and a master summary CSV file.

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/trading-strategy-backtesting.git
    cd trading-strategy-backtesting
    ```

2. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

## Configuration

The script uses a configuration file `config.cfg` to set various parameters. An example configuration file is provided below:

```ini
[trade_symbol]
symbols_file = symbols.txt
create_chart = true

[time_management]
from_date = 2015-01-01
to_date = 2020-12-31

[risk_management]
capital = 1000000
no_of_stock_to_trade = 10
compound = true
target_percentage = 15
stop_loss_percentage = 25
total_charges_percentage = 0.1206

[house_keeping]
cleanup_logs = true
```

### Configuration Parameters

- **symbols_file**: File containing the list of stock symbols to trade.
- **create_chart**: Whether to create charts for the trades.
- **from_date**: Start date for fetching historical data.
- **to_date**: End date for fetching historical data.
- **capital**: Initial capital for trading.
- **no_of_stock_to_trade**: Number of stocks to trade at a time.
- **compound**: Whether to compound profits/losses into the next position's capital allocation.
- **target_percentage**: Target percentage for setting sell targets.
- **stop_loss_percentage**: Stop loss percentage for setting stop losses.
- **total_charges_percentage**: Total charges percentage for calculating trading costs.
- **cleanup_logs**: Whether to clean up old log files and directories.

## Usage

1. Prepare the symbols file (`symbols.txt`), containing the stock symbols you want to trade, one per line.

2. Run the script:

    ```sh
    python main.py
    ```

3. The script will fetch data, simulate trades, and generate reports.

## Output

The script generates the following output:

- **Reports**: CSV files containing detailed trade data for each stock in the `Reports_{from_date}_to_{to_date}` directory.
- **Charts**: PNG files containing charts with buy signals, targets, and stop losses in the `Charts_{from_date}_to_{to_date}` directory (if `create_chart` is set to `true`).
- **Summary**: CSV files containing summary data for each stock in the `Summary_{from_date}_to_{to_date}` directory.
- **Master**: A master CSV file containing an overall summary in the `Master_{from_date}_to_{to_date}` directory.

## Example

An example of the main function call for processing stocks:

```python
if __name__ == "__main__":
    # Read configuration
    config = configparser.ConfigParser()
    config.read('config.cfg')
    symbols_file = config['trade_symbol']['symbols_file']
    create_chart = config.getboolean('trade_symbol', 'create_chart')
    from_date = config['time_management']['from_date']
    to_date = config['time_management']['to_date']
    capital = config.getfloat('risk_management', 'capital')
    no_of_stock_to_trade = config.getint('risk_management', 'no_of_stock_to_trade')
    compound = config.getboolean('risk_management', 'compound')
    target_percentage = config.getfloat('risk_management', 'target_percentage')
    stop_loss_percentage = config.getfloat('risk_management', 'stop_loss_percentage')
    cleanup_logs = config.getboolean('house_keeping', 'cleanup_logs')
    
    # Initialize variables
    total_charges_for_all_stocks = 0

    # Read symbols and process each stock
    with open(f'./symbols/{symbols_file}', 'r') as file:
        stocks = [line.split('#')[0].strip() for line in file if not line.lstrip().startswith('#')]
    total_stocks = len(stocks)
    print(f"Total number of stocks: {total_stocks}")

    for count, stock in enumerate(stocks, 1):
        print(f"Processing stock {count}/{total_stocks}: {stock}")
        charges_paid, trades = main(stock + ".NS", from_date, to_date, capital, target_percentage, stop_loss_percentage)
        total_charges_for_all_stocks += charges_paid

    # Create master file
    create_master_file(Summary_Dir)

    print(f"Total charges paid for all stocks: ₹{total_charges_for_all_stocks:.2f}")
    print("All stocks processed.")
```

## License

This project is licensed under the *** License - see the [****](****) file for details.
``