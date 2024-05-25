import os
import pandas as pd
import matplotlib.pyplot as plt
import configparser

def read_and_process_csv(filepath):
    df = pd.read_csv(filepath, parse_dates=['Buy Date', 'Exited Date'])
    return df

def process_all_files(directory):
    all_trades = []

    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            df = read_and_process_csv(filepath)
            all_trades.append(df)

    return pd.concat(all_trades, ignore_index=True)

def plot_profit_trend(all_trades, initial_capital):
    # Ensure trades are sorted by 'Buy Date'
    all_trades = all_trades.sort_values(by='Buy Date')

    # Calculate cumulative profit
    all_trades['Cumulative Profit'] = all_trades['Profit Amount'].cumsum()

    # Calculate capital over time
    all_trades['Capital'] = initial_capital + all_trades['Cumulative Profit']

    plt.figure(figsize=(14, 7))
    plt.plot(all_trades['Buy Date'], all_trades['Capital'], marker='.', linestyle='-', color='b', label='Capital Over Time')
    # Plot the line
    #plt.plot(all_trades['Buy Date'], all_trades['Capital'], linestyle='-', color='b', label='Capital Over Time')

    # Plot the markers
    #plt.scatter(all_trades['Buy Date'], all_trades['Capital'], color='r', marker='.', label='Buy Points')

    # Annotate initial capital and final capital
    plt.annotate(f'Start: ₹{initial_capital}', xy=(all_trades['Buy Date'].iloc[0], initial_capital), xytext=(all_trades['Buy Date'].iloc[0], initial_capital),
                 arrowprops=dict(facecolor='green', shrink=0.05))
    plt.annotate(f'End: ₹{all_trades["Capital"].iloc[-1]:.2f}', xy=(all_trades['Buy Date'].iloc[-1], all_trades['Capital'].iloc[-1]), xytext=(all_trades['Buy Date'].iloc[-1], all_trades['Capital'].iloc[-1]),
                 arrowprops=dict(facecolor='red', shrink=0.05))
    plt.title('Capital Growth Over Time')
    plt.xlabel('Date')
    plt.ylabel('Capital (₹)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'capital_drawdown.png')
    plt.close()
    plt.show()

def main(directory, initial_capital):
    all_trades = process_all_files(directory)
    plot_profit_trend(all_trades, initial_capital)

if __name__ == "__main__":
    # Create a ConfigParser object
    config = configparser.ConfigParser()
    # Read the cfg file
    config.read('config.cfg')
    symbols_file = str(config['trade_symbol']['symbols_file'])
    create_chart = str(config['trade_symbol']['create_chart'])
    create_chart = True if create_chart == 'true' else False
    # Access sections and keys
    from_date = str(config['time_management']['from_date'])
    to_date = str(config['time_management']['to_date'])
    #from_date = datetime.strptime(config['time_management']['from_date'], '%Y-%m-%d').date()
    #to_date = datetime.strptime(config['time_management']['to_date'], '%Y-%m-%d').date()
    year_wise = str(config['time_management']['year_wise'])
    year_wise = True if year_wise == 'true' else False
    year_split = int(config['time_management']['year_split'])
    capital = float(config['risk_management']['capital'])
    no_of_stock_to_trade = int(config['risk_management']['no_of_stock_to_trade'])
    compound = str(config['risk_management']['compound'])
    compound = True if compound == 'true' else False
    target_percentage = float(config['risk_management']['target_percentage'])
    stop_loss_percentage = float(config['risk_management']['stop_loss_percentage'])
    total_charges_percentage = float(config['risk_management']['charges_percentage'])

    symbols_type = symbols_file.split('.')[0]
    Reports_Dir = f'{symbols_type}_Reports_{from_date}_to_{to_date}'
    Charts_Dir = f'{symbols_type}_Charts_{from_date}_to_{to_date}'
    Summary_Dir = f'{symbols_type}_Summary_{from_date}_to_{to_date}'
    Master_Dir = f'{symbols_type}_Master_{from_date}_to_{to_date}'
    main(Reports_Dir, capital)