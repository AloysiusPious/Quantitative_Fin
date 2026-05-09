# utils/config_loader.py

import configparser
from datetime import datetime


class Config:
    pass


def load_config(path):
    p = configparser.ConfigParser()
    p.read(path)

    c = Config()

    # Mode
    c.live = p.getboolean("MODE", "live")

    # Data
    c.csv_dir = p.get("DATA", "csv_dir")
    c.vix_csv = p.get("DATA", "vix_csv")

    c.from_date = datetime.strptime(
        p.get("DATA", "from_date"), "%Y-%m-%d"
    )
    c.to_date = datetime.strptime(
        p.get("DATA", "to_date"), "%Y-%m-%d"
    )

    # Portfolio
    c.initial_capital = p.getfloat("PORTFOLIO", "initial_capital")
    c.max_positions = p.getint("PORTFOLIO", "max_positions")

    # Execution
    c.method = p.get("EXECUTION", "method")

    # Zerodha
    c.enctoken = p.get("ZERODHA_ENCTOKEN", "enctoken", fallback="")
    c.api_key = p.get("ZERODHA_API", "api_key", fallback="")
    c.access_token = p.get("ZERODHA_API", "access_token", fallback="")

    return c
