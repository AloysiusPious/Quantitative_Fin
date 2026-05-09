# execution/execution_factory.py

from .paper_broker import PaperBroker
from .enctoken_broker import EnctokenBroker
from .kite_api_broker import KiteApiBroker


def get_broker(config):
    """
    Returns broker instance based on config.
    """

    # Backtest / paper mode
    if not config.live:
        return PaperBroker()

    # Live mode
    if config.method == "enctoken":
        return EnctokenBroker(config.enctoken)

    if config.method == "api":
        return KiteApiBroker(
            api_key=config.api_key,
            access_token=config.access_token
        )

    raise ValueError(f"Unknown execution method: {config.method}")
