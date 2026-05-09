def get_broker(config):
    if not config.live:
        return PaperBroker()

    if config.execution_method == "api":
        return KiteApiBroker(config)

    if config.execution_method == "enctoken":
        return EnctokenBroker(config)

    raise ValueError("Invalid execution method")
