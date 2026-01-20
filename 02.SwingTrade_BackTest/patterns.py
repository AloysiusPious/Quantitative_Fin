import pandas as pd

def morning_star(data):
    """
    Detect Morning Star pattern (non-repainting, NSE-friendly)

    Returns:
        pd.Series[bool] aligned with data.index
    """
    # Candle metrics
    body = (data['Close'] - data['Open']).abs()
    range_ = data['High'] - data['Low']

    # 1️⃣ Bearish context (relaxed)
    red_context = (
            (data['Close'].shift(2) < data['Open'].shift(2)) |
            (data['Close'].shift(3) < data['Open'].shift(3))
    )

    # 2️⃣ Yesterday exhaustion (relaxed)
    yesterday_exhaustion = (
            (data['Low'].shift(1) < data['Low'].shift(2)) |
            (data['Low'].shift(1) < data['Low'].shift(3))
    )

    # 3️⃣ Today controlled bullish candle
    today_controlled_bull = (
            (data['Close'] > data['Open']) &  # green
            (data['Close'] > data['Close'].shift(1)) &  # higher close
            (data['Low'] >= data['Low'].shift(1)) &  # no lower low
            (body < 0.6 * range_) &  # not big candle
            (body < body.rolling(10).mean())  # relative small
    )

    # 4️⃣ Avoid extended breakouts
    not_overextended = (
            (data['Close'] < data['High'].shift(2)) |
            (data['Close'] < data['High'].shift(3))
    )

    # 🌅 Relaxed Morning Star (risk-aware)
    data['Morning_Star_Risk_Controlled'] = (
            red_context &
            yesterday_exhaustion &
            today_controlled_bull &
            not_overextended
    )

