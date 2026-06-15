from __future__ import annotations

import datetime as dt
from typing import List, Optional

import pandas as pd
from . import util as ut
from .plotting import save_chart


def simple_moving_avg(
    symbol: Optional[List[str]] = None,
    sd: dt.datetime = dt.datetime(2008, 1, 1),
    ed: dt.datetime = dt.datetime(2009, 12, 31),
    window: int = 10,
) -> pd.Series:
    
    """
    Compute the price-to-SMA ratio for a symbol

    Prices are normalized to the first trading day before computing the SMA,
    so the ratio reflects how far the current price sits relative to its
    rolling mean on a scale anchored at 1.0.

    Returns values around 1.0. Values > 1 mean price is above SMA; < 1 below.

    Parameters
    ----------
    symbol : list of str
        Tickers to retrieve. Only the first symbol is used

    sd : dt.datetime
        Start of the date range
        
    ed : dt.datetime
        End of the date range

    window : int
        Rolling window length in trading days

    Returns
    -------
    pd.Series
        Price / SMA ratio indexed by trading date

    """
    if symbol is None:
        symbol = ["JPM"]
    prices = ut.get_data(symbol, pd.date_range(sd, ed))
    prices = prices[symbol]
    prices = prices / prices.iloc[0]  # normalize to first trading day
    sma = prices.rolling(window = window).mean()
    prices["SMA"] = sma
    prices["SMA Ratio"] = prices.values[:, 0] / prices.values[:, 1]
    return prices["SMA Ratio"]


def bollinger_band(
    symbol: Optional[List[str]] = None,
    sd: dt.datetime = dt.datetime(2008, 1, 1),
    ed: dt.datetime = dt.datetime(2009, 12, 31),
    window: int = 10,
) -> pd.Series:
    
    """
    
    Compute the Bollinger Band Percentage (BBP) for a symbol

    BBP = (price - lower_band) / (upper_band - lower_band)

    Bands are set at ±2 standard deviations from the rolling SMA. A value
    of 0.0 means the price is exactly on the lower band; 1.0 on the upper band.
    Values outside [0, 1] indicate the price has broken out of the bands

    Parameters
    ----------
    symbol : list of str
        Tickers to retrieve. Only the first symbol is used

    sd : dt.datetime
        Start of the date range

    ed : dt.datetime
        End of the date range

    window : int
        Rolling window length in trading days

    Returns
    -------
    pd.Series
        BBP values indexed by trading date.
    """

    if symbol is None:
        symbol = ["JPM"]
    prices = ut.get_data(symbol, pd.date_range(sd, ed))
    prices = prices[symbol]
    prices = prices / prices.iloc[0]  # normalize to first trading day

    sma = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    lower_band = sma - 2 * rolling_std
    upper_band = sma + 2 * rolling_std

    # fraction of position within the band; <0 or >1 means a breakout
    bbp = (prices - lower_band) / (upper_band - lower_band)
    prices["Upper Band"] = upper_band
    prices["Lower Band"] = lower_band
    prices["BBP Ratio"] = bbp
    return prices["BBP Ratio"]


def momentum(
    symbol: Optional[List[str]] = None,
    sd: dt.datetime = dt.datetime(2008, 1, 1),
    ed: dt.datetime = dt.datetime(2009, 12, 31),
    window: int = 10,
) -> pd.Series:
    
    """
    Compute price momentum (rate of change) for a symbol

    Momentum at time t = (price[t] - price[t - window]) / price[t - window]

    Positive values mean the price has risen over the lookback period;
    negative values mean it has fallen. The first `window` rows are NaN.

    Parameters
    ----------
    symbol : list of str
        Tickers to retrieve. Only the first symbol is used

    sd : dt.datetime
        Start of the date range

    ed : dt.datetime
        End of the date range

    window : int
        Lookback period in trading days

    Returns
    -------
    pd.Series
        Momentum values indexed by trading date.
    """

    if symbol is None:
        symbol = ["JPM"]
    prices = ut.get_data(symbol, pd.date_range(sd, ed))
    prices = prices[symbol]
    prices = prices / prices.iloc[0]  # normalize to first trading day
    moment = (prices - prices.shift(window)) / prices.shift(window)
    
    prices["Momentum"] = moment
    prices["Momentum Ratio"] = moment
    return prices["Momentum Ratio"]


"""
    Graphing helpers
    These accept a pre-loaded prices DataFrame, compute the indicator, and save the charts
"""

def graph_simple_moving_avg(prices: pd.DataFrame, window: int) -> pd.Series:
    # Compute SMA ratio and save a chart. Modifies prices in-place 
    prices = prices / prices.iloc[0]
    sma = prices.rolling(window = window).mean()
    prices["SMA"] = sma
    prices["SMA Ratio"] = prices.values[:, 0] / prices.values[:, 1]
    save_chart(prices, "Simple Moving Average.png", "SMA Ratio", "Date", "Normalized Price")
    return prices["SMA Ratio"]


def graph_bollinger_band(prices: pd.DataFrame, window: int) -> pd.Series:
    # Compute Bollinger Band Percentage and save a chart. Modifies prices in-place
    prices = prices / prices.iloc[0]
    sma = prices.rolling(window = window).mean()
    rolling_std = prices.rolling(window = window).std()
    lower_band = sma - 2 * rolling_std
    upper_band = sma + 2 * rolling_std
    bbp = (prices - lower_band) / (upper_band - lower_band)
    prices["Upper Band"] = upper_band
    prices["Lower Band"] = lower_band
    save_chart(prices, "BollingerBand.png", "Bollinger Bands", "Date", "Normalized Price")
    prices["BBP Ratio"] = bbp
    return prices["BBP Ratio"]


def graph_momentum(prices: pd.DataFrame, window: int) -> pd.Series:
    # Compute momentum (rate of change) and save a chart. Modifies prices in-place
    prices = prices / prices.iloc[0]
    moment = (prices - prices.shift(window)) / prices.shift(window)
    prices["Momentum"] = moment
    prices["Momentum Ratio"] = moment
    save_chart(prices, "Momentum.png", "Price Momentum", "Date", "Rate of Change")
    return prices["Momentum Ratio"]


if __name__ == "__main__":
    _symbol = ["JPM"]
    _sd = dt.datetime(2008, 1, 1)
    _ed = dt.datetime(2009, 12, 31)
    _window = 10

    _prices = ut.get_data(_symbol, pd.date_range(_sd, _ed))
    _prices = _prices[_symbol]

    graph_bollinger_band(_prices.copy(), _window)
    graph_simple_moving_avg(_prices.copy(), _window)
    graph_momentum(_prices.copy(), _window)
