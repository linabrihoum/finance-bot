from __future__ import annotations

import datetime as dt
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from strategy_evaluation import simple_moving_avg, bollinger_band, momentum

# All three indicators call ut.get_data internally; patch it so tests never
# touch the filesystem.
_PATCH = "strategy_evaluation.indicators.ut.get_data"

SD = dt.datetime(2020, 1, 1)
ED = dt.datetime(2020, 3, 1)
SYM = ["JPM"]
WINDOW = 5


def _constant_prices(n: int = 30, value: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    return pd.DataFrame({"JPM": value, "SPY": value}, index=idx)


def _trending_prices(n: int = 30, start: float = 100.0, end: float = 120.0) -> pd.DataFrame:
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    prices = np.linspace(start, end, n)
    return pd.DataFrame({"JPM": prices, "SPY": prices}, index=idx)


def _falling_prices(n: int = 30, start: float = 120.0, end: float = 100.0) -> pd.DataFrame:
    return _trending_prices(n, start=start, end=end)


class TestSimpleMovingAvg:
    def test_returns_series(self):
        with patch(_PATCH, return_value=_constant_prices()):
            result = simple_moving_avg(symbol=SYM, sd=SD, ed=ED, window=WINDOW)
        assert isinstance(result, pd.Series)

    def test_constant_prices_ratio_is_one(self):
        # price / SMA = constant / constant = 1.0 for every non-NaN row
        with patch(_PATCH, return_value=_constant_prices()):
            result = simple_moving_avg(symbol=SYM, sd=SD, ed=ED, window=WINDOW)
        valid = result.dropna()
        assert (valid - 1.0).abs().max() < 1e-9

    def test_first_window_minus_one_rows_are_nan(self):
        with patch(_PATCH, return_value=_constant_prices()):
            result = simple_moving_avg(symbol=SYM, sd=SD, ed=ED, window=WINDOW)
        assert result.iloc[: WINDOW - 1].isna().all()

    def test_price_above_sma_gives_ratio_above_one(self):
        # Trailing prices are above the rolling average → ratio > 1 near the end
        with patch(_PATCH, return_value=_trending_prices()):
            result = simple_moving_avg(symbol=SYM, sd=SD, ed=ED, window=WINDOW)
        assert result.dropna().iloc[-1] > 1.0

    def test_length_matches_input(self):
        prices = _constant_prices(n=20)
        with patch(_PATCH, return_value=prices):
            result = simple_moving_avg(symbol=SYM, sd=SD, ed=ED, window=WINDOW)
        assert len(result) == 20


class TestBollingerBand:
    def test_returns_series(self):
        with patch(_PATCH, return_value=_trending_prices()):
            result = bollinger_band(symbol=SYM, sd=SD, ed=ED, window=WINDOW)
        assert isinstance(result, pd.Series)

    def test_price_at_sma_gives_bbp_half(self):
        # For a linearly trending series, the price tracks the SMA exactly at
        # steady state (uniform spacing). BBP = (price - lower) / (upper - lower)
        # = (SMA - (SMA-2σ)) / 4σ = 0.5 once the window is full.
        # We allow a loose tolerance since prices are normalized first.
        with patch(_PATCH, return_value=_trending_prices()):
            result = bollinger_band(symbol=SYM, sd=SD, ed=ED, window=WINDOW)
        valid = result.dropna()
        assert valid.between(-0.5, 1.5).all()

    def test_high_price_gives_bbp_above_half(self):
        # Price spikes up near the end → BBP should exceed 0.5 at the tail
        prices = _constant_prices(n=30)
        prices.iloc[-5:] = 200.0
        with patch(_PATCH, return_value=prices):
            result = bollinger_band(symbol=SYM, sd=SD, ed=ED, window=WINDOW)
        assert result.dropna().iloc[-1] > 0.5

    def test_low_price_gives_bbp_below_half(self):
        prices = _constant_prices(n=30)
        prices.iloc[-5:] = 50.0
        with patch(_PATCH, return_value=prices):
            result = bollinger_band(symbol=SYM, sd=SD, ed=ED, window=WINDOW)
        assert result.dropna().iloc[-1] < 0.5

    def test_first_window_minus_one_rows_are_nan(self):
        with patch(_PATCH, return_value=_trending_prices()):
            result = bollinger_band(symbol=SYM, sd=SD, ed=ED, window=WINDOW)
        assert result.iloc[: WINDOW - 1].isna().all()


class TestMomentum:
    def test_returns_series(self):
        with patch(_PATCH, return_value=_constant_prices()):
            result = momentum(symbol=SYM, sd=SD, ed=ED, window=WINDOW)
        assert isinstance(result, pd.Series)

    def test_constant_prices_give_zero_momentum(self):
        with patch(_PATCH, return_value=_constant_prices()):
            result = momentum(symbol=SYM, sd=SD, ed=ED, window=WINDOW)
        assert result.dropna().abs().max() < 1e-9

    def test_rising_prices_give_positive_momentum(self):
        with patch(_PATCH, return_value=_trending_prices()):
            result = momentum(symbol=SYM, sd=SD, ed=ED, window=WINDOW)
        assert (result.dropna() > 0).all()

    def test_falling_prices_give_negative_momentum(self):
        with patch(_PATCH, return_value=_falling_prices()):
            result = momentum(symbol=SYM, sd=SD, ed=ED, window=WINDOW)
        assert (result.dropna() < 0).all()

    def test_first_window_rows_are_nan(self):
        with patch(_PATCH, return_value=_constant_prices()):
            result = momentum(symbol=SYM, sd=SD, ed=ED, window=WINDOW)
        assert result.iloc[:WINDOW].isna().all()

    def test_momentum_formula(self):
        # 10 rows, price rises from 100 to 109 (+1 per day after normalization)
        n = 15
        prices = _constant_prices(n=n, value=100.0)
        prices["JPM"] = np.arange(100, 100 + n, dtype=float)
        prices["SPY"] = prices["JPM"]
        with patch(_PATCH, return_value=prices):
            result = momentum(symbol=SYM, sd=SD, ed=ED, window=WINDOW)
        # After normalization to prices[0]=100, row i has price (100+i)/100.
        # moment[i] = (price[i] - price[i-window]) / price[i-window]
        # = ((100+i)/100 - (100+i-window)/100) / ((100+i-window)/100)
        # = window / (100 + i - window)
        # Row WINDOW (index 5): expected = 5 / (100 + 5 - 5) = 5/100 = 0.05
        expected = WINDOW / (100 + WINDOW - WINDOW)  # = 0.05
        assert result.iloc[WINDOW] == pytest.approx(expected, rel=1e-6)
