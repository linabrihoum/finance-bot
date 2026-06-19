from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from strategy_evaluation.market_simulator import compute_portvals

# compute_portvals imports get_data directly; patch it at the usage site.
_PATCH = "strategy_evaluation.market_simulator.get_data"

SYMBOL = "JPM"
START_VAL = 100_000.0


def _price_df(price: float, n: int = 5) -> pd.DataFrame:
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    return pd.DataFrame({SYMBOL: price, "SPY": price}, index=idx)


def _trades_df(shares: list[float], n: int = 5) -> pd.DataFrame:
    padded = shares + [0.0] * (n - len(shares))
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    return pd.DataFrame({SYMBOL: padded}, index=idx, dtype=float)


class TestZeroTrades:
    def test_portfolio_stays_at_start_val(self):
        trades = _trades_df([0, 0, 0, 0, 0])
        with patch(_PATCH, return_value=_price_df(50.0)):
            result = compute_portvals(trades, start_val=START_VAL, commission=0.0, impact=0.0)
        assert (result == START_VAL).all()

    def test_returns_series(self):
        trades = _trades_df([0, 0, 0, 0, 0])
        with patch(_PATCH, return_value=_price_df(50.0)):
            result = compute_portvals(trades, start_val=START_VAL, commission=0.0, impact=0.0)
        assert isinstance(result, pd.Series)

    def test_index_matches_trade_dates(self):
        trades = _trades_df([0, 0, 0, 0, 0])
        with patch(_PATCH, return_value=_price_df(50.0)):
            result = compute_portvals(trades, start_val=START_VAL, commission=0.0, impact=0.0)
        assert result.index.equals(trades.index)


class TestBuyOrder:
    def test_portfolio_value_conserved_at_purchase_price(self):
        # Buy 1000 shares at $50 with no costs:
        # cash   = 100000 - 1000*50 = 50000
        # equity = 1000 * 50        = 50000  → total = 100000
        trades = _trades_df([1000])
        with patch(_PATCH, return_value=_price_df(50.0)):
            result = compute_portvals(trades, start_val=START_VAL, commission=0.0, impact=0.0)
        assert result.iloc[0] == pytest.approx(START_VAL)

    def test_portfolio_grows_when_price_rises(self):
        # Buy on day 0 at $50; price rises to $60 on remaining days
        n = 5
        idx = pd.date_range("2020-01-02", periods=n, freq="B")
        prices = pd.DataFrame({SYMBOL: [50.0] + [60.0] * (n - 1), "SPY": 50.0}, index=idx)
        trades = _trades_df([1000])
        with patch(_PATCH, return_value=prices):
            result = compute_portvals(trades, start_val=START_VAL, commission=0.0, impact=0.0)
        assert result.iloc[-1] > START_VAL

    def test_commission_reduces_final_value(self):
        trades = _trades_df([1000])
        with patch(_PATCH, return_value=_price_df(50.0)):
            no_comm = compute_portvals(trades, start_val=START_VAL, commission=0.0, impact=0.0)
        with patch(_PATCH, return_value=_price_df(50.0)):
            with_comm = compute_portvals(trades, start_val=START_VAL, commission=9.95, impact=0.0)
        assert with_comm.iloc[-1] < no_comm.iloc[-1]
        assert no_comm.iloc[-1] - with_comm.iloc[-1] == pytest.approx(9.95)

    def test_impact_reduces_cash_on_buy(self):
        # Impact = 0.01 means buy costs price * (1 + 0.01) per share
        price = 100.0
        shares = 1000
        impact = 0.01
        trades = _trades_df([shares])
        with patch(_PATCH, return_value=_price_df(price)):
            no_impact = compute_portvals(trades, start_val=200_000.0, commission=0.0, impact=0.0)
        with patch(_PATCH, return_value=_price_df(price)):
            with_impact = compute_portvals(trades, start_val=200_000.0, commission=0.0, impact=impact)
        # Extra cost = shares * price * impact = 1000 * 100 * 0.01 = 1000
        assert no_impact.iloc[-1] - with_impact.iloc[-1] == pytest.approx(shares * price * impact)


class TestSellOrder:
    def test_sell_raises_cash(self):
        # Sell (short) 1000 shares at $50 from a flat position
        # cash = 100000 + 1000*50*(1-0) - 0 = 150000
        # equity = -1000 * 50 = -50000 → total = 100000
        trades = _trades_df([-1000])
        with patch(_PATCH, return_value=_price_df(50.0)):
            result = compute_portvals(trades, start_val=START_VAL, commission=0.0, impact=0.0)
        assert result.iloc[0] == pytest.approx(START_VAL)

    def test_impact_reduces_proceeds_on_sell(self):
        price = 100.0
        shares = 1000
        impact = 0.01
        trades = _trades_df([-shares])
        with patch(_PATCH, return_value=_price_df(price)):
            no_impact = compute_portvals(trades, start_val=START_VAL, commission=0.0, impact=0.0)
        with patch(_PATCH, return_value=_price_df(price)):
            with_impact = compute_portvals(trades, start_val=START_VAL, commission=0.0, impact=impact)
        # Sell proceeds fall by shares * price * impact
        assert no_impact.iloc[-1] - with_impact.iloc[-1] == pytest.approx(shares * price * impact)


class TestRoundTrip:
    def test_buy_then_sell_at_same_price_net_zero_without_costs(self):
        # Day 0: buy 1000; Day 1: sell 1000 → flat position, no costs → back to start_val
        trades = _trades_df([1000, -1000])
        with patch(_PATCH, return_value=_price_df(50.0)):
            result = compute_portvals(trades, start_val=START_VAL, commission=0.0, impact=0.0)
        assert result.iloc[-1] == pytest.approx(START_VAL)

    def test_buy_then_sell_at_higher_price_nets_profit(self):
        n = 5
        idx = pd.date_range("2020-01-02", periods=n, freq="B")
        # Buy at $100 on day 0, sell at $110 on day 1
        prices = pd.DataFrame({SYMBOL: [100.0, 110.0, 110.0, 110.0, 110.0], "SPY": 100.0}, index=idx)
        trades = _trades_df([1000, -1000])
        with patch(_PATCH, return_value=prices):
            result = compute_portvals(trades, start_val=START_VAL, commission=0.0, impact=0.0)
        assert result.iloc[-1] > START_VAL
