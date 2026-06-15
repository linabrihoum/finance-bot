from __future__ import annotations

import datetime as dt

import matplotlib.pyplot as plt

from . import strategy_learner as sl
from . import manual_strategy as ms
from . import market_simulator as mk
from .plotting import save_chart


def experiment1(
    symbol: str = "JPM",
    sd: dt.datetime = dt.datetime(2008, 1, 1),
    ed: dt.datetime = dt.datetime(2009, 12, 31),
    sv: int = 100_000,
    impact: float = 0.0,
    commission: float = 0.0,
) -> None:
    
    """
    Compare Manual Strategy, Strategy Learner, and buy-and-hold benchmark

    Trains the Strategy Learner on [sd, ed], evaluates all three strategies
    over the same in-sample window, normalizes portfolio values to 1.0 at
    the start date, and saves the comparison chart

    Parameters
    ----------
    symbol : str
        Ticker to trade

    sd : dt.datetime
        Start of the evaluation window

    ed : dt.datetime
        End of the evaluation window

    sv : int
        Starting portfolio value 

    impact : float
        Market impact cost per trade (fraction of share price)

    commission : float
        Flat commission per trade

    """
    manual_learner = ms.ManualStrategy()
    m_trades = manual_learner.testPolicy(symbol = symbol, sd = sd, ed = ed)

    learner = sl.StrategyLearner(verbose = False, impact = impact)
    learner.add_evidence(symbol = symbol, sd = sd, ed = ed, sv = sv)

    m_portv = mk.compute_portvals(
        manual_learner.testPolicy(symbol = symbol, sd = sd, ed = ed),
        start_val = sv,
        commission = commission,
        impact = impact,
    )
    sl_portv = mk.compute_portvals(
        learner.testPolicy(symbol = symbol, sd = sd, ed = ed, sv = sv),
        start_val = sv,
        commission = commission,
        impact = impact,
    )

    # benchmark: buy 1000 shares on day 1, sell on the last day
    b_trades = m_trades.copy()
    b_trades[:] = 0
    b_trades.iloc[0, 0] = 1000
    b_trades.iloc[-1, 0] = -1000
    b_portv = mk.compute_portvals(b_trades, start_val = sv, commission = commission, impact = impact)

    results = m_trades.copy()
    results["Manual Strategy"] = m_portv / m_portv.iloc[0]
    results["Strategy Learner"] = sl_portv / sl_portv.iloc[0]
    results["Benchmark"] = b_portv / b_portv.iloc[0]
    results = results.drop(results.columns[0], axis = 1)

    save_chart(results, "Experiment1.png", "Strategy Comparison (In-Sample)", "Date", "Normalized Portfolio Value")
