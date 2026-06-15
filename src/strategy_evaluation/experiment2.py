from __future__ import annotations

import pandas as pd
from . import market_simulator as mksim
from .plotting import save_chart


def experiment2(s_trades: pd.DataFrame, sv: int, commission: float) -> None:
    """
    
    Show how market impact cost affects Strategy Learner portfolio returns

    Runs the same trade schedule through the market simulator three times with
    increasing impact values, then normalizes the portfolio curves to
    show the sensitivity to slippage.

    Parameters
    ----------

    s_trades : pd.DataFrame
        Trade schedule produced by StrategyLearner.testPolicy()

    sv : int
        Starting portfolio value

    commission : float
        Flat commission per trade

    """
    portvals_no_impact = mksim.compute_portvals(s_trades, start_val = sv, commission = commission, impact = 0.000)
    portvals_low_impact = mksim.compute_portvals(s_trades, start_val = sv, commission = commission, impact = 0.005)
    portvals_high_impact = mksim.compute_portvals(s_trades, start_val = sv, commission = commission, impact = 0.010)

    results = s_trades.copy()
    results["Impact=0.000"] = portvals_no_impact / portvals_no_impact.iloc[0]
    results["Impact=0.005"] = portvals_low_impact / portvals_low_impact.iloc[0]
    results["Impact=0.010"] = portvals_high_impact / portvals_high_impact.iloc[0]
    results = results.drop(results.columns[0], axis = 1)

    save_chart(results, "Experiment2.png", "Market Impact Sensitivity (In-Sample)", "Date", "Normalized Portfolio Value")
