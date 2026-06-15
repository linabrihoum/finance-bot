"""
Strategy evaluation - Q-learning and rules-based trading strategies with a portfolio simulator.

API
---------------
    QLearner           — tabular Q-learning agent with optional Dyna-Q
    StrategyLearner    — RL-based trading strategy built on QLearner
    ManualStrategy     — rules-based voting strategy (SMA, BBP, Momentum)
    compute_portvals   — portfolio simulator with commission and impact costs

Indicators (returned as pd.Series indexed by date)
---------------------------------------------------
    simple_moving_avg  — price / rolling-SMA ratio
    bollinger_band     — Bollinger Band Percentage (BBP)
    momentum           — rate-of-change over a lookback window
    
"""

from .q_learner import QLearner
from .strategy_learner import StrategyLearner
from .manual_strategy import ManualStrategy
from .market_simulator import compute_portvals
from .indicators import simple_moving_avg, bollinger_band, momentum

__all__ = [
    "QLearner",
    "StrategyLearner",
    "ManualStrategy",
    "compute_portvals",
    "simple_moving_avg",
    "bollinger_band",
    "momentum",
]