from __future__ import annotations

import datetime as dt

import pandas as pd

from . import indicators as ind
from . import market_simulator as ms
from . import q_learner as ql
from . import util as ut


class StrategyLearner:
    """
    Q-learning agent that learns to trade a single equity

    The state space is formed by discretizing three technical indicators
    (Bollinger Band Percentage, SMA ratio, Momentum) into `window`
    quantile bins each, then combining them into a single integer state:

        state = 100 * bb_bin + 10 * sma_bin + momentum_bin

    This yields up to 1000 distinct states (10 × 10 × 10). The agent can take one of three actions: 
    1. Go short (-1000 shares)
    2. Stay flat (0)
    3. Go long (+1000 shares)

    Training runs the agent repeatedly over the in-sample window until the
    final portfolio value converges between episodes
    """

    def __init__(
        self,
        verbose: bool = False,
        impact: float = 0.0,
        commission: float = 0.0,
    ) -> None:
        """
        Parameters
        ----------
        verbose : bool
            Print diagnostic output during training when True

        impact : float
            Market impact cost per trade as a fraction of share price.
            Used to shape the reward signal so the agent accounts for slippage when learning

        commission : float
            Flat commission per trade

        """
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.q_learner = ql.QLearner(num_states=1000, num_actions=3, alpha=0.2, gamma=0.9, rar=0.8, radr=0.9)

    def _discretize(self, series: pd.Series, n_bins: int) -> pd.Series:
        """
        Bin a continuous indicator series into integer quantile labels.

        Uses quantile-based discretization so each bin contains roughly the
        same number of observations. Duplicate bin edges (common with sparse
        or low-variance data) are silently merged rather than raising an error.

        Parameters
        ----------
        series : pd.Series
            Indicator values to discretize

        n_bins : int
            Number of bins

        Returns
        -------
        pd.Series
            Integer bin labels in [0, n_bins - 1]; NaN where input was NaN

        """
        return pd.qcut(series, n_bins, labels=False, duplicates="drop")

    def add_evidence(
        self,
        symbol: str = "IBM",
        sd: dt.datetime = dt.datetime(2008, 1, 1),
        ed: dt.datetime = dt.datetime(2009, 1, 1),
        sv: int = 10_000,
    ) -> None:
        """
        Train the Q-learner on historical data for the given symbol and window.

        Iterates over the training period repeatedly, updating the Q-table on
        each step using the daily return (adjusted for impact) as the reward.
        Stops when the final portfolio value changes by less than $100 between
        consecutive episodes and at least 30 episodes have completed.

        Parameters
        ----------
        symbol : str
            Ticker to train on

        sd : dt.datetime
            Start of the training window

        ed : dt.datetime
            End of the training window

        sv : int
            Starting portfolio value used to evaluate convergence

        """
        self.sv = sv
        syms = [symbol]
        window = 10

        prices_all = ut.get_data(syms, pd.date_range(sd, ed))
        prices = prices_all[syms]

        bb = ind.bollinger_band(symbol = syms, sd = sd, ed = ed, window = window)
        sma = ind.simple_moving_avg(symbol = syms, sd = sd, ed = ed, window = window)
        mom = ind.momentum(symbol = syms, sd = sd, ed = ed, window = window)

        # Slice from window (not window-1) so momentum NaNs are excluded.
        # Momentum needs window lookback rows, so index window-1 is still NaN.
        bb_bins = self._discretize(bb[window:], window)
        sma_bins = self._discretize(sma[window:], window)
        mom_bins = self._discretize(mom[window:], window)
        states = pd.DataFrame(
            (100 * bb_bins + 10 * sma_bins + mom_bins).astype(int),
            index=bb_bins.index,
            columns=syms,
        )

        # Re-initialize the learner at the start of training
        self.q_learner = ql.QLearner(num_states = 1000, num_actions = 3, alpha = 0.2, gamma = 0.9, rar = 0.8, radr = 0.9)

        df_trades = prices.copy()
        df_trades[:] = 0

        converged = False
        episode = 0
        portval_prev = float(sv)

        while not converged:
            df_trades[:] = 0

            # Actions: 0 = short, 1 = flat, 2 = long
            action = self.q_learner.querysetstate(int(states.iloc[0].values[0]))
            if action == 0:
                holding = -1000
            elif action == 2:
                holding = 1000
            else:
                holding = 0
            df_trades.loc[states.index[0]] = holding

            for i in range(1, states.shape[0]):
                # Impact multiplier aligns the reward sign with the position direction
                impact_mult = -1 if action == 0 else (1 if action == 2 else 0)
                daily_return = (
                    prices.loc[states.index[i]].values[0]
                    / prices.loc[states.index[i - 1]].values[0]
                ) - 1
                reward = holding * (daily_return - self.impact * impact_mult)

                action = self.q_learner.query(int(states.iloc[i].values[0]), reward)

                if action == 0:
                    if holding == 0:
                        df_trades.loc[states.index[i]] = -1000.0
                    elif holding == 1000:
                        df_trades.loc[states.index[i]] = -2000.0
                elif action == 2:
                    if holding == 0:
                        df_trades.loc[states.index[i]] = 1000.0
                    elif holding == -1000:
                        df_trades.loc[states.index[i]] = 2000.0

                holding = int(holding + df_trades.loc[states.index[i]].values[0])

            portval = ms.compute_portvals(df_trades, start_val=sv, commission=0, impact=0)

            # Convergence - final portfolio value stable within $100 after 30+ episodes
            if abs(float(portval.iloc[-1]) - portval_prev) < 100 and episode > 30:
                converged = True
            episode += 1
            portval_prev = float(portval.iloc[-1])

    def testPolicy(
        self,
        symbol: str = "IBM",
        sd: dt.datetime = dt.datetime(2009, 1, 1),
        ed: dt.datetime = dt.datetime(2010, 1, 1),
        sv: int = 10_000,
    ) -> pd.DataFrame:
        """
        Apply the trained policy to a date range and return a trade schedule

        Runs the greedy policy (no exploration) using the Q-table learned
        during add_evidence(). Can be called on either in-sample or out-of-sample data.

        Parameters
        ----------
        symbol : str
            Ticker to evaluate

        sd : dt.datetime
            Start of the evaluation window

        ed : dt.datetime
            End of the evaluation window

        sv : int
            Starting portfolio value (unused; only included for API consistency)

        Returns
        -------
        pd.DataFrame
            Single-column DataFrame indexed by trading date. Positive values
            are buy orders; negative values are sell orders; zero is no trade

        """
        syms = [symbol]
        window = 10

        prices_all = ut.get_data([symbol], pd.date_range(sd, ed))  # SPY added automatically for market-hours alignment
        prices = prices_all[syms]

        bb = ind.bollinger_band(symbol = syms, sd = sd, ed = ed, window = window)
        sma = ind.simple_moving_avg(symbol = syms, sd = sd, ed = ed, window = window)
        mom = ind.momentum(symbol = syms, sd = sd, ed = ed, window = window)

        bb_bins = self._discretize(bb[window:], window)
        sma_bins = self._discretize(sma[window:], window)
        mom_bins = self._discretize(mom[window:], window)
        states = pd.DataFrame(
            (100 * bb_bins + 10 * sma_bins + mom_bins).astype(int),
            index=bb_bins.index,
            columns=syms,
        )

        df_trades = prices.copy()
        df_trades[:] = 0
        position = 0  # current share holdings

        for i in range(states.shape[0]):
            action = self.q_learner.querysetstate(int(states.iloc[i].values[0]))

            if action == 0:  # short
                if position == 0:
                    df_trades.loc[states.index[i]] = -1000.0
                elif position == 1000:
                    df_trades.loc[states.index[i]] = -2000.0
            elif action == 2:  # long
                if position == 0:
                    df_trades.loc[states.index[i]] = 1000.0
                elif position == -1000:
                    df_trades.loc[states.index[i]] = 2000.0

            position = int(position + df_trades.loc[states.index[i]].values[0])

        return df_trades
