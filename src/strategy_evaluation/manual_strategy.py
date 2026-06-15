from __future__ import annotations

import datetime as dt
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

from . import indicators as ind
from . import market_simulator as mk
from . import util as ut


class ManualStrategy:
    """
    Rules-based trading strategy using three technical indicators.

    Combines SMA ratio, Bollinger Band Percentage, and momentum into a
    composite signal. Each indicator votes -1 (sell), 0 (neutral), or +1
    (buy). The net vote determines the target position: +1000 shares (long),
    0 (flat), or -1000 shares (short). Trades are sized to move from the
    current position to the new target in a single order.
    """

    def __init__(
        self,
        verbose: bool = False,
        impact: float = 0.0,
        commission: float = 9.95,
    ) -> None:
        """
        Parameters
        ----------
        verbose : bool
            Print diagnostic output when True

        impact : float
            Market impact cost per trade as a fraction of share price

        commission : float
            Flat commission per trade
        """
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.window = 10

        # populated by testPolicy(); used by in_sample/out_sample for chart annotations
        self.long: List[dt.datetime] = []
        self.short: List[dt.datetime] = []

    def testPolicy(
        self,
        symbol: str = "JPM",
        sd: dt.datetime = dt.datetime(2008, 1, 1),
        ed: dt.datetime = dt.datetime(2009, 12, 31),
        sv: int = 100_000,
    ) -> pd.DataFrame:
        """Generate a trade schedule using the three-indicator voting rule.

        Each indicator is converted to a directional vote:
        - SMA ratio > 1.02  → sell (-1),  < 0.98 → buy (+1), else neutral
        - BBP > 1.5         → sell (-1),  < -0.5 → buy (+1), else neutral
        - Momentum > 1.02   → sell (-1),  < 0.98 → buy (+1), else neutral

        The sum of votes determines the target holding. Trade sizes are the
        difference between consecutive target holdings.

        Parameters
        ----------
        symbol : str
            Ticker to trade.
        sd : dt.datetime
            Start of the evaluation window.
        ed : dt.datetime
            End of the evaluation window.
        sv : int
            Starting portfolio value (unused; included for API consistency).

        Returns
        -------
        pd.DataFrame
            Single-column DataFrame indexed by trading date. Positive values
            are buy orders; negative values are sell orders; zero is no trade.
        """
        bb_indicator = ind.bollinger_band(symbol=[symbol], sd=sd, ed=ed)
        sma_indicator = ind.simple_moving_avg(symbol=[symbol], sd=sd, ed=ed)
        mom_indicator = ind.momentum(symbol=[symbol], sd=sd, ed=ed)

        signals = pd.DataFrame(index=bb_indicator.index)
        signals["sma"] = [-1 if x > 1.02 else 1 if x < 0.98 else 0 for x in sma_indicator]
        signals["bb"] = [-1 if x > 1.5 else 1 if x < -0.5 else 0 for x in bb_indicator]
        signals["momentum"] = [-1 if x > 1.02 else 1 if x < 0.98 else 0 for x in mom_indicator]
        # net vote: 3 signals each contributing ±1; majority rules
        signals["holding"] = [0 if x == 0 else 1000 if x > 0 else -1000 for x in signals.sum(axis=1)]

        df_trades = pd.DataFrame(index=signals.index, columns=[symbol], data=0, dtype="float64")
        df_trades[symbol].values[1:] = signals["holding"].values[1:] - signals["holding"].values[:-1]

        self.long = df_trades.index[df_trades[symbol] > 0].tolist()
        self.short = df_trades.index[df_trades[symbol] < 0].tolist()

        return df_trades

    def benchmark(
        self,
        symbol: str,
        sd: dt.datetime,
        ed: dt.datetime,
        sv: int = 100_000,
    ) -> pd.Series:
        """Compute a buy-and-hold benchmark for the given symbol and date range.

        Parameters
        ----------
        symbol : str
            Ticker to trade.
        sd : dt.datetime
            Start date.
        ed : dt.datetime
            End date.
        sv : int
            Starting portfolio value

        Returns
        -------
        pd.Series
            Daily portfolio value for the benchmark strategy.
        """
        symbols = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(symbols, dates)
        prices = prices_all[symbols]
        return mk.compute_portvals(prices, start_val = sv, commission = 9.95, impact = 0.005)

    def in_sample(
        self,
        symbol: str = "JPM",
        sd: dt.datetime = dt.datetime(2008, 1, 1),
        ed: dt.datetime = dt.datetime(2009, 12, 31),
        sv: int = 100_000,
        impact: float = 0.0,
        commission: float = 0.0,
        verbose: bool = False,
    ) -> None:
        """Run and plot the in-sample strategy comparison."""
        ms = ManualStrategy(verbose, impact, commission)
        df_trades = ms.testPolicy(symbol, sd, ed, sv)
        port_vals = mk.compute_portvals(df_trades, sv, commission, impact)
        port_vals_bench = mk.compute_portvals(df_trades, start_val = sv, commission = 9.95, impact = 0.005)

        portvals_normalized = port_vals / port_vals.iloc[0]
        portvals_bench_normalized = port_vals_bench / port_vals_bench.iloc[0]

        joined = portvals_normalized.to_frame().join(
            portvals_bench_normalized.to_frame(), lsuffix = "top", rsuffix = "b"
        )
        joined.columns = ["Manual Strategy", "Benchmark"]
        fig = joined.plot(title = "Manual Strategy vs Benchmark (In-Sample)", color = ["red", "green"])
        ymin, ymax = fig.get_ylim()
        plt.vlines(ms.long, ymin, ymax, color = "blue", label = "Enter Long")
        plt.vlines(ms.short, ymin, ymax, color = "black", label = "Enter Short")
        plt.xlim(portvals_bench_normalized.index.min(), portvals_bench_normalized.index.max())
        fig.set_xlabel("Date")
        fig.set_ylabel("Normalized Portfolio Value")
        plt.legend()
        plt.grid(which = "both")
        plt.savefig("Manual Strategy vs Benchmark In-Sample.png")
        plt.clf()

    def out_sample(
        self,
        symbol: str = "JPM",
        sd: dt.datetime = dt.datetime(2010, 1, 1),
        ed: dt.datetime = dt.datetime(2011, 12, 31),
        sv: int = 100_000,
        impact: float = 0.0,
        commission: float = 0.0,
        verbose: bool = False,
    ) -> None:
        # Run and plot the out-of-sample strategy comparison
        ms = ManualStrategy(verbose, impact, commission)
        df_trades = ms.testPolicy(symbol, sd, ed, sv)
        port_vals = mk.compute_portvals(df_trades, sv, commission, impact)
        port_vals_bench = mk.compute_portvals(df_trades, start_val = sv, commission = 9.95, impact = 0.005)

        portvals_normalized = port_vals / port_vals.iloc[0]
        portvals_bench_normalized = port_vals_bench / port_vals_bench.iloc[0]

        joined = portvals_normalized.to_frame().join(
            portvals_bench_normalized.to_frame(), lsuffix = "top", rsuffix = "b"
        )
        joined.columns = ["Manual Strategy", "Benchmark"]
        fig = joined.plot(title = "Manual Strategy vs Benchmark (Out-of-Sample)", color = ["red", "green"])
        ymin, ymax = fig.get_ylim()
        fig.set_xlabel("Date")
        fig.set_ylabel("Normalized Portfolio Value")
        plt.vlines(ms.long, ymin, ymax, color = "blue", label = "Enter Long")
        plt.vlines(ms.short, ymin, ymax, color = "black", label = "Enter Short")
        plt.xlim(portvals_bench_normalized.index.min(), portvals_bench_normalized.index.max())
        plt.legend()
        plt.grid(which = "both")
        plt.savefig("Manual Strategy vs Benchmark Out-Sample.png")
        plt.clf()


def print_stats(
    sd: dt.datetime,
    ed: dt.datetime,
    sv: int,
    symbol: str,
    commission: float,
    impact: float,
    verbose: bool,
) -> None:
    # Print cumulative return, standard deviation, and mean daily return
    ms = ManualStrategy(verbose, impact, commission)
    df_trades = ms.testPolicy(symbol, sd, ed, sv)
    port_vals = mk.compute_portvals(df_trades, sv, commission, impact)
    port_vals_bench = mk.compute_portvals(df_trades, start_val = sv, commission = 9.95, impact = 0.005)

    daily_ret = (port_vals / port_vals.shift(1)) - 1
    cum_ret = (port_vals.iloc[-1] / port_vals.iloc[0]) - 1
    avg_daily_ret = daily_ret.mean()
    std_daily_ret = daily_ret.std()

    daily_ret_b = (port_vals_bench / port_vals_bench.shift(1)) - 1
    cum_ret_b = (port_vals_bench.iloc[-1] / port_vals_bench.iloc[0]) - 1
    avg_daily_ret_b = daily_ret_b.mean()
    std_daily_ret_b = daily_ret_b.std()

    print(f"\n Cumulative Return  — Strategy: {cum_ret:.4f}  |  Benchmark: {cum_ret_b:.4f} ")
    print(f" Std Daily Return   — Strategy: {std_daily_ret:.6f}  |  Benchmark: {std_daily_ret_b:.6f} ")
    print(f" Mean Daily Return  — Strategy: {avg_daily_ret:.6f}  |  Benchmark: {avg_daily_ret_b:.6f} \n")


if __name__ == "__main__":
    learner = ManualStrategy()
    learner.in_sample(
        symbol="JPM",
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 12, 31),
        sv=100_000,
    )
    learner.out_sample(
        symbol="JPM",
        sd=dt.datetime(2010, 1, 1),
        ed=dt.datetime(2011, 12, 31),
        sv=100_000,
    )
    print_stats(
        sd = dt.datetime(2008, 1, 1),
        ed = dt.datetime(2009, 12, 31),
        sv = 100_000,
        symbol = "JPM",
        commission = 0.0,
        impact = 0.0,
        verbose = False,
    )
