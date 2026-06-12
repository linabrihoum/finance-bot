

import pandas as pd
import datetime as dt
import util as ut
import indicators as ind
import marketsimcode as mk

import matplotlib.pyplot as plt


class ManualStrategy(object):
    # constructor
    def __init__(self, verbose=False, impact=0.0, commission=9.95):
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.window = 10
        self.short = []
        self.long = []

    def testPolicy(self, symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
        bb_indicator = ind.bollinger_band(symbol=[symbol], sd=sd, ed=ed)
        sma_indicator = ind.simple_moving_avg(symbol=[symbol], sd=sd, ed=ed)
        mom_indicator = ind.momentum(symbol=[symbol], sd=sd, ed=ed)

        signals = pd.DataFrame(index=bb_indicator.index)
        signals['sma'] = [-1 if x > 1.02 else 1 if x < 0.98 else 0 for x in sma_indicator]
        signals['bb'] = [-1 if x > 1.5 else 1 if x < -0.5 else 0 for x in bb_indicator]
        signals['momentum'] = [-1 if x > 1.02 else 1 if x < 0.98 else 0 for x in mom_indicator]
        signals['holding'] = [0 if x == 0 else 1000 if x > 0 else -1000 for x in signals.sum(axis=1)]

        df_trades = pd.DataFrame(index=signals.index, columns=[symbol], data=0, dtype='float64')
        df_trades[symbol].values[1:] = signals['holding'].values[1:] - signals['holding'].values[:-1]

        self.long = df_trades.index[df_trades[symbol] > 0].tolist()
        self.short = df_trades.index[df_trades[symbol] < 0].tolist()

        return df_trades

    def benchmark(self, symbol, sd, ed, sv=100000):
        symbols = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(symbols, dates)
        prices = prices_all[symbols]

        portvals = mk.compute_portvals(prices, start_val=sv, commission=9.95, impact=0.005)
        return portvals

    def MainManualStrategy(self):
        sd = '2008-1-1'
        ed = '2009-12-31'
        sv = 100000
        symbol = 'JPM'
        commission = 0
        impact = 0
        verbose = False

        ms = ManualStrategy()

        print_stats(sd, ed, sv, symbol, commission, impact, verbose)
        ms.out_sample(symbols="JPM", sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv=100000, impact=0.000, commission=0, verbose=False)
        ms.in_sample(symbols="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000, impact=0.000, commission=0, verbose=False)

    def in_sample(self, symbols = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000, impact = 0.000, commission = 0, verbose = False):
        ms = ManualStrategy(verbose, impact, commission)
        df_trades = ms.testPolicy(symbols, sd, ed, sv)
        port_vals = mk.compute_portvals(df_trades, sv, commission, impact)

        port_vals_bench = mk.compute_portvals(df_trades, start_val=sv, commission=9.95, impact=0.005)

        portvals_normalized = port_vals / port_vals.iloc[0,]
        portvals_bench_normalized = port_vals_bench / port_vals_bench.iloc[0,]

        joined = portvals_normalized.to_frame().join(portvals_bench_normalized.to_frame(), lsuffix="top", rsuffix="b")
        joined.columns = ["Manual Strategy", "Benchmark"]
        fig = joined.plot(title="Manual Strategy vs Benchmark In-Sample", color=["red", "green"])
        ymin, ymax = fig.get_ylim()
        plt.vlines(self.long, ymin, ymax, color='blue', label='Long')
        plt.vlines(self.short, ymin, ymax, color='black', label='Short')
        plt.xlim(portvals_bench_normalized.index.min(), portvals_bench_normalized.index.max())
        fig.set_xlabel("Date")
        fig.set_ylabel("Price")
        plt.grid(which='both')
        plt.savefig("Manual Strategy vs Benchmark In-Sample")
        plt.clf()

    def out_sample(self, symbols = "JPM", sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv = 100000, impact = 0.000, commission = 0, verbose = False):
        ms = ManualStrategy(verbose, impact, commission)
        df_trades = ms.testPolicy(symbols, sd, ed, sv)
        port_vals = mk.compute_portvals(df_trades, sv, commission, impact)

        port_vals_bench = mk.compute_portvals(df_trades, start_val=sv, commission=9.95, impact=0.005)

        portvals_normalized = port_vals / port_vals.iloc[0,]
        portvals_bench_normalized = port_vals_bench / port_vals_bench.iloc[0,]

        joined = portvals_normalized.to_frame().join(portvals_bench_normalized.to_frame(), lsuffix="top", rsuffix="b")
        joined.columns = ["Manual Strategy", "Benchmark"]
        fig = joined.plot(title="Manual Strategy vs Benchmark Out-Sample", color=["red", "green"])
        ymin, ymax = fig.get_ylim()
        fig.set_xlabel("Date")
        fig.set_ylabel("Price")
        plt.vlines(self.long, ymin, ymax, color='blue', label='Long')
        plt.vlines(self.short, ymin, ymax, color='black', label='Short')
        plt.xlim(portvals_bench_normalized.index.min(), portvals_bench_normalized.index.max())
        plt.grid(which='both')
        plt.savefig("Manual Strategy vs Benchmark Out-Sample")
        plt.clf()


def print_stats(sd, ed, sv, symbols, commission, impact, verbose):
    ms = ManualStrategy(verbose, impact, commission)
    df_trades = ms.testPolicy(symbols, sd, ed, sv)
    port_vals = mk.compute_portvals(df_trades, sv, commission, impact)

    port_vals_bench = mk.compute_portvals(df_trades, start_val=sv, commission=9.95, impact=0.005)

    daily_ret = (port_vals / port_vals.shift(1)) - 1
    cum_ret = (port_vals[-1] / port_vals[0]) - 1
    avg_daily_ret = daily_ret.mean()
    std_daily_ret = daily_ret.std()

    daily_ret_b = (port_vals_bench / port_vals_bench.shift(1)) - 1
    cum_ret_b = (port_vals_bench[-1] / port_vals_bench[0]) - 1
    avg_daily_ret_b = daily_ret_b.mean()
    std_daily_ret_b = daily_ret_b.std()

    print()
    print("Cumulative Return of Manual Strategy: " + str(cum_ret))
    print("Cumulative Return of Benchmark: " + str(cum_ret_b))
    print()
    print("Standard Deviation of Manual Strategy: " + str(std_daily_ret))
    print("Standard Deviation of Benchmark : " + str(std_daily_ret_b))
    print()
    print("Average Daily Return of Manual Strategy: " + str(avg_daily_ret))
    print("Average Daily Return of Benchmark : " + str(avg_daily_ret_b))


if __name__ == '__main__':

    manualLearner = ManualStrategy()
    manualLearner.MainManualStrategy()

