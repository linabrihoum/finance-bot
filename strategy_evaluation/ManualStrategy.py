"""
Project 8: Strategy Evaluation
Student Name: Lina Brihoum
GT User ID: lbrihoum3
GT ID: 903261368
ML4T CS7646
ManualStrategy.py
"""

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

    def author(self):
        return 'lbrihoum3'

    def testPolicy(self, symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000):
        df = ut.get_data([symbol], pd.date_range(sd, ed))
        p_df = df[[symbol]]
        p_df = p_df.fillna(method='ffill')
        p_df = p_df.fillna(method='bfill')

        df_trades = df[['JPM']]
        df_trades = df_trades.rename(columns={'JPM': symbol}).astype({symbol: 'int32'})
        df_trades[:] = 0

        p_df = p_df / p_df.iloc[0]

        bb_indicator = ind.bollinger_band(symbol=[symbol], sd=sd, ed=ed)
        sma_indicator = ind.simple_moving_avg(symbol=[symbol], sd=sd, ed=ed)
        mom_indicator = ind.momentum(symbol=[symbol], sd=sd, ed=ed)

        p_df['SMA'] = ind.simple_moving_avg(symbol=[symbol], sd=sd, ed=ed)
        p_df['BBP'] = ind.bollinger_band(symbol=[symbol], sd=sd, ed=ed)
        p_df['Momentum'] = ind.momentum(symbol=[symbol], sd=sd, ed=ed)

        temp = 0

        for date, row in p_df.iterrows():
            # Buy
            if row['BBP'] <= 0.25 and row['SMA'] > 0.95 and row['Momentum'] < 0:
                if temp == 0:
                    temp = 1000
                    df_trades.loc[date, symbol] = 1000
                    self.short.append(date)
                elif temp == -1000:
                    temp = 1000
                    df_trades.loc[date, symbol] = 2000
                    self.long.append(date)

            # Sell
            elif row['SMA'] >= 0.9 and row['BBP'] < 1.05 and row['Momentum'] > 0:
                if temp == 0:
                    temp = -1000
                    df_trades.loc[date, symbol] = -1000
                    self.short.append(date)
                elif temp == 1000:
                    temp = -1000
                    df_trades.loc[date, symbol] = -2000
                    self.long.append(date)

        execute = pd.DataFrame(index=bb_indicator.index)

        execute['sma'] = [-1 if x > 1.02 else 1 if x < 0.98 else 0 for x in sma_indicator]
        execute['bb'] = [-1 if x > 1.5 else 1 if x < -.5 else 0 for x in bb_indicator]
        execute['momentum'] = [-1 if x > 1.02 else 1 if x < .98 else 0 for x in mom_indicator]
        execute['holding'] = [0 if x == 0 else 1000 if x > 0 else -1000 for x in execute.sum(axis=1)]

        execute[symbol] = 0
        execute[symbol].values[1:] = execute['holding'].values[1:] - execute['holding'].values[:-1]
        df_trades = execute[[symbol]]

        return df_trades

    def benchmark(self, symbol, sd, ed, sv=100000):
        symbol = ['JPM']

        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(symbol, dates)  # automatically adds SPY
        prices = prices_all[symbol]

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

        # change the date above before running this for out-sample
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

