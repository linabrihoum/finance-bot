

import matplotlib.pyplot as plt
import datetime as dt
import StrategyLearner as sl
import ManualStrategy as ms
import marketsimcode as mk


def author():
    return 'lbrihoum3'


def graph(plotFig, fig_name, title, x_label, y_label):
    plotFig.plot()
    plt.legend()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(fig_name)
    plt.clf()


def experiment1(symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000, impact = 0.000, commission = 0):
    manual_learner = ms.ManualStrategy()
    m_trades = manual_learner.testPolicy(symbol=symbol, sd=sd, ed=ed)

    learner = sl.StrategyLearner(verbose=False, impact=impact)
    learner.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)

    m_portv = mk.compute_portvals(manual_learner.testPolicy(symbol=symbol, sd=sd, ed=ed), start_val=sv, commission=commission, impact=impact)
    sl_portv = mk.compute_portvals(learner.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv), start_val=sv,
                                      commission=commission, impact=impact)

    b_trades = m_trades.copy()
    b_trades[:] = 0
    b_trades.ix[0,0] = 1000
    b_trades.ix[-1,0] = -1000

    df = m_trades.copy()
    df['Manual Strategy'] = (m_portv[0:] / m_portv.ix[0])
    df['Strategy Learner'] = (sl_portv[0:] / sl_portv.ix[0])
    df['Benchmark'] = ((mk.compute_portvals(b_trades, start_val=sv, commission=commission, impact=impact))[0:] / (mk.compute_portvals(b_trades, start_val=sv, commission=commission, impact=impact)).ix[0])
    df = df.drop(df.columns[0], axis=1)

    # plotFig, fig_name, title, x_label, y_label
    graph(df, "Experiment1", "Stock Prices", "Date", "Normalized Portfolio Return")