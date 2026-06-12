

import matplotlib.pyplot as plt
import marketsimcode as mksim


def graph(plotFig, fig_name, title, x_label, y_label):
    plotFig.plot()
    plt.legend()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(fig_name)
    plt.clf()


def experiment2(s_trades, sv, commission):
    sl1 = mksim.compute_portvals(s_trades, start_val=sv, commission=commission, impact=0.000)
    sl2 = mksim.compute_portvals(s_trades, start_val=sv, commission=commission, impact=0.005)
    sl3 = mksim.compute_portvals(s_trades, start_val=sv, commission=commission, impact=0.010)

    df = s_trades.copy()
    df['Impact=0.000'] = sl1 / sl1.iloc[0]
    df['Impact=0.005'] = sl2 / sl2.iloc[0]
    df['Impact=0.010'] = sl3 / sl3.iloc[0]
    df = df.drop(df.columns[0], axis=1)

    # plotFig, fig_name, title, x_label, y_label
    graph(df, "Experiment2", "Stock Prices", "Date", "Normalized Portfolio Return for In-Sample Data")
