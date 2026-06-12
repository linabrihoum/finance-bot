

import pandas as pd
import datetime as dt
import util as ut
import matplotlib.pyplot as plt

def graph(plotFig, fig_name, title, x_label, y_label):
    plotFig.plot()
    plt.legend()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(fig_name)
    plt.clf()

def simple_moving_avg(symbol=['JPM'], sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,12,31), window = 10):
    prices = ut.get_data(symbol,pd.date_range(sd,ed))
    prices = prices[symbol]
    prices = prices[0:]/prices.iloc[0] # normalized prices
    simple_moving_avg = prices.rolling(window=window).mean() #SMA
    prices['SMA'] = simple_moving_avg #adding SMA to prices df
    prices["SMA Ratio"] = prices.values[:, 0] / prices.values[:, 1]

    return prices["SMA Ratio"]

def bollinger_band(symbol=['JPM'], sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,12,31), window = 10):
    prices = ut.get_data(symbol, pd.date_range(sd,ed))
    prices = prices[symbol]
    prices = prices[0:]/prices.iloc[0] # normalized prices
    simple_moving_avg = prices.rolling(window=window).mean()
    rolling_std_dev = prices.rolling(window=window).std()
    bottom_band = simple_moving_avg - 2 * rolling_std_dev
    top_band = simple_moving_avg + 2 * rolling_std_dev
    bbp = (prices - bottom_band) / (top_band - bottom_band) #bollinger band percentage for buying and selling
    prices['Upper Band'] = top_band
    prices['Lower Band'] = bottom_band

    prices['BBP Ratio'] = bbp

    return prices['BBP Ratio']

def momentum(symbol=['JPM'], sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,12,31), window = 10):
    prices = ut.get_data(symbol, pd.date_range(sd, ed))
    prices = prices[symbol]
    prices = prices[0:] / prices.iloc[0]  # normalized prices

    # rate of change: (price[t] - price[t-n]) / price[t-n]
    moment = (prices - prices.shift(window)) / prices.shift(window)
    prices['Momentum'] = moment
    prices["Momentum Ratio"] = moment

    return prices['Momentum Ratio']


def graph_simple_moving_avg(prices, window):

    prices = prices[0:]/prices.iloc[0]
    simple_moving_avg = prices.rolling(window=window).mean()
    prices['SMA'] = simple_moving_avg
    prices["SMA Ratio"] = prices.values[:, 0] / prices.values[:, 1]

    # (plotFig, fig_name, title, x_label, y_label)
    graph(prices, "Simple Moving Average.png", "Stock Prices", "Date", "Prices")

    return prices["SMA Ratio"]


def graph_bollinger_band(prices, window):

    prices = prices[0:]/prices.iloc[0] # normalized prices
    simple_moving_avg = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    bottom_band = simple_moving_avg - 2 * rolling_std
    top_band = simple_moving_avg + 2 * rolling_std
    bbp = (prices - bottom_band) / (top_band - bottom_band)
    prices['Upper Band'] = top_band
    prices['Lower Band'] = bottom_band

    # (plotFig, fig_name, title, x_label, y_label)
    graph(prices, "BollingerBand.png", "Stock Prices", "Date", "Prices")

    prices['BBP Ratio'] = bbp
    return prices['BBP Ratio']


def graph_momentum(prices, window):

    prices = prices[0:] / prices.iloc[0]  # normalized prices

    moment = (prices - prices.shift(window)) / prices.shift(window)
    prices['Momentum'] = moment
    prices["Momentum Ratio"] = moment

    # (plotFig, fig_name, title, x_label, y_label)
    graph(prices, "Momentum.png", "Stock Prices", "Date", "Prices")

    return prices['Momentum Ratio']


if __name__ == '__main__':
    symbol = ['JPM']
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    window = 10

    prices = ut.get_data(symbol, pd.date_range(sd, ed))
    prices = prices[symbol]

    graph_bollinger_band(prices, window)
    graph_simple_moving_avg(prices, window)
