
import pandas as pd
from util import get_data
import numpy as np


def compute_portvals(df_trades, start_val=1000000, commission=9.95, impact=0.005):
    if isinstance(df_trades, str):
        df_trades = pd.read_csv(df_trades, index_col="Date", parse_dates=True)

    start_date = df_trades.index.min()
    end_date = df_trades.index.max()
    dates = pd.date_range(start_date, end_date)

    symbol = df_trades.columns[0]

    orders = pd.DataFrame()
    orders['Date'] = df_trades.index
    orders['Symbol'] = symbol
    orders['Order'] = ["BUY" if x > 0 else "SELL" for x in df_trades.values]
    orders['Shares'] = abs(df_trades.values)
    orders.set_index('Date', inplace=True)

    df_p = get_data([symbol], dates)
    df_p = df_p[[symbol]]
    df_p['CASH'] = 1.00

    df_trades = df_p.copy()
    df_trades[:] = 0

    for index, row in orders.iterrows():
        if index in df_p.index:  # index = date
            if row['Order'] == 'BUY':
                df_trades.loc[index, row['Symbol']] += row['Shares']
                df_trades.loc[index, 'CASH'] += -1 * (row['Shares'] * df_p.loc[index, row['Symbol']] * (1 + impact)) - commission
            elif row['Order'] == 'SELL':
                df_trades.loc[index, row['Symbol']] -= row['Shares']
                df_trades.loc[index, 'CASH'] += row['Shares'] * df_p.loc[index, row['Symbol']] * (1 - impact) - commission

    trades_copy = df_trades.copy()
    trades_copy.loc[start_date, 'CASH'] += start_val
    trades_copy = trades_copy.cumsum(axis=0)

    sum = df_p * trades_copy
    port_vals = sum.sum(axis=1)

    return port_vals
