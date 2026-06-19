from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from .util import get_data


def compute_portvals(
    df_trades: Union[str, pd.DataFrame],
    start_val: float = 1_000_000,
    commission: float = 9.95,
    impact: float = 0.005,
) -> pd.Series:
    """
    Simulate a portfolio and return its daily value over time

    Accepts either a trade schedule DataFrame or a path to a CSV file with
    the same format. The trade schedule must have a DatetimeIndex and a
    single column containing signed share counts: positive values are buys,
    negative values are sells, and zero means no trade on that day.

    Transaction costs are applied per trade:
    - Buy:  price * shares * (1 + impact) + commission
    - Sell: price * shares * (1 - impact) - commission  (impact lowers proceeds)

    Parameters
    ----------
    df_trades : str or pd.DataFrame
        Trade schedule. If a string, treated as a CSV file path with a 'Date' column used as the index

    start_val : float
        Initial cash

    commission : float
        Flat fee in dollars charged per executed trade

    impact : float
        Market impact as a fraction of share price, applied on each trade

    Returns
    -------
    pd.Series
        Daily total portfolio value (cash + holdings) indexed by date

    """
    if isinstance(df_trades, str):
        df_trades = pd.read_csv(df_trades, index_col = "Date", parse_dates = True)

    start_date = df_trades.index.min()
    end_date = df_trades.index.max()
    dates = pd.date_range(start_date, end_date)
    symbol = df_trades.columns[0]

    # convert the signed-share-count column into labeled BUY/SELL orders
    orders = pd.DataFrame()
    orders["Date"] = df_trades.index
    orders["Symbol"] = symbol
    orders["Order"] = ["BUY" if x > 0 else "SELL" for x in df_trades.values]
    orders["Shares"] = abs(df_trades.values)
    orders.set_index("Date", inplace=True)

    # price data — includes a synthetic CASH column always worth $1.00
    df_p = get_data([symbol], dates)
    df_p = df_p[[symbol]]
    df_p["CASH"] = 1.00

    # delta ledger - each row records the share and cash change on that date
    ledger = df_p.copy()
    ledger[:] = 0

    for date, row in orders.iterrows():
        if date not in df_p.index:
            continue

        price = df_p.loc[date, row["Symbol"]]
        shares = row["Shares"]
        if shares == 0:
            continue

        if row["Order"] == "BUY":
            ledger.loc[date, row["Symbol"]] += shares
            ledger.loc[date, "CASH"] += -(shares * price * (1 + impact)) - commission
            
        elif row["Order"] == "SELL":
            ledger.loc[date, row["Symbol"]] -= shares
            ledger.loc[date, "CASH"] += (shares * price * (1 - impact)) - commission

    # cumulative sum converts daily deltas into running holdings
    holdings = ledger.copy()
    holdings.loc[start_date, "CASH"] += start_val
    holdings = holdings.cumsum(axis=0)

    # dot product of holdings with prices gives total portfolio value each day
    portfolio_value = (df_p * holdings).sum(axis=1)

    return portfolio_value
