from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt

import pandas as pd

# Resolve the data directory relative to this file so the default works
# regardless of the working directory the process was launched from.
_REPO_DATA_DIR = str(Path(__file__).parent.parent.parent / "data")


def symbol_to_path(symbol: str, base_dir: Optional[str] = None) -> str:
    """
    Return the CSV file path for a given ticker

    Parameters
    ----------
    symbol : str
        Ticker symbol (ex. "SPY")

    base_dir : str, optional
        Directory containing CSV files. Defaults to the repo-level data/
        directory, or the value of the MARKET_DATA_DIR environment variable

    Returns
    -------
    str
        Absolute or relative path to <base_dir>/<symbol>.csv

    """
    if base_dir is None:
        base_dir = os.environ.get("MARKET_DATA_DIR", _REPO_DATA_DIR)
    return os.path.join(base_dir, f"{symbol}.csv")


def get_data(
    symbols: List[str],
    dates: pd.DatetimeIndex,
    addSPY: bool = True,
    colname: str = "Adj Close",
) -> pd.DataFrame:
    """
    Load adjusted close prices for the given symbols over the given dates.

    SPY is always included as the first column (unless already in the list)
    because it is used to filter out non-trading days from the date range.

    Parameters
    ----------
    symbols : list of str
        Ticker symbols to load

    dates : pd.DatetimeIndex
        Full date range. Rows with no SPY data (weekends, holidays) are dropped

    addSPY : bool
        Prepend SPY to the symbol list if it is not already present

    colname : str
        Column name in the CSV to read (default: "Adj Close")

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by trading date with one column per symbol

    """

    df = pd.DataFrame(index=dates)
    if addSPY and "SPY" not in symbols:
        symbols = ["SPY"] + list(symbols)

    for symbol in symbols:
        df_temp = pd.read_csv(
            symbol_to_path(symbol),
            index_col = "Date",
            parse_dates = True,
            usecols = ["Date", colname],
            na_values = ["nan"],
        )
        df_temp = df_temp.rename(columns = {colname: symbol})
        df = df.join(df_temp)
        if symbol == "SPY":
            df = df.dropna(subset=["SPY"])

    return df


def plot_data(
    df: pd.DataFrame,
    title: str = "Stock prices",
    xlabel: str = "Date",
    ylabel: str = "Price",
) -> None:
    # Plot & display a DataFrame of price data

    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()