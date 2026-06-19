from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd

# Resolve the data directory relative to this file so the default works
# regardless of the working directory the process was launched from.
_REPO_DATA_DIR = Path(__file__).parent.parent.parent / "data"

# In-memory cache: symbol -> full DataFrame loaded from CSV.
# Avoids repeated disk reads within a single process run.
_cache: Dict[str, pd.DataFrame] = {}


def symbol_to_path(symbol: str, base_dir: Optional[str] = None) -> Path:
    """Return the CSV file path for a given ticker.

    Parameters
    ----------
    symbol : str
        Ticker symbol (e.g. "SPY").

    base_dir : str, optional
        Directory containing CSV files. Defaults to the repo-level data/
        directory, or the value of the MARKET_DATA_DIR environment variable.

    Returns
    -------
    Path
        Absolute path to <base_dir>/<symbol>.csv.

    """
    if base_dir is None:
        env_dir = os.environ.get("MARKET_DATA_DIR")
        base_dir = Path(env_dir) if env_dir else _REPO_DATA_DIR
    return Path(base_dir) / f"{symbol}.csv"


def _fetch_and_cache(symbol: str, start: str, end: str, csv_path: Path) -> None:
    """Download adjusted close prices from yfinance and save to CSV.

    If a CSV already exists, the fetch range is expanded to cover the union of
    the existing data and the new request so future calls with different date
    windows don't trigger another network round-trip.

    Requires yfinance: `pip install yfinance` or `pip install strategy-evaluation[data]`.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            f"CSV not found for '{symbol}' and yfinance is not installed.\n"
            "Install it with:  pip install yfinance\n"
            f"Or place {symbol}.csv in the data directory manually."
        )

    # Expand the fetch range to include any data already on disk.
    if csv_path.exists():
        try:
            existing_idx = pd.read_csv(csv_path, index_col="Date", parse_dates=True).index
            if not existing_idx.empty:
                start = min(start, existing_idx.min().strftime("%Y-%m-%d"))
                end = max(end, (existing_idx.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
        except Exception:
            pass

    print(f"[data] Fetching '{symbol}' from yfinance ({start} to {end})...")
    df = yf.download(symbol, start=start, end=end, progress=False)
    if df.empty:
        raise ValueError(f"yfinance returned no data for '{symbol}'. Check the ticker symbol.")

    # yfinance >=1.4 uses a MultiIndex (Price, Ticker); earlier versions use flat columns.
    if isinstance(df.columns, pd.MultiIndex):
        close = df[("Close", symbol)]
    else:
        close = df.get("Adj Close") or df["Close"]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    close.rename("Adj Close").to_csv(csv_path, index_label="Date")
    print(f"[data] Cached {len(close)} rows -> {csv_path}")

    # Invalidate the in-memory cache so the next read picks up the new file.
    _cache.pop(symbol, None)


def _ensure_csv_covers(symbol: str, fetch_start: str, fetch_end: str) -> None:
    """Fetch from yfinance if the local CSV is missing or doesn't cover [start, end)."""
    csv_path = symbol_to_path(symbol)
    if not csv_path.exists():
        _fetch_and_cache(symbol, fetch_start, fetch_end, csv_path)
        return

    # Use 7-day tolerance at each boundary so weekends / public holidays at the
    # edge of the requested range don't incorrectly trigger a re-fetch.
    _TOLERANCE = pd.Timedelta(days=7)
    cached = _cache.get(symbol)
    if cached is None:
        try:
            cached = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
        except Exception:
            _fetch_and_cache(symbol, fetch_start, fetch_end, csv_path)
            return

    idx = cached.index
    needs_update = (
        idx.empty
        or idx.min() > pd.Timestamp(fetch_start) + _TOLERANCE
        or idx.max() < pd.Timestamp(fetch_end) - _TOLERANCE
    )
    if needs_update:
        _fetch_and_cache(symbol, fetch_start, fetch_end, csv_path)


def get_data(
    symbols: List[str],
    dates: pd.DatetimeIndex,
    addSPY: bool = True,
    colname: str = "Adj Close",
) -> pd.DataFrame:
    """Load adjusted close prices for the given symbols over the given dates.

    If a ticker's CSV is not present in the data directory, it is fetched
    automatically from yfinance and cached for future runs. Subsequent calls
    for the same symbol within one process use an in-memory cache.

    SPY is always included as the first column (unless already in the list)
    because it is used to filter out non-trading days from the date range.

    Parameters
    ----------
    symbols : list of str
        Ticker symbols to load.

    dates : pd.DatetimeIndex
        Full date range. Rows with no SPY data (weekends, holidays) are dropped.

    addSPY : bool
        Prepend SPY to the symbol list if it is not already present.

    colname : str
        Column name in the CSV to read (default: "Adj Close").

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by trading date with one column per symbol.

    """
    df = pd.DataFrame(index=dates)
    if addSPY and "SPY" not in symbols:
        symbols = ["SPY"] + list(symbols)

    # yfinance end date is exclusive, so add one day to include the last date.
    fetch_start = dates.min().strftime("%Y-%m-%d")
    fetch_end = (dates.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    for symbol in symbols:
        _ensure_csv_covers(symbol, fetch_start, fetch_end)

        if symbol not in _cache:
            _cache[symbol] = pd.read_csv(
                symbol_to_path(symbol),
                index_col="Date",
                parse_dates=True,
            )

        df_temp = _cache[symbol][[colname]].rename(columns={colname: symbol})
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
