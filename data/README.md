# Data

Price data is **fetched automatically** from Yahoo Finance on first run and cached here as CSV files. No manual setup is required.

## How it works

When `scripts/run_experiments.py` runs, `util.get_data()` checks whether each ticker's CSV exists and covers the requested date range. If not, it calls yfinance, downloads the data, and saves it here. Subsequent runs load from the local cache with no network calls.

```
$ python scripts/run_experiments.py
[data] Fetching 'SPY' from yfinance (2008-01-01 to 2010-01-01)...
[data] Cached 505 rows -> .../data/SPY.csv
[data] Fetching 'JPM' from yfinance (2008-01-01 to 2010-01-01)...
[data] Cached 505 rows -> .../data/JPM.csv
Running Experiment 1...
```

These CSV files are excluded from git (see `.gitignore`).

## Manual override

If you need to supply your own data (e.g. behind a firewall), place a file named `<TICKER>.csv` in this directory with at least these columns:

```
Date,Adj Close
2008-01-02,41.560001
2008-01-03,41.040001
...
```

- `Date` must be parseable as a datetime (used as the index).
- `Adj Close` is the adjusted closing price used for all calculations.
- `SPY.csv` is always required — it aligns the date index to actual trading days.

## Custom data directory

Override the default location with the `MARKET_DATA_DIR` environment variable:

```bash
export MARKET_DATA_DIR=/path/to/your/data
python scripts/run_experiments.py
```
