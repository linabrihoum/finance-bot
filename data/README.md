# Data

This is where the market data CSV files should be placed. It is excluded from this repo as data files must be obtained separately.

## Expected Format

Each symbol requires a CSV file named `<TICKER>.csv` (e.g. `JPM.csv`, `SPY.csv`). The file must have at least the following columns:

```
Date, Adj Close
2008-01-02, 41.560001
2008-01-03, 41.040001
...
```

- `Date` must be parseable as a datetime and is used as the index.
- `Adj Close` is the adjusted closing price used for all calculations.
- The `SPY` file is always required — it is used to align the date index to actual trading days.

## Getting Data

### Option A — yfinance (recommended)

```python
import yfinance as yf

for ticker in ["JPM", "SPY", "AAPL"]:
    df = yf.download(ticker, start="2007-01-01", end="2012-12-31")
    df[["Adj Close"]].to_csv(f"data/{ticker}.csv", index_label="Date")
```

### Option B — Manual download

Download adjusted close price history from any financial data provider (Yahoo Finance, Quandl, etc.) and save in the format above.

## Setting a Custom Data Directory

The data path defaults to `../data/` relative to the source files. Override it by setting the `MARKET_DATA_DIR` environment variable:

```bash
export MARKET_DATA_DIR=/path/to/your/data
python scripts/run_experiments.py
```