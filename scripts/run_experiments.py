"""
Run both experiments and generate charts

"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

# Allow running from the repo root without installing package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from strategy_evaluation import StrategyLearner, ManualStrategy
from strategy_evaluation.experiment1 import experiment1
from strategy_evaluation.experiment2 import experiment2
from strategy_evaluation.manual_strategy import print_stats


# In-sample has been seen during training
IN_SAMPLE_START = dt.datetime(2008, 1, 1)
IN_SAMPLE_END = dt.datetime(2009, 12, 31)

# Out-of-sample refer to whether data was
OUT_SAMPLE_START = dt.datetime(2010, 1, 1)
OUT_SAMPLE_END = dt.datetime(2011, 12, 31)

SYMBOL = "JPM"
START_VAL = 100_000


def main() -> None:
    print("Running Experiment 1: Manual Strategy vs Strategy Learner vs Benchmark...")
    experiment1(
        symbol = SYMBOL,
        sd = IN_SAMPLE_START,
        ed = IN_SAMPLE_END,
        sv = START_VAL,
        impact = 0.0,
        commission = 0.0,
    )

    print("Running Experiment 2: Market impact sensitivity analysis...")
    learner = StrategyLearner(verbose = False, impact = 0.0)
    learner.add_evidence(symbol = SYMBOL, sd = IN_SAMPLE_START, ed = IN_SAMPLE_END, sv = START_VAL)
    s_trades = learner.testPolicy(symbol = SYMBOL, sd = IN_SAMPLE_START, ed = IN_SAMPLE_END)
    experiment2(s_trades, sv = START_VAL, commission = 0.0)

    print("Running Manual Strategy in-sample and out-of-sample...")
    manual = ManualStrategy()
    manual.in_sample(symbol = SYMBOL, sd = IN_SAMPLE_START, ed = IN_SAMPLE_END, sv = START_VAL)
    manual.out_sample(symbol = SYMBOL, sd = OUT_SAMPLE_START, ed = OUT_SAMPLE_END, sv = START_VAL)

    print("\n In-sample performance statistics: ")
    print_stats(
        sd = IN_SAMPLE_START,
        ed = IN_SAMPLE_END,
        sv = START_VAL,
        symbol = SYMBOL,
        commission = 0.0,
        impact = 0.0,
        verbose = False,
    )

    print("Charts have been saved.")


if __name__ == "__main__":
    main()