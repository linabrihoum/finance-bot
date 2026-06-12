# Refactor Roadmap

Tracking the transformation of this project from a course submission into a
professional portfolio repository.

---

## Phase 1 — Remove Course Artifacts & Fix Critical Bugs

- [x] Remove GT student headers from all files
- [ ] Strip GT whitespace fingerprints from all .py files
- [ ] Remove `author()` methods returning GT user ID
- [ ] Delete `grade_strategy_learner.py`
- [ ] Fix `momentum` indicator (currently computes SMA, not rate-of-change)
- [ ] Fix broken `isinstance(type(...), type(""))` in `marketsimcode.py`
- [ ] Fix dead code in `ManualStrategy.testPolicy` (indicators computed twice, df_trades overwritten)
- [ ] Fix all deprecated `.ix` → `.loc` / `.iloc` (affects 5 files)
- [ ] Fix deprecated `fillna(method='ffill')` → `.ffill()` / `.bfill()`
- [ ] Add `np.random.seed()` / `random.seed()` to `QLearner` for reproducibility

---

## Phase 2 — Code Quality

- [ ] Add type hints to all public functions and methods
- [ ] Rename `temp` → `position` in `ManualStrategy`
- [ ] Rename `sum` → `portfolio_value` in `marketsimcode`
- [ ] Rename `sl1`, `sl2`, `sl3` → descriptive names in `experiment2`
- [ ] Extract duplicate `graph()` function (defined identically in 3 files) into `plotting.py`
- [ ] Add class and method docstrings
- [ ] Fix docstring typo in `QLearner.query`: "The ne state" → correct description of `r`
- [ ] Add `duplicates='drop'` to `pd.qcut` calls in `StrategyLearner` to handle low-variance data

---

## Phase 3 — Repository Structure

- [ ] Create `src/strategy_evaluation/` layout
- [ ] Rename files to `snake_case` (e.g., `QLearner.py` → `q_learner.py`)
- [ ] Create `scripts/` (move `testproject.py` → `scripts/run_experiments.py`)
- [ ] Create `data/README.md` documenting required CSV format + yfinance alternative
- [ ] Create `outputs/` with `.gitkeep`

---

## Phase 4 — Reproducibility

- [ ] Update `environment.yml` to Python 3.11 with current dependency versions
- [ ] Add `pyproject.toml` with project metadata and dev dependencies
- [ ] Add `config.yaml` externalizing all hyperparameters (window, alpha, gamma, rar, radr, etc.)

---

## Phase 5 — Tests

- [ ] `tests/test_q_learner.py` — Q-table updates, RAR decay, Dyna-Q, convergence
- [ ] `tests/test_indicators.py` — SMA window, BBP midpoint, momentum correctness
- [ ] `tests/test_market_simulator.py` — buy/sell cash, commission, zero-trade baseline

---

## Phase 6 — README

- [ ] Project overview and problem statement
- [ ] Approach: indicators, state space design, Q-learning with Dyna-Q, reward shaping
- [ ] Results (charts)
- [ ] Installation and usage
- [ ] Experiment design
- [ ] Known limitations and future work