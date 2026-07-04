# -*- coding: utf-8 -*-
"""
Central configuration for the regime-switching paper trading system.
Change parameters here — daily_update.py and regime_lib.py both import from this file.
"""

# Universe
TICKERS = ['AAPL', 'JPM', 'XOM', 'JNJ', 'WMT']
N_ASSETS = len(TICKERS)

# History window pulled from Yahoo Finance. The HMM/regime portfolios are
# retrained every day on an EXPANDING window from this start date through
# "today" — there is no separate frozen model anymore.
TRAIN_START = '2005-01-01'

# Minimum number of valid feature-days required before we'll attempt the
# train/test split + HMM fit (guards against nonsense fits very early on).
MIN_HISTORY_DAYS = 500

# Strategy parameters (same meaning as in the original backtest)
VOL_TARGET = 0.15          # pick ONE target for live paper trading (was a list in the backtest)
VOL_LOOKBACK = 60
NORM_LOOKBACK = 252
MAX_LEVERAGE = 2.0
TCOST = 0.001
RISK_FREE_RATE = 0.03

# Per-asset weight cap used in BOTH the max-Sharpe (bull) and min-variance
# (bear/sideways) optimizers, to tame the instability of unconstrained
# max-Sharpe. Must satisfy MAX_WEIGHT * N_ASSETS >= 1 (else long-only +
# fully-invested is infeasible). 0.35 allows meaningful over/underweight vs.
# equal-weight (0.20) while forcing diversification across at least ~3 names.
MAX_WEIGHT = 0.35
assert MAX_WEIGHT * N_ASSETS >= 1.0, "MAX_WEIGHT too low to be feasible long-only/fully-invested"

# Paths (relative to repo root)
# DATA_DIR lives under docs/ so GitHub Pages (serving the docs/ folder) can
# serve the CSV logs directly to the dashboard via a simple relative fetch.
DATA_DIR = "docs/data"
STATE_PATH = f"{DATA_DIR}/state.json"
EQUITY_LOG_PATH = f"{DATA_DIR}/equity_log.csv"
WEIGHTS_LOG_PATH = f"{DATA_DIR}/weights_log.csv"
