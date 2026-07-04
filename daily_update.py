# -*- coding: utf-8 -*-
"""
Run this once per trading day (after market close) to advance the paper
trading track record by one day. Safe to re-run multiple times a day (it
no-ops if there's no new trading day) and safe to run after a missed day or
two (it replays every missed trading day in order, so nothing is skipped —
though note each replayed day triggers its own full HMM refit, so a long gap
will take a while to catch up).

Every day, the regime model itself is refit from scratch on an EXPANDING
window of history through that day, split in half (see regime_lib.fit_regime_model
for why): first half fits the HMM, second half labels regimes and optimizes
the three regime-specific portfolios. There is no persisted/frozen model
between runs — only the trading state below is persisted.

State (docs/data/state.json) holds only:
    prev_weights       - weights decided as of the last processed day
                          (these are the weights that earn the NEXT day's return)
    pending_cost       - transaction cost already charged against prev_weights,
                          to be deducted when its return is realized
    equity             - cumulative paper-trading equity (starts at 1.0)
    last_decision_date - last trading day processed

Logs:
    docs/data/equity_log.csv   - one row per REALIZED trading day: date, return, equity
    docs/data/weights_log.csv  - one row per DECISION day: date, regime probs, weights,
                                  plus a few diagnostics about that day's HMM fit
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from config import (
    EQUITY_LOG_PATH, MAX_LEVERAGE, MIN_HISTORY_DAYS, STATE_PATH,
    TCOST, TICKERS, VOL_LOOKBACK, VOL_TARGET, WEIGHTS_LOG_PATH,
)
from regime_lib import build_features, download_prices, fit_regime_model


def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            s = json.load(f)
        s["prev_weights"] = np.array(s["prev_weights"]) if s["prev_weights"] is not None else None
        return s
    return {"prev_weights": None, "pending_cost": 0.0, "equity": 1.0, "last_decision_date": None}


def save_state(state):
    s = dict(state)
    s["prev_weights"] = state["prev_weights"].tolist() if state["prev_weights"] is not None else None
    with open(STATE_PATH, "w") as f:
        json.dump(s, f, indent=2)


def append_csv(path, row: dict):
    df_row = pd.DataFrame([row])
    if os.path.exists(path):
        df_row.to_csv(path, mode="a", header=False, index=False)
    else:
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        df_row.to_csv(path, mode="w", header=True, index=False)


def main():
    state = load_state()
    n = len(TICKERS)

    print("Downloading latest prices...")
    prices = download_prices()
    returns, hmm_features, rolling_mean, rolling_std, valid_dates = build_features(prices)

    last_decision_date = state["last_decision_date"]
    if last_decision_date is None:
        dates_to_process = valid_dates[-1:]
    else:
        last_ts = pd.Timestamp(last_decision_date)
        dates_to_process = valid_dates[valid_dates > last_ts]

    if len(dates_to_process) == 0:
        print(f"No new trading day since {last_decision_date}. Nothing to do.")
        return

    prev_weights = state["prev_weights"]
    pending_cost = state["pending_cost"]
    equity = state["equity"]

    for d in dates_to_process:
        # --- 0. Slice the full pull down to "everything known as of d" ---
        sub_dates = valid_dates[valid_dates <= d]
        if len(sub_dates) < MIN_HISTORY_DAYS:
            print(f"{d.date()}  only {len(sub_dates)} valid feature-days available "
                  f"(< MIN_HISTORY_DAYS={MIN_HISTORY_DAYS}) — skipping decision.")
            continue

        sub_returns = returns.loc[sub_dates]
        sub_features = hmm_features.loc[sub_dates]
        sub_mean = rolling_mean.loc[sub_dates]
        sub_std = rolling_std.loc[sub_dates]

        # --- 1. Retrain today: split-half HMM fit + regime portfolio optimization ---
        fit = fit_regime_model(sub_returns, sub_features, sub_mean, sub_std, sub_dates)
        bull, bear, sideways = fit["bull"], fit["bear"], fit["sideways"]
        w_bull = fit["weights_by_regime"]["bull"]
        w_bear = fit["weights_by_regime"]["bear"]
        w_side = fit["weights_by_regime"]["sideways"]
        probs = fit["probs_today"]

        w_raw = probs[bull] * w_bull + probs[bear] * w_bear + probs[sideways] * w_side

        # --- 2. Vol-target using the trailing realized covariance (unchanged from before) ---
        hist = returns.shift(1).loc[:d].tail(VOL_LOOKBACK)
        if len(hist) == VOL_LOOKBACK:
            cov_t = LedoitWolf().fit(hist.values).covariance_
            vol = np.sqrt(w_raw.T @ cov_t @ w_raw) * np.sqrt(252)
            w = min(VOL_TARGET / (vol + 1e-8), MAX_LEVERAGE) * w_raw
        else:
            w = w_raw

        prior_for_turnover = prev_weights if prev_weights is not None else np.zeros(n)
        turnover = float(np.sum(np.abs(w - prior_for_turnover)))
        cost = TCOST * turnover

        # --- 3. Realize the return for d using YESTERDAY's decision (prev_weights) ---
        if prev_weights is not None:
            ret = float(prev_weights @ returns.loc[d].values - pending_cost)
            equity *= (1 + ret)
            append_csv(EQUITY_LOG_PATH, {
                "date": d.date().isoformat(),
                "return": ret,
                "equity": equity,
            })
            print(f"{d.date()}  realized return={ret:.4%}  equity={equity:.4f}")
        else:
            print(f"{d.date()}  bootstrap day, no prior weights yet — no return recorded")

        # --- 4. Log today's decision (applies to the NEXT trading day) ---
        row = {
            "date": d.date().isoformat(),
            "prob_bull": probs[bull], "prob_bear": probs[bear], "prob_sideways": probs[sideways],
        }
        for t, wt in zip(TICKERS, w):
            row[f"w_{t}"] = wt
        row["gross_leverage"] = float(np.sum(np.abs(w)))
        row["split_date"] = fit["split_date"]
        row["n_train_days"] = fit["n_train"]
        row["n_test_days"] = fit["n_test"]
        append_csv(WEIGHTS_LOG_PATH, row)

        # --- 5. Roll state forward ---
        prev_weights = w
        pending_cost = cost

    state = {
        "prev_weights": prev_weights,
        "pending_cost": pending_cost,
        "equity": equity,
        "last_decision_date": dates_to_process[-1].date().isoformat(),
    }
    save_state(state)
    print("State saved. Last decision date:", state["last_decision_date"])


if __name__ == "__main__":
    main()
