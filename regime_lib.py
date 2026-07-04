# -*- coding: utf-8 -*-
"""
Shared logic used by daily_update.py.

Key design change from the original notebook: to avoid fitting the HMM and
the regime-specific portfolios on the same data (which lets the portfolio
optimizer "see" the very days used to define its own regime labels), the
available history as of each decision day is split in half:

    - First half  -> fit the HMM (regime discovery only)
    - Second half -> decode regimes with that (already-fit) HMM, and use
                     ONLY this out-of-sample-to-the-HMM-fit half to assign
                     bull/bear/sideways labels and optimize the three
                     regime-specific portfolios.

This whole procedure is redone daily on an expanding window (see
daily_update.py), matching the retraining frequency used in the backtest.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from config import MAX_WEIGHT, NORM_LOOKBACK, TICKERS, TRAIN_START


def download_prices(end=None):
    """Download adjusted close prices for the configured universe."""
    data = yf.download(TICKERS, start=TRAIN_START, end=end, auto_adjust=False, progress=False)
    prices = data['Adj Close'].dropna()
    return prices


def build_features(prices):
    """Reproduces the exact feature set from the original notebook."""
    returns = prices.pct_change().dropna()

    hmm_features = pd.DataFrame(index=returns.index)
    hmm_features['return_mean'] = returns.mean(axis=1)
    hmm_features['rolling_vol_10d'] = hmm_features['return_mean'].rolling(10).std()
    hmm_features['momentum_10d'] = hmm_features['return_mean'].rolling(10).sum()
    hmm_features = hmm_features.dropna()

    rolling_mean = hmm_features.rolling(window=NORM_LOOKBACK, min_periods=NORM_LOOKBACK).mean()
    rolling_std = hmm_features.rolling(window=NORM_LOOKBACK, min_periods=NORM_LOOKBACK).std() + 1e-8

    valid_dates = hmm_features.index[
        rolling_mean.notna().all(axis=1) & rolling_std.notna().all(axis=1)
    ]

    hmm_features = hmm_features.loc[valid_dates]
    rolling_mean = rolling_mean.loc[valid_dates]
    rolling_std = rolling_std.loc[valid_dates]
    returns = returns.loc[valid_dates]

    return returns, hmm_features, rolling_mean, rolling_std, valid_dates


def max_sharpe(mu, cov, max_weight=MAX_WEIGHT):
    n = len(mu)
    w0 = np.ones(n) / n
    cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, max_weight)] * n
    return minimize(
        lambda w: -(w @ mu) / (np.sqrt(w.T @ cov @ w) + 1e-8),
        w0, bounds=bounds, constraints=cons, method='SLSQP'
    ).x


def min_var(cov, max_weight=MAX_WEIGHT):
    n = cov.shape[0]
    w0 = np.ones(n) / n
    cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, max_weight)] * n
    return minimize(
        lambda w: w.T @ cov @ w,
        w0, bounds=bounds, constraints=cons, method='SLSQP'
    ).x


def fit_regime_model(returns, hmm_features, rolling_mean, rolling_std, valid_dates,
                      n_iter=4000, random_state=42):
    """
    Fit the HMM on the first half of `valid_dates` and use the second half
    (out-of-sample to that fit) to label regimes and optimize the three
    regime-specific portfolios. Returns everything needed to make today's
    decision, where "today" = valid_dates[-1] (the last day of the second half).

    Returns a dict:
        hmm, bull, bear, sideways, weights_by_regime (dict of np arrays),
        probs_today (np array of length 3, indexed like the raw HMM states),
        split_date (str), n_train, n_test
    """
    n_total = len(valid_dates)
    split_idx = n_total // 2
    train_dates = valid_dates[:split_idx]
    test_dates = valid_dates[split_idx:]

    features_std = (hmm_features - rolling_mean) / rolling_std

    hmm = GaussianHMM(n_components=3, covariance_type='full', n_iter=n_iter, random_state=random_state)
    hmm.fit(features_std.loc[train_dates].values)

    test_std = features_std.loc[test_dates].values
    test_states = hmm.predict(test_std)               # Viterbi decode, out-of-sample to the fit
    test_probs = hmm.predict_proba(test_std)           # smoothed state probabilities

    regime_df = hmm_features.loc[test_dates].copy()
    regime_df['regime'] = test_states

    means = regime_df.groupby('regime')['return_mean'].mean()
    present_states = list(means.index)
    bull = int(means.idxmax())
    bear = int(means.idxmin())
    remaining = list(set(present_states) - {bull, bear})
    sideways = int(remaining[0]) if remaining else bear  # degenerate fallback if a state never occurs

    test_returns = returns.loc[test_dates]
    weights_by_regime = {}
    for state, name in zip([bull, bear, sideways], ['bull', 'bear', 'sideways']):
        R = test_returns[regime_df['regime'].values == state]
        if len(R) < 10:
            # Too few observations of this regime in the test half to fit
            # anything sensible — fall back to equal weight rather than
            # let LedoitWolf/optimizer run on a near-empty sample.
            weights_by_regime[name] = np.ones(len(TICKERS)) / len(TICKERS)
            continue
        mu = R.mean().values
        cov = LedoitWolf().fit(R.values).covariance_
        weights_by_regime[name] = (
            max_sharpe(mu, cov) if name == 'bull' else min_var(cov)
        )

    probs_today = test_probs[-1]  # today == test_dates[-1] == valid_dates[-1]

    return {
        "hmm": hmm,
        "bull": bull, "bear": bear, "sideways": sideways,
        "weights_by_regime": weights_by_regime,
        "probs_today": probs_today,
        "split_date": str(train_dates[-1].date()),
        "n_train": len(train_dates), "n_test": len(test_dates),
    }
