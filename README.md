# Regime-Switching Paper Trading

Turns the `RegimeSwitchingFinal` backtest into a live, automated paper trading
book with zero infrastructure to manage — it runs entirely on GitHub's free
tier (Actions for compute/scheduling, Pages for the dashboard).

## How it works

- There is **no separate training step or frozen model anymore.** Every day,
  `daily_update.py` retrains from scratch on an *expanding* window of all
  history through that day (matching the backtest's daily retraining cadence).
- To avoid fitting the HMM and the regime-specific portfolios on the same
  data, each day's available history is **split in half**:
  - **First half** → fits the HMM (pure regime discovery, unsupervised).
  - **Second half** → regimes are decoded with that already-fit HMM
    (out-of-sample to the fit), and only this half is used to (a) label which
    state is bull/bear/sideways and (b) optimize the three regime-specific
    portfolios. "Today" is always the last day of the second half.
- Regime portfolios are **long-only, fully invested, per-asset capped**
  (`MAX_WEIGHT` in `config.py`, default 0.35) in both the max-Sharpe (bull)
  and min-variance (bear/sideways) optimizers — this tames the instability of
  unconstrained max-Sharpe. Long-only is intentional: the strategy rebalances
  a standing book rather than liquidating between regimes.
- On top of the regime-blended weights, the same vol-targeting overlay as the
  backtest scales exposure (up to `MAX_LEVERAGE`) to hit `VOL_TARGET`.
- `daily_update.py` realizes **yesterday's** decision as today's P&L (no
  lookahead), then makes tomorrow's decision, and appends to the logs in
  `docs/data/`. It's idempotent (safe to re-run) and self-healing (a missed
  day is replayed on the next run — though each replayed day means another
  full refit, so don't let gaps get too long).
- `docs/index.html` is a static dashboard that reads those CSV logs directly —
  no backend, just GitHub Pages serving the `docs/` folder. It also shows
  each day's train/test split diagnostic (how many days went into the HMM fit
  vs. the regime-labeling/optimization half) for transparency.
- One GitHub Actions workflow (`daily.yml`) does the scheduling: cron,
  `22:00 UTC` on weekdays (after US market close), runs `daily_update.py` and
  commits the updated logs back to the repo. Also triggerable manually from
  the Actions tab.

## One-time setup (~5 minutes)

1. **Create a new GitHub repo** (public, so Pages/raw CSVs work for free) and
   push everything in this folder to it.

2. **Enable Actions permissions**: repo → Settings → Actions → General →
   "Workflow permissions" → select **Read and write permissions**. (This lets
   the bot commit logs back to the repo.)

3. **Kick off the first daily update**: Actions tab → "Daily paper trading
   update" → Run workflow. This makes the first weight decision (bootstrap
   day, no return yet — that comes the following trading day). It'll take
   a bit longer than subsequent days since it's fitting on ~10+ years of history.

4. **Enable GitHub Pages**: Settings → Pages → Source: "Deploy from a branch"
   → Branch: `main`, folder: `/docs`. Save. Your dashboard will be live at
   `https://<your-username>.github.io/<repo-name>/` within a minute or two.

That's it — from here, `daily.yml` runs automatically every weekday.

## Configuration

Everything tunable lives in `config.py`: tickers, vol target, lookbacks,
leverage cap, per-asset weight cap (`MAX_WEIGHT`), transaction cost
assumption, and the minimum history required before a decision is made
(`MIN_HISTORY_DAYS`). Changes take effect on the next scheduled run — there's
nothing to retrain separately since retraining happens daily automatically.

## Honesty about what this is and isn't

- This is **paper trading** — no real orders are placed anywhere.
- Fills are simulated at that day's close with a flat proportional cost
  (`TCOST` in `config.py`); no slippage/market-impact model, no partial fills.
- The split-half procedure fixes the most obvious same-data leakage (HMM and
  regime-portfolio optimization no longer share data), but it doesn't
  eliminate every subtlety — e.g. the universe itself (5 large, long-since-
  proven blue chips) still reflects hindsight, and daily refitting means
  which physical HMM state means "bull" can occasionally re-sort day to day.
  The bull/bear/sideways *labels* stay semantically consistent (always
  highest/lowest/middle mean-return state) even when the underlying state
  index changes.
- `yfinance` occasionally has outages/rate limits; if a scheduled run fails,
  it'll just pick up the missed day(s) on the next successful run.

## Running locally (optional, for debugging)

```bash
pip install -r requirements.txt
python daily_update.py     # once per day you want to advance
```
