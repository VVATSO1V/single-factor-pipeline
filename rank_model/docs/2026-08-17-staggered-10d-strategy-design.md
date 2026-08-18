# Staggered 10-Trading-Day Strategy Design

## Purpose

Evaluate each of the five frozen 10-day rank models with ten independent,
full-capital rebalance paths. Each path rebalances every ten official trading
days, but the paths use offsets 1 through 10 so every eligible signal date is
represented exactly once across the timing study. This measures sensitivity to
the rebalance starting date without selecting the best offset.

The implementation extends the existing static strategy stage. It must reuse
the current valuation, tradeability, sizing, cost, set-difference, reporting,
immutability, and provenance logic. Existing daily strategy code and artifacts
remain unchanged.

The frozen model set remains:

- `ridge_rank_regression`
- `xgboost_rank_regression`
- `lightgbm_rank_regression`
- `lightgbm_lambdarank`
- `mlp_top100_hybrid_rank`

## Inputs And Information Boundary

Each run reads the same sealed inputs as the daily strategy:

- `locked_test_runs/<model>/predictions_10d.parquet`;
- `model/data/market_panel.csv`;
- `model/data/trading_calendar.csv`;
- `frozen_models.json`;
- the locked-test conclusion and recorded artifact hashes.

Only `date`, `stock_code`, `score_raw`, `split`, and `horizon` may enter signal
selection. Targets and locked-test performance metrics are forbidden. A signal
observed after the close on T can execute only at the next official open T+1.

The model target is:

```text
target_10d(T) = post_open(T+11) / post_open(T+1) - 1
```

The sealed prediction files contain 485 daily cross-sections from 2024-01-02
through 2025-12-31. Only the 474 signal dates from 2024-01-02 through
2025-12-16 have a complete T+1 through T+11 interval inside 2024-2025. The
remaining eleven late-2025 signals are excluded from this static test and
reserved for 2026 forward simulation.

## Offset Schedule

Number the 474 eligible signal dates from 1 through 474. Offset `k`, where
`k` is 1 through 10, receives signal numbers:

```text
k, k + 10, k + 20, ...
```

For a scheduled signal on T:

- select the deterministic Top100 after T close;
- execute the rebalance at T+1 open;
- mark the full 10-trading-day horizon at T+11 open;
- the next scheduled rebalance for that offset also executes at this T+11
  open, using the signal observed at T+10 close.

The schedule validator must prove that the ten offsets partition all 474
eligible signal dates with no gaps and no duplicates. Offset 1 first executes
on 2024-01-03; offset 10 first executes on 2024-01-16. Each path starts with
NAV 1.0 on 2024-01-02 and remains fully in cash until its own first execution.

## Daily State Machine

Every path is valued on every official trading day from 2024-01-02 through its
own final complete T+11 date. Existing positions are marked with valid
`post_open`; missing or suspended positions carry the last valid adjusted-open
mark and recognize the full move when a valid mark returns.

On a scheduled rebalance date:

1. Replace the path's desired set with the new signal Top100.
2. Cancel a pending exit if that stock has returned to the new desired set.
3. Preserve units for holdings in the desired-set intersection.
4. Attempt to sell holdings outside the desired set.
5. Attempt to buy desired names not already held.

On a non-rebalance date:

- mark all holdings and compute daily NAV;
- retry only previously blocked sell orders whose stocks remain outside the
  current desired set;
- do not retry failed buys and do not read a new prediction Top100.

A failed buy is not replaced by rank 101 and remains cash for that cycle. If
the name appears in the next scheduled Top100, it becomes a fresh buy candidate
then. A blocked sale is retried daily until it executes or a later scheduled
Top100 makes the holding desired again. Blocked exits can make holdings exceed
100 and can reduce cash available for new names.

All existing execution rules remain binding:

- buys require non-ST, non-suspended, listing age at least 120 trading days,
  valid open/adjusted-open/limit data, and an open at neither price limit;
- sells are blocked only by suspension, missing valid execution or valuation
  data, or an open at limit-down;
- retained holdings are never sold and repurchased merely to resize them;
- eligible new names request `pre_trade_nav / 100` gross notional;
- insufficient cash scales all eligible buys by one common factor;
- commission is 1 bp on both sides, slippage is 5 bps on both sides, and stamp
  duty is 5 bps on sells.

## Path Boundaries

Each path includes the common 2024-01-02 cash row and ends at the open that
completes its last full T+1 to T+11 holding interval. The final positions are
marked but not forcibly liquidated, matching the existing strategy contract.
Consequently, the ten path end dates can differ by at most nine trading days.
Every path manifest and metric table records its offset, first signal, first
execution, final signal, final horizon date, and elapsed observation count.

## Path Metrics And Timing Distribution

Each offset publishes the existing daily strategy reports and metric formulas.
For every model, an offset detail table contains one row per offset. A model
summary reports the arithmetic mean, median, sample standard deviation,
minimum, and maximum of every comparable path metric, including:

- cumulative and annualized return;
- annualized volatility and Sharpe ratio;
- maximum drawdown and win rate;
- cash ratio, turnover, and total costs;
- attempted/executed order counts, fill rates, and blocked-sale days.

No offset is accepted, rejected, or selected. Mean performance describes the
timing-averaged result; dispersion and the worst offset describe timing risk.

## Average NAV And Aggregate Metrics

The model also publishes a research average NAV line. It uses only the common
calendar interval for which all ten paths have genuine reported NAV values:

- start: 2024-01-02, with not-yet-entered paths held at NAV 1.0 in cash;
- end: the earliest of the ten final complete-horizon dates, expected to be
  2025-12-18 under the sealed calendar;
- no forward fill beyond a path's actual final horizon date.

For normalized full-capital path NAVs:

```text
average_nav(t) = sum(path_nav_k(t), k=1..10) / 10
average_cash(t) = sum(path_cash_k(t), k=1..10) / 10
average_buy_notional(t) = sum(path_buy_notional_k(t), k=1..10) / 10
average_sell_notional(t) = sum(path_sell_notional_k(t), k=1..10) / 10
average_cost(t) = sum(path_cost_k(t), k=1..10) / 10
```

Aggregate daily returns are recomputed from `average_nav`, not averaged from
the ten reported Sharpes or returns. Aggregate turnover is recomputed from
average buy and sell notional divided by aggregate pre-trade NAV. Cash ratio is
average cash divided by average NAV. The average-line report independently
computes cumulative return, CAGR, annualized volatility, Sharpe ratio, maximum
drawdown, win rate, cash ratio, turnover, and costs.

The average line is the primary timing-averaged model result. Offset statistics
remain necessary to detect a high mean caused by a small number of favorable
starting dates.

## Public Interface And Artifacts

The public CLI adds:

```powershell
& $python -m rank_model.pipeline --config rank_model\config.toml `
  backtest-strategy-10d --model ridge_rank_regression
& $python -m rank_model.pipeline --config rank_model\config.toml `
  compare-strategy-10d
```

Outputs are separate from and must never overwrite daily strategy artifacts:

```text
rank_model/strategy_10d_runs/
  <model>/
    offset_01/
      daily_nav.csv
      trades.parquet
      positions.parquet
      execution_diagnostics.csv
      ending_positions.csv
      metrics_summary.json
      manifest.json
    ...
    offset_10/
    offset_metrics.csv
    offset_summary.csv
    average_nav.csv
    average_metrics.json
    manifest.json

rank_model/strategy_10d_comparison.csv
rank_model/strategy_10d_comparison.manifest.json
rank_model/daily_vs_10d_comparison.csv
rank_model/daily_vs_10d_comparison.manifest.json
```

The model-level manifest binds all ten offset manifests, the aggregate files,
the exact calendar partition, and the same sealed source hashes. Comparison
manifests bind the exact five model manifests and their output hashes. Existing
valid outputs are verified without rewriting; incomplete, stale, or tampered
artifact sets fail closed.

The 10-day comparison reports all five models and contains no automatic winner
field. The daily-versus-10-day table compares return, Sharpe, volatility,
drawdown, turnover, costs, and cash ratio under the two rebalance policies.

## Verification Contract

Temporary tests and independent artifact checks must cover:

- exact 474-date partition across ten offsets;
- T-close to T+1-open alignment and T+11 horizon completion;
- cash-only waiting periods before first entry;
- no target-column dependence;
- no non-scheduled Top100 refresh;
- daily blocked-sale retry and no failed-buy retry;
- pending-sale cancellation when a stock re-enters the desired set;
- set-difference intersection-unit preservation;
- buy/sell restrictions, costs, cash scaling, marking, and no replacement;
- distinct final horizon dates and no forced liquidation;
- path metrics, offset distribution statistics, average NAV, and aggregate
  metric reconciliation;
- immutable publication, path containment, provenance, tamper detection, and
  idempotent validation;
- unchanged daily strategy artifacts and formulas.

Temporary test files and caches are removed after verification. The coding
agent must not run the five final real-model backtests; it delivers the public
PowerShell commands for the user to run manually.

## Decision Discipline

The 2024-2025 results may support a one-time deployment decision between the
already frozen models and rebalance policies. They must not be used to retune,
refit, add features, or revise strategy parameters while still being described
as an untouched test. Any selected configuration is evaluated prospectively
from 2026 onward.
