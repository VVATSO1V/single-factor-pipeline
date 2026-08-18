# Rank Model Static Strategy Execution Design

## Purpose

Run one shared, immutable 2024-2025 strategy simulation for all five frozen
10-day rank models. The strategy stage measures executable portfolio outcomes;
it must not retrain a model, change a parameter, select a winner, or write any
development, final-model, prediction, or locked-test evaluation artifact.

The frozen model set is:

- `ridge_rank_regression`
- `xgboost_rank_regression`
- `lightgbm_rank_regression`
- `lightgbm_lambdarank`
- `mlp_top100_hybrid_rank`

The strategy reports every model under identical rules. Results may be compared
descriptively, but 2024-2025 results must not be used for model selection or
retuning. Dates from 2026 onward remain reserved for forward simulation.

## Inputs And Boundary

Each strategy run reads only:

- one sealed `locked_test_runs/<model>/predictions_10d.parquet`;
- `model/data/market_panel.csv`;
- `model/data/trading_calendar.csv`;
- `frozen_models.json`;
- the sealed locked-test conclusion and artifact hashes.

Predictions must cover exactly 1,000 unique stocks on every 2024-2025 signal
date. The strategy selects the 100 highest finite `score_raw` values, with
`stock_code` ascending as the deterministic tie breaker. It never reads
`target_10d`, `rank_target_10d`, or any locked-test metric when generating an
order or sizing a position.

A signal observed after the close on official trading date T can only execute
at the open of the next official trading date T+1. Signals whose execution date
falls outside 2024-2025 are not executed. In particular, the 2025-12-31 signal
is reserved for the 2026 forward stage.

## Price And Valuation Contract

Tradeability uses unadjusted `raw_open`, `limit_up`, and `limit_down`. Portfolio
valuation and units use adjusted `post_open`, so open-to-open returns remain
continuous across corporate actions. The initial net asset value is 1.0 and
fractional units are allowed; board lots and minimum commissions belong to the
later live-order adapter.

At each execution open, existing positions are marked to that date's valid
`post_open` before orders execute. A suspended or otherwise missing-price
holding keeps its most recent valid adjusted-open mark, has zero marked return
while no price is available, and cannot trade. When a valid price returns, the
full move from the last valid mark is recognized.

The backtest ends without forced liquidation. Remaining positions are valued at
the last available 2025-12-31 marks and published as the starting snapshot for
forward simulation.

## Portfolio Transition

Rebalancing is daily and uses strict set-difference turnover:

1. Holdings that remain in the new Top100 are untouched; their units do not
   change.
2. Holdings outside the new Top100 are sell candidates.
3. New Top100 names not already held are buy candidates.
4. A blocked sale remains in the portfolio and is retried on later dates while
   the stock remains outside the desired set.
5. A failed buy is not replaced by rank 101 or another stock.

Each eligible new name requests gross notional equal to pre-trade NAV divided
by 100. Buy orders are capped at that slot size. If cash is insufficient after
sales and costs, all eligible buys are scaled by one common factor so stock-code
order cannot determine allocation. Capital assigned to an ineligible buy stays
in cash. Blocked exits may make the realized holding count exceed 100.

## Execution Rules

A new buy at T+1 requires all of the following:

- the stock belongs to the T signal Top100;
- it is not ST;
- it is not suspended;
- `listing_days >= 120`;
- positive, finite `raw_open`, `limit_up`, `limit_down`, and `post_open`;
- `limit_up > limit_down`;
- the open is neither at limit-up nor at limit-down under the shared tolerance.

A sell at T+1 is blocked only when the stock is suspended, lacks a valid open
or adjusted-open valuation, or opens at limit-down. ST status, listing age, and
an open at limit-up do not block a sale.

## Cost Contract

Costs are applied to executed gross notional and deducted from cash:

- commission: 1 basis point on buys and sells;
- slippage: 5 basis points on buys and sells;
- stamp duty: 5 basis points on sells only.

Thus the modeled buy cost is 6 basis points and the sell cost is 11 basis
points. There is no minimum commission in this normalized research simulation.

## Public Interface And Outputs

The unified public entry point adds two commands:

```powershell
& $python -m rank_model.pipeline --config rank_model\config.toml `
  backtest-strategy --model lightgbm_lambdarank
& $python -m rank_model.pipeline --config rank_model\config.toml `
  compare-strategy
```

One internal strategy stage serves all models. Each immutable model directory
under `rank_model/strategy_runs` contains:

- `daily_nav.csv`: NAV, return, cash, holdings, costs, and turnover;
- `trades.parquet`: attempted and executed orders with rejection reasons;
- `positions.parquet`: daily end-of-open position snapshots;
- `execution_diagnostics.csv`: daily execution-state counts;
- `ending_positions.csv`: the final 2025 position and cash snapshot;
- `metrics_summary.json`: return, risk, drawdown, cash, cost, and turnover metrics;
- `manifest.json`: fixed settings and SHA-256 input/output contracts.

`strategy_comparison.csv` contains one row per completed model and no winner,
acceptance, or parameter-update field.

## Metrics

The daily report includes gross and net return, NAV, cash ratio, realized
holding count, desired Top100 count, buy and sell notional, costs, gross
turnover `(buys + sells) / pre_trade_nav`, one-way turnover equal to half that
value, and counts for attempted, executed, and blocked orders.

The summary includes cumulative return, annualized return and volatility,
zero-risk-free Sharpe ratio, maximum drawdown, daily win rate, average cash
ratio, average and annualized turnover, total costs, buy and sell counts, fill
rates, blocked-sale days, and ending NAV. All formulas and annualization factors
are recorded in the manifest.

## Integrity And Verification

The strategy command validates exact date-to-calendar mapping, prediction and
market-panel key uniqueness, finite scores, five-model membership, all input
hashes, and the frozen date/cost contract before simulation. Publication uses a
temporary directory, a writer lock, and atomic rename; an existing run is never
silently overwritten.

Temporary tests cover deterministic Top100 ties, T-to-T+1 alignment, buy and
sell restrictions, set-difference invariants, blocked-sale carry, no buy
replacement, proportional cash scaling, suspension marking, costs, turnover,
period boundaries, and target-column independence. Temporary tests and caches
are removed after production acceptance.
