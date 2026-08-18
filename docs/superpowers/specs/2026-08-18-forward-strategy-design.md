# Forward Strategy Runner Design

## Goal

Add an independent, up-to-date forward strategy entrypoint that refreshes the
latest available CSI1000 data, scores stocks with the five already-frozen rank
models, and runs the existing staggered 10-trading-day strategy without
training, refitting, tuning, or overwriting the sealed 2024-2025 artifacts.

## Scope

The entrypoint covers only the forward data, inference, and strategy layers:

1. Resolve the latest completed official trading date.
2. Optionally refresh the 17 factor tables and market panel through RiceQuant.
3. Build the same point-in-time feature schema used by the frozen models.
4. Load the five frozen model artifacts and produce 10-day cross-sectional scores.
5. Restrict signals to dates whose T+1 execution and T+11 exit are available.
6. Run the existing ten-offset staggered strategy state machine.
7. Publish immutable forward outputs and a provenance manifest.

It does not modify the training split, validation split, frozen model files,
2024-2025 locked-test files, or the existing 2024-2025 strategy outputs.

## User-facing commands

Run from the repository root:

```powershell
$python = ".\\.venv\\Scripts\\python.exe"

# Refresh factor and market inputs, then score and backtest.
& $python -m rank_model.forward_strategy `
  --config rank_model\\forward_strategy.toml `
  --run-id forward-latest `
  --update-data

# Reuse already refreshed local inputs without any data access.
& $python -m rank_model.forward_strategy `
  --config rank_model\\forward_strategy.toml `
  --run-id forward-latest-local `
  --local-only
```

The command must refuse an existing run ID and must never overwrite a previous
forward result.

## Data flow

### Update mode

`--update-data` uses the existing factor builders and market-panel builder. It
passes the same historical start date needed by rolling and point-in-time
features and an automatically resolved latest completed trading date as the
end date. Each factor remains in its own research directory. The forward entry
only orchestrates those existing builders; it does not duplicate factor
formulas.

The updater must write through temporary files and publish the refreshed local
inputs only after successful completion. A failure must leave the previous
inputs untouched and must identify the failed factor or market source.

### Local-only mode

`--local-only` reads the configured factor CSVs and market panel as they exist
locally. It validates that every required factor and market source reaches the
same resolved as-of date. It never initializes RiceQuant.

### Feature and score generation

The forward runner builds only the rows needed after the last frozen
development date. It preserves the established 17-factor feature order,
missing-value indicators, T-date market-cap and industry inputs, market and
industry environment features, and train-fitted categorical/preprocessing
artifacts. It loads, but never writes, each frozen model and produces one
prediction table per model with:

```text
date,stock_code,score_raw,pred_rank_pct,pred_rank_position
```

The score is an ordinal ranking score. It is not treated as a calibrated return
forecast. The stock-code ascending order is the deterministic tie breaker.

## Point-in-time and execution rules

- A signal uses only data visible through the close of T.
- T+1 is the execution date and uses the adjusted open for valuation.
- The 10-trading-day exit is the T+11 open.
- The latest signal date is computed from the official calendar, not calendar
  days, as the latest date whose T+1 and T+11 observations are available.
- A signal with an incomplete holding window is retained only in a diagnostic
  coverage report and is excluded from performance metrics.
- The existing strategy state machine remains authoritative for Top100
  selection, set-difference rebalancing, blocked sells, failed buys, costs,
  cash, and valuation.

## Outputs

Every run is written below the configured forward root:

```text
rank_model/forward_strategy_runs/<run-id>/
  asof.json
  factor_update_manifest.json
  feature_snapshot.parquet
  predictions/
    ridge_rank_regression.parquet
    xgboost_rank_regression.parquet
    lightgbm_rank_regression.parquet
    lightgbm_lambdarank.parquet
    mlp_top100_hybrid_rank.parquet
  strategy_10d_runs/<model-name>/...
  strategy_10d_comparison.csv
  manifest.json
```

The manifest records the resolved as-of date, last complete signal date,
source paths and hashes, frozen model hashes, configuration hash, model list,
row counts, and output hashes. Outputs are immutable and are independent from
`rank_model/locked_test_runs`, `rank_model/strategy_runs`, and
`rank_model/strategy_10d_runs`.

## Failure behavior

The runner stops before publishing a run when any of the following occurs:

- RiceQuant credentials are missing in update mode.
- A factor builder does not produce the required schema or date coverage.
- Market data does not cover the official trading calendar or T+1/T+11 fields.
- A frozen model, feature schema, or fitted preprocessor is missing or has a
  hash mismatch.
- The five prediction tables do not have the same date-stock key set.
- The latest date does not leave at least one complete 10-day signal.
- An existing run ID or partially published run is found.

No fallback silently uses future data, a different model, or the old locked
test output.

## Compatibility boundary

The original `rank_model.pipeline` commands remain unchanged. The forward
runner is additive and is the only command allowed to create the new forward
output tree. Model training and final refitting remain outside this flow.
