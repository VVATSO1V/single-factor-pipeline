# Rank Model 2024-2025 Locked Test Design

## Purpose

Complete the third and fourth stages after model specification freeze and final
2019-2023 refit:

1. Build the isolated 2024-2025 rank-model input and generate static predictions
   from every frozen final model.
2. Evaluate and compare those predictions with one shared metric implementation.

This stage measures out-of-sample decay. It must not retrain a model, change a
parameter, perform early stopping, or automatically select a winner.

## Data Boundary

The local `model/data/model_dataset.parquet` contains the full 2019-2025
cross-section. The locked-test builder reads only local artifacts and does not
call Ricequant. It reconstructs the same feature families used in development:

- daily cross-sectional factor transforms;
- stock market-cap and industry-missing features;
- point-in-time market and industry context from `market_panel.csv`;
- the existing 10-day adjusted-open return and exit date.

Context rolling windows are calculated on history through each signal date so
the first 2024 observations retain their 2023 lookback. Published rows are
restricted to signal dates from 2024-01-01 through 2025-12-31. A label is valid
only when its recorded `exit_date_10d` is no later than 2025-12-31. Late-2025
rows without a complete 10-day outcome keep their keys and receive a missing
label.

`entry_tradeable`, ST, suspension, limit status, and all T+1 fields are excluded
from model features and do not filter the model-evaluation sample. Those fields
belong to the later strategy-execution stage.

## Feature Contract

The locked-test schema must match the sealed development rank schema exactly:

- same continuous source columns and order;
- same categorical industry column;
- same daily percentile-rank formula and direction;
- exactly 1000 unique `(date, stock_code)` keys per signal date;
- finite prepared continuous values;
- no forbidden future or execution feature.

Every source file, development contract, and published locked-test artifact is
recorded by SHA-256. Preparation uses temporary files and atomic publication.
An existing locked-test dataset is not silently overwritten.

## Static Inference

Each model under `rank_model/final_runs/<model_name>` is loaded together with
its fitted 2019-2023 preprocessor. Artifact hashes, model name, transformed
feature order, frozen-spec hash, and final-training purpose are verified before
inference. The preprocessor is never fitted again.

The five frozen candidates are:

- `ridge_rank_regression`
- `xgboost_rank_regression`
- `lightgbm_rank_regression`
- `lightgbm_lambdarank`
- `mlp_top100_hybrid_rank`

Each immutable output directory under `rank_model/locked_test_runs` contains a
prediction Parquet and an inference manifest. Predictions retain all 1000 keys
per date and use the existing contract:

```text
date,stock_code,split,horizon,target_10d,rank_target_10d,
score_raw,pred_rank_pct,pred_rank_position
```

The split is always `test` and horizon is always `10`.

## Evaluation And Comparison

The existing ranking metrics are reused without changing formulas. Each model
receives overall 2024-2025 output plus daily, monthly, and yearly diagnostics,
decile returns, and Top100 detail. This naturally exposes 2024 and 2025 as
separate yearly rows while keeping one combined summary.

A single comparison CSV contains all five completed locked-test summaries. It
does not contain `accept`, `reject`, `champion`, or another automatic decision
field. Reading test results is one-way: no command in this stage writes model
parameters or final-model artifacts.

## Public Commands

```powershell
& $python -m rank_model.pipeline --config rank_model\config.toml prepare-test
& $python -m rank_model.pipeline --config rank_model\config.toml predict-test --model ridge_rank_regression
& $python -m rank_model.pipeline --config rank_model\config.toml evaluate-test --model ridge_rank_regression
& $python -m rank_model.pipeline --config rank_model\config.toml compare-test
```

`predict-test`, `evaluate-test`, and `compare-test` fail if a required prior
artifact is missing or its hash no longer matches. Existing output directories
are immutable and are never silently replaced.

## Verification

Temporary tests cover date and exit boundaries, rank labels, exact feature
contract, absence of execution features, model reload for every artifact type,
prediction key preservation, immutable writes, and comparison output. Real-data
acceptance verifies 2024-2025 coverage, 1000 keys per date, finite scores,
artifact hashes, and the absence of any model-training write. Temporary tests
and caches are removed before delivery.
