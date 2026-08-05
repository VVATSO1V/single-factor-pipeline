# CSI1000 10-Day Rank Model

This package trains fixed, cross-sectional ranking specifications on the
prepared CSI1000 10-day development dataset. It only reads the source bundle
under `model/data`; it publishes prepared data and immutable model runs under
`rank_model`.

## Production Workflow

Run these commands from the repository root after the package is integrated:

```powershell
.\.venv\Scripts\python.exe -m rank_model.pipeline doctor
.\.venv\Scripts\python.exe -m rank_model.pipeline prepare
.\.venv\Scripts\python.exe -m rank_model.pipeline train --model ridge_rank_regression --run-id ridge_rank_regression_10d
.\.venv\Scripts\python.exe -m rank_model.pipeline evaluate --run-id ridge_rank_regression_10d --split validation
```

`doctor` is read-only. It verifies source presence and hashes, date and exit
boundaries, one thousand unique non-null keys per source date, finite declared
continuous features, forbidden-feature exclusions, runtime imports, published
hashes when present, and path containment. `prepare` writes the rank dataset.
`train` creates an immutable run: reusing a run ID fails rather than replacing
existing artifacts. `evaluate` only accepts the `validation` split. To compare
completed runs without selecting a winner, use:

```powershell
.\.venv\Scripts\python.exe -m rank_model.pipeline compare --run-ids ridge_rank_regression_10d,xgboost_rank_regression_10d
```

## Registered Models

- `ridge_rank_regression`
- `xgboost_rank_regression`
- `lightgbm_rank_regression`
- `mlp_rank_regression`
- `xgboost_pairwise_rank`
- `lightgbm_lambdarank`
- `mlp_pairwise_rank`

The four regression models fit the continuous percentile label. The three
native rankers group observations strictly by trading date; no training pair or
ranking query crosses dates. Training rows from 2019-2022 and their exit dates
remain in that period. Validation rows and exits remain in 2023. All training
dates receive equal total loss weight.

## Data Contracts

The source schema declares the keys `date` and `stock_code`, continuous feature
columns, categorical `industry`, raw target `target_10d`, split `split_10d`,
and exit date `exit_date_10d`. Source dates have exactly 1000 unique,
non-null keys. The preparation step preserves raw returns and adds
`rank_target_10d`:

```text
(average_rank(target_10d) - 1) / (finite_count - 1)
```

Ranks are ascending, so a larger finite raw return has a larger rank label.
Rows without a raw target retain their keys and have a missing rank label. The
model feature list rejects keys, targets, split fields, exit dates,
`entry_tradeable`, and future-status fields. Continuous source features must be
finite after source preparation; industry is fitted as a train-only categorical
vocabulary and unknown values map to `UNKNOWN`.

`score_raw` is an ordinal model score, not a calibrated expected return. Its
scale and score differences must not be read as return forecasts. Evaluation
converts scores to daily percentiles for ordering metrics while retaining raw
`target_10d` for return, Top100, decile, and spread reporting.

## Artifacts

`prepare` creates the following files under `rank_model/data`:

```text
rank_dataset_10d.parquet
rank_dataset_10d_schema.json
rank_label_coverage.csv
```

The schema records source hashes, output hash, features, target formula, and
row/date counts. Every run under `rank_model/runs/<run-id>` contains:

```text
config_snapshot.toml
feature_schema.json
preprocessor.joblib
model artifact(s)
predictions_10d.parquet
metrics_summary.json
daily_metrics.csv
monthly_metrics.csv
yearly_metrics.csv
decile_returns.csv
top100_detail.parquet
manifest.json
```

Predictions use this schema:

```text
date,stock_code,split,horizon,target_10d,rank_target_10d,score_raw,pred_rank_pct,pred_rank_position
```

The comparison output has one row per completed run and metric columns only.
It intentionally contains no automatic `accept`, `reject`, or `champion`
field: specification choice remains a human decision.
