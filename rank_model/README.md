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
- `mlp_top100_hybrid_rank`

The four regression models fit the continuous percentile label. The three
native rankers and the hybrid MLP group observations strictly by trading date;
no training pair or ranking query crosses dates. Training rows from 2019-2022
and their exit dates remain in that period. Validation rows and exits remain in
2023. All training dates receive equal total loss weight.

`mlp_top100_hybrid_rank` combines equal-date rank MSE with a weighted pairwise
loss between each realized training Top100 stock and lower-ranked opponents.
It updates once per complete-date batch and restores the epoch with the highest
2023 mean daily NDCG@100, subject to the configured patience and minimum delta.
Rank IC is recorded at the selected epoch as a broad-ordering diagnostic. This
is model selection on validation data, not a 2024-2025 test result.

Run the hybrid candidate and compare it with the immutable baselines:

```powershell
& $python -m rank_model.pipeline --config rank_model\config.toml train `
  --model mlp_top100_hybrid_rank `
  --run-id real-2023-mlp-top100-hybrid

& $python -m rank_model.pipeline --config rank_model\config.toml evaluate `
  --run-id real-2023-mlp-top100-hybrid `
  --split validation

& $python -m rank_model.pipeline --config rank_model\config.toml compare `
  --run-ids real-2023-ridge,real-2023-xgb-reg,real-2023-lgb-reg,real-2023-xgb-rank,real-2023-lgb-rank,real-2023-mlp-reg,real-2023-mlp-rank,real-2023-mlp-top100-hybrid
```

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

## Freeze And Final Refit

After choosing specifications with the 2023 validation results, freeze the
five selected candidates before using the locked 2024-2025 test period:

```powershell
& $python -m rank_model.pipeline --config rank_model\config.toml freeze
```

`frozen_models.json` records each source validation run, its manifest and
feature-schema SHA-256, transformed feature order, fixed model parameters,
source-data hashes, a logical audit of the 2023 labels, and the exact physical
and full-table logical hashes of the sealed 2019-2023 rank dataset. It is immutable;
deleting it is an explicit research decision, not part of a normal rerun.

Refit each frozen specification on all eligible 2019-2023 labels:

```powershell
& $python -m rank_model.pipeline --config rank_model\config.toml refit --model ridge_rank_regression
& $python -m rank_model.pipeline --config rank_model\config.toml refit --model xgboost_rank_regression
& $python -m rank_model.pipeline --config rank_model\config.toml refit --model lightgbm_rank_regression
& $python -m rank_model.pipeline --config rank_model\config.toml refit --model lightgbm_lambdarank
& $python -m rank_model.pipeline --config rank_model\config.toml refit --model mlp_top100_hybrid_rank
```

Final refits include finite labels whose signal date is in 2019-2023 and whose
10-day target exit is no later than 2023-12-31. This reclaims observations that
were purged only at the old 2022/2023 train-validation boundary, while still
excluding labels that need 2024 prices. No validation split, early stopping,
test prediction, or strategy rule is used. The selected hybrid MLP runs a fixed
12 epochs. Final artifacts are immutable under
`rank_model/final_runs/<model-name>` and intentionally contain no prediction
file; 2024-2025 prediction is a later, separate locked-test step. `refit`
verifies the sealed rank file before opening it and does not read upstream
source data.

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
