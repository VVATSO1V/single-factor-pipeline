# Rank Model 10-Day Cross-Sectional Ranking Design

## 1. Purpose

Build an isolated `rank_model` pipeline that predicts the relative 10-day
performance ordering of CSI 1000 constituents. The pipeline trains on a daily
cross-sectional percentile target instead of an absolute-return target.

The first release answers two questions:

1. Does replacing absolute-return regression with percentile-rank regression
   improve cross-sectional ordering?
2. Do native learning-to-rank objectives add value beyond rank regression when
   the features, dates, and evaluation are held constant?

The pipeline reports every metric and does not automatically label a run as
`accept`, `reject`, or `champion`.

## 2. Scope

### In scope

- One horizon only: 10 trading days.
- Entry at the adjusted open on `T+1` and exit at the adjusted open on `T+11`.
- Daily CSI 1000 membership determined by `in_universe` on `T`.
- A continuous daily target in `[0, 1]` derived from the realized 10-day return.
- Full point-in-time stock, market-cap, industry, market, and industry-context
  features already produced by the existing context pipeline.
- Four rank-regression baselines and three native-ranking extensions.
- 2019-2022 development training and 2023 validation.
- Complete cross-sectional, Top100, stability, data-quality, and runtime output.

### Out of scope for the first release

- 1-day and 5-day ranking models.
- Strategy execution, `T+1 entry_tradeable`, transaction costs, sell limits,
  cash accounting, and mandatory Top100 holdings.
- Automatic model selection or a composite score.
- Large hyperparameter grids.
- Market/industry/alpha target decomposition. Percentile ranks are ordinal and
  cannot be reconstructed by adding separately predicted return components.
- Reading or evaluating 2024-2025 during development.

## 3. Isolation and Reuse

`rank_model` is independent from `model` at the artifact boundary:

- `model/data/context_dataset_10d.parquet` and its schema are read-only inputs.
- Source code patterns may be copied and adapted from `model`, but imports from
  mutable internal `model.stages` modules are avoided.
- New datasets, preprocessors, models, predictions, and reports are written only
  under `rank_model`.
- Existing `model` files and run artifacts are never overwritten.
- Historical `model/runs` and large source datasets are not copied.

Proposed structure:

```text
rank_model/
  __init__.py
  config.toml
  pipeline.py
  README.md
  requirements.txt
  stages/
    __init__.py
    dataset.py
    preprocessing.py
    ranking.py
    training.py
    evaluation.py
  data/
    rank_dataset_10d.parquet
    rank_dataset_10d_schema.json
    rank_label_coverage.csv
  runs/
  docs/
    2026-08-05-rank-model-design.md
```

`pipeline.py` is the only public command entry point. Individual models do not
receive separate Python entry files.

## 4. Source Data and Feature Contract

The source is:

```text
model/data/context_dataset_10d.parquet
model/data/context_dataset_10d_schema.json
```

The source currently contains 2019-2023 development rows. The ranking pipeline
selects feature names from the schema rather than inferring numeric columns.

### Included features

- 17 factor values.
- 17 factor-missing indicators.
- `log_market_cap`, `market_cap_missing`, and `industry_missing`.
- The point-in-time `industry` category.
- Existing market-context features based on `T` and earlier observations.
- Existing industry-context features based on `T` and earlier observations.

Industry encoding is fitted on training data only. Missing industries are mapped
to `UNKNOWN`; categories not seen during training are also mapped to `UNKNOWN`.
All model families receive the same encoded industry information.

### Forbidden features

The feature whitelist must reject:

```text
stock_code
target_10d
rank_target_10d
market_target_10d
industry_target_10d
alpha_target_10d
exit_date_10d
split_10d
entry_tradeable
all T+1 or later status fields
```

`date` and `stock_code` are keys only. `industry` is a categorical feature but
is not interpreted as an ordered number.

## 5. Ranking Target

For stock `i` on date `t`, retain the existing absolute return:

\[
r^{10}_{i,t}=\frac{Open^{adj}_{i,T+11}}{Open^{adj}_{i,T+1}}-1
\]

The ranking universe is the `T`-day CSI 1000 constituent set. Among members with
a finite `target_10d`, sort returns ascending and assign average ranks for ties.
For `N_t >= 2` valid labels:

\[
q_{i,t}=\frac{rank_{avg}(r^{10}_{i,t})-1}{N_t-1}
\]

Higher future return therefore produces a higher `rank_target_10d`.

- The worst unique return maps to `0` and the best unique return maps to `1`.
- Tied returns receive the average of their occupied rank positions.
- A date with fewer than two finite returns cannot form a ranking target and is
  treated as a data-integrity failure.
- Rows with an unavailable future return remain in the key table with a missing
  rank target and do not enter supervised loss or target-based evaluation.
- Missing-label counts and coverage are reported for every date.
- `T+1 entry_tradeable` never changes the ranking universe or training sample.

The prepared table keeps both `target_10d` and `rank_target_10d`. The original
return is required for Top100 return, decile return, and spread evaluation.

## 6. Time Boundaries

Development keeps the existing split discipline:

```text
2019-2022: train and historical diagnostics
2023: validation and human analysis
2019-2023: final retraining only after the user chooses a specification
2024-2025: later frozen static test
```

For each split, the target exit date must remain inside the permitted boundary.
Training rows whose `T+11` exit falls in 2023 cannot enter 2019-2022 training.
The development commands must not read 2024-2025 feature or target rows.

The repository already discloses that 2024-2025 target aggregates were viewed
during an earlier audit. That period remains a static comparison period but is
not described as a pristine untouched test.

## 7. Preprocessing and Date Weighting

- Continuous preprocessing is fitted using training rows only.
- Ridge and MLP receive train-fitted scaling; tree models do not require numeric
  scaling.
- Industry one-hot vocabulary is fitted using training rows only and is shared
  across model families.
- The percentile target is not standardized.
- Feature order and transformed-column order are stored in every run manifest.
- Missing features follow the existing point-in-time cross-sectional transform
  and explicit missing-indicator contract.

Each date receives equal total training weight. If date `t` has `N_t` valid
labels, each row receives a weight proportional to `1 / N_t`. This prevents a
date with more valid labels from dominating another date.

## 8. Model Registry

The first release registers seven models.

### Rank-regression baselines

1. `ridge_rank_regression`
2. `xgboost_rank_regression`
3. `lightgbm_rank_regression`
4. `mlp_rank_regression`

These models fit `rank_target_10d` as a continuous response. The first run uses
the already established 10-day baseline capacity as a fixed starting point:

- Ridge: lambda `1.0`, existing solver and alpha convention.
- XGBoost: depth `4`, minimum child weight `100`, 81 rounds, learning rate
  `0.05`, row/feature subsampling `0.8`, L2 `10`, L1 `0.1`, seed `42`.
- LightGBM: 31 leaves, depth `5`, minimum 1000 rows per leaf, 21 rounds,
  learning rate `0.05`, row/feature subsampling `0.8`, L2 `10`, L1 `0.1`,
  deterministic seed `42`.
- MLP: layers `[128, 64, 32]`, ReLU, dropout `0.1`, AdamW learning rate
  `0.001`, weight decay `0.0001`, row batch size `8192`, 12 fixed epochs,
  gradient clipping `1.0`, seed `42`.

No multi-candidate grid or validation-driven early stopping is run in the first
comparison. The fixed round and epoch counts make the seven first-pass results
traceable and prevent one family from receiving substantially more search.

### Native-ranking extensions

5. `xgboost_pairwise_rank`
6. `lightgbm_lambdarank`
7. `mlp_pairwise_rank`

All native-ranking comparisons are confined to a single trading date.

- XGBoost uses a grouped pairwise ranking objective with date as query ID.
- LightGBM uses LambdaRank with date as group. Its internal relevance label is a
  deterministic 100-level discretization of the continuous percentile target;
  evaluation always uses the original continuous target. Linear label gains are
  used so the default exponential gain does not dominate the top level, and the
  LambdaRank truncation level is `100` to match the stated Top100 use case.
- MLP uses date batches and a logistic pairwise loss. Each valid stock appears as
  an anchor and samples eight opponents from the same date. Half are sampled
  across the broad cross-section and half from neighboring percentile bands.
  Batches contain eight dates, sampling is reproducible from seed `42`, and no
  pair crosses dates. The native MLP uses the same network and 12 epochs as its
  rank-regression counterpart.

Native model scores may have any finite numeric scale. They are converted to a
daily predicted percentile only after inference. Scores are not clipped before
ranking because clipping would create artificial ties.

## 9. Prediction Contract

Every run writes one prediction row per predicted key:

```csv
date,stock_code,split,horizon,target_10d,rank_target_10d,score_raw,pred_rank_pct,pred_rank_position
```

- `horizon` is always `10` in the first release.
- `score_raw` is the direct model output.
- `pred_rank_pct` is the daily average-rank percentile of `score_raw`, computed
  across every `T`-day universe member with a finite model score, not only rows
  whose future target is available.
- `pred_rank_position` is the daily ascending position, with the highest score
  receiving the largest position.
- Constant-score dates remain visible: all predicted percentiles become the
  tied average, rank correlation is missing for that date, and the date is
  counted in a prediction-degeneracy diagnostic.

Prediction output is Parquet to avoid creating another very large CSV. Reports
and metric tables use CSV or JSON where human inspection is useful.

## 10. Evaluation

No metric automatically determines a winner. Each run reports:

### Full cross-section

- Daily Spearman RankIC: mean, median, standard deviation, quantiles, and
  positive-date ratio.
- ICIR reported without mechanical annualization because 10-day targets overlap.
- Kendall Tau and pairwise ordering accuracy.
- Rank MAE and rank RMSE between `pred_rank_pct` and `rank_target_10d`.
- Valid and invalid metric-date counts.

### Top100

- NDCG@100 using the continuous true percentile as evaluation relevance.
- Precision@100, Recall@100, and Jaccard overlap with the realized Top100.
- Mean and median raw 10-day return of the predicted Top100.
- Predicted Top100 return minus same-day universe mean return.
- Predicted Top100 minus predicted Bottom100 return spread.

### Group structure

- Ten equal-count groups based on predicted score.
- Mean and median raw return for every group.
- Top-minus-bottom spread and group monotonicity diagnostics.

### Stability and uncertainty

- Daily, monthly, and yearly detail.
- Historical year-by-year diagnostics where available.
- Newey-West, lag `10`, confidence intervals for overlapping daily RankIC and
  Top100 spread series.
- No annualized performance claim from overlapping 10-day observations.

### Data and operational quality

- Rank-label coverage by date.
- Unknown-industry and missing-feature rates.
- Daily unique prediction count, tie ratio, and constant-score dates.
- Row, date, and feature counts.
- Training time, inference time, artifact size, source hashes, configuration
  hash, code version, and seed.

## 11. Standard Run Artifacts

Each run directory contains:

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

A comparison command creates a cross-run table with all metrics side by side.
It does not create `accept`, `reject`, or `champion` fields.

## 12. Commands

The public command shape is:

```powershell
.\.venv\Scripts\python.exe -m rank_model.pipeline prepare

.\.venv\Scripts\python.exe -m rank_model.pipeline train `
  --model ridge_rank_regression `
  --run-id ridge_rank_regression_10d

.\.venv\Scripts\python.exe -m rank_model.pipeline evaluate `
  --run-id ridge_rank_regression_10d `
  --split validation

.\.venv\Scripts\python.exe -m rank_model.pipeline compare `
  --run-ids ridge_rank_regression_10d,xgboost_rank_regression_10d
```

Every `run_id` is immutable. Existing runs are not silently overwritten.

## 13. Validation and Failure Handling

Preparation fails before publishing output when:

- source hashes do not match the source schema;
- `(date, stock_code)` is missing or duplicated;
- a source date does not have the expected CSI 1000 cross-section;
- fewer than two valid target returns exist on a date;
- the rank direction is inconsistent with raw-return ordering;
- a forbidden target, split, key, exit, or future-status field enters features;
- development data extends beyond 2023;
- split exit dates cross their allowed boundary.

Training fails when date groups are not contiguous, group sizes disagree with
rows, transformed features are non-finite, or predictions do not cover the
expected validation keys.

Writes use temporary files followed by atomic replacement. Necessary tests use
temporary directories and their generated artifacts are removed after checks;
no standalone test-data files are delivered with the pipeline.

## 14. Interpretation Boundary

The model predicts an ordinal cross-sectional score, not a calibrated expected
return. A score difference of `0.2` does not imply a 20% return difference.
Top100 raw returns remain evaluation outcomes, not model outputs.

The ranking model may improve stock selection while absolute-return R-squared
remains unavailable or irrelevant. Strategy conclusions require the later
execution layer with `T+1 entry_tradeable`, transaction costs, sell constraints,
cash, and non-replacement of failed buys; those mechanics are not part of this
design.
