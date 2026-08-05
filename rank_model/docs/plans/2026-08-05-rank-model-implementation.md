# Rank Model 10-Day Ranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated `rank_model` pipeline that trains and evaluates seven 10-day CSI 1000 cross-sectional ranking models using a daily `[0, 1]` percentile target.

**Architecture:** `rank_model` reads the existing 2019-2023 context dataset and schema as immutable inputs, creates its own rank dataset, and exposes `prepare`, `doctor`, `train`, `evaluate`, and `compare` through one module entry point. Shared dataset, preprocessing, ranking, training, and evaluation modules are copied and narrowed from proven `model` patterns; all generated artifacts remain under `rank_model`.

**Tech Stack:** Python 3.11, pandas, NumPy, SciPy, PyArrow, scikit-learn, joblib, XGBoost, LightGBM, PyTorch, TOML, stdlib `unittest`.

## Global Constraints

- Implement only the 10-trading-day horizon.
- Keep `model/data/context_dataset_10d.parquet` and `model/data/context_dataset_10d_schema.json` read-only.
- Do not modify files outside `rank_model`.
- Use adjusted open `T+1` to adjusted open `T+11` absolute return already stored as `target_10d`.
- Build `rank_target_10d` within the `T`-day CSI 1000 cross-section using average ranks and `(rank - 1) / (N - 1)`.
- Do not filter training or ranking labels using `entry_tradeable` or any `T+1` status.
- Use the complete schema-approved stock, market-cap, industry, market-context, and industry-context feature set.
- Never use `stock_code`, targets, split fields, exit dates, or future-status fields as features.
- Train on 2019-2022, evaluate on 2023, and do not read 2024-2025 during development.
- Give every training date equal total sample weight.
- Preserve every model's metrics; do not emit `accept`, `reject`, or `champion` decisions.
- Use immutable run IDs and atomic artifact publication.
- Temporary tests may exist during implementation but must be deleted before delivery.
- Use `apply_patch` for manual file creation and edits.

---

## File Map

| File | Responsibility |
|---|---|
| `rank_model/__init__.py` | Package marker and version |
| `rank_model/config.toml` | Paths, time bounds, feature restrictions, and fixed model parameters |
| `rank_model/pipeline.py` | Public CLI and command dispatch only |
| `rank_model/stages/dataset.py` | Source verification, percentile labels, coverage, schema, and atomic writes |
| `rank_model/stages/preprocessing.py` | Train-only scaler, industry encoding, date weights, and predicted percentiles |
| `rank_model/stages/ranking.py` | Date groups, relevance levels, deterministic Pair sampling, and pairwise loss |
| `rank_model/stages/training.py` | Seven-model registry, fitting, prediction, reload checks, and run publication |
| `rank_model/stages/evaluation.py` | Daily/monthly/yearly, Top100, decile, HAC, and cross-run metrics |
| `rank_model/README.md` | Data contract, PowerShell commands, outputs, and interpretation limits |
| `rank_model/requirements.txt` | Nine local-model dependencies: pandas, NumPy, SciPy, PyArrow, scikit-learn, joblib, XGBoost, LightGBM, and PyTorch |
| `rank_model/tests/` | Temporary implementation tests; removed in the final task |

---

### Task 1: Package, Configuration, and Percentile Dataset

**Files:**
- Create: `rank_model/__init__.py`
- Create: `rank_model/config.toml`
- Create: `rank_model/pipeline.py`
- Create: `rank_model/stages/__init__.py`
- Create: `rank_model/stages/dataset.py`
- Create: `rank_model/requirements.txt`
- Create temporarily: `rank_model/tests/test_dataset.py`

**Interfaces:**
- Produces: `percentile_rank_from_returns(values: pd.Series) -> pd.Series`
- Produces: `build_rank_dataset(source_dataset: Path, source_schema: Path, output_dataset: Path, output_schema: Path, coverage_path: Path, development_end: date) -> dict[str, Any]`
- Produces: `load_rank_dataset(dataset_path: Path, schema_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]`
- Produces CLI: `python -m rank_model.pipeline prepare`

- [ ] **Step 1: Write failing label and source-contract tests**

Create `rank_model/tests/test_dataset.py` with concrete cases for strict ordering, ties, missing targets, duplicate keys, forbidden development dates, and source-hash mismatch:

```python
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from rank_model.stages.dataset import percentile_rank_from_returns


class PercentileRankTests(unittest.TestCase):
    def test_strict_order_maps_to_zero_half_one(self) -> None:
        values = pd.Series([-0.1, 0.0, 0.2], dtype="float64")
        actual = percentile_rank_from_returns(values)
        np.testing.assert_allclose(actual.to_numpy(), [0.0, 0.5, 1.0])

    def test_ties_receive_average_positions(self) -> None:
        values = pd.Series([0.0, 0.0, 0.2], dtype="float64")
        actual = percentile_rank_from_returns(values)
        np.testing.assert_allclose(actual.to_numpy(), [0.25, 0.25, 1.0])

    def test_missing_target_remains_missing(self) -> None:
        values = pd.Series([0.0, np.nan, 0.2], dtype="float64")
        actual = percentile_rank_from_returns(values)
        np.testing.assert_allclose(actual.iloc[[0, 2]].to_numpy(), [0.0, 1.0])
        self.assertTrue(np.isnan(actual.iloc[1]))

    def test_fewer_than_two_labels_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two finite returns"):
            percentile_rank_from_returns(pd.Series([np.nan, 0.1]))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify the missing module/function failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest rank_model.tests.test_dataset -v
```

Expected: import failure for `rank_model.stages.dataset` or missing `percentile_rank_from_returns`.

- [ ] **Step 3: Implement the minimal target formula and validated dataset writer**

Copy only the proven key normalization, SHA-256, Parquet metadata, temporary-file, and atomic replacement patterns from `model/stages/dataset.py`. Implement the target core exactly as:

```python
def percentile_rank_from_returns(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").astype("float64")
    finite = np.isfinite(numeric.to_numpy())
    count = int(finite.sum())
    if count < 2:
        raise ValueError("a rank date requires at least two finite returns")
    output = pd.Series(np.nan, index=values.index, dtype="float64")
    average_rank = numeric.loc[finite].rank(method="average", ascending=True)
    output.loc[finite] = (average_rank - 1.0) / float(count - 1)
    return output
```

`build_rank_dataset` must:

1. Verify the source Parquet hash against `context_dataset_10d_schema.json`.
2. Read only schema-declared keys, features, `target_10d`, `split_10d`, and `exit_date_10d`.
3. Reject duplicate or missing `(date, stock_code)` keys.
4. Reject dates after `2023-12-31`.
5. Require exactly 1000 source rows per date.
6. Compute `rank_target_10d` separately for each date from finite `target_10d`.
7. Verify that return ordering and rank ordering never disagree.
8. Write `rank_label_coverage.csv` with `date,total_rows,valid_targets,coverage,tie_rows`.
9. Write Parquet and JSON through temporary files, then atomically replace final paths.
10. Record source hashes, row/date counts, feature lists, forbidden columns, target formula, and output hash in the schema.

Create a small `argparse` CLI with `prepare`, `doctor`, `train`, `evaluate`, and `compare` subcommands, but only wire `prepare` in this task. `config.toml` must resolve paths relative to itself and define `development_end = "2023-12-31"`.

- [ ] **Step 4: Extend tests to cover bundle publication**

Add a temporary Parquet fixture with two dates, four rows per date, source metadata, and a matching schema. Assert:

```python
self.assertEqual(result["row_count"], 8)
self.assertEqual(result["date_count"], 2)
self.assertEqual(result["rank_target_column"], "rank_target_10d")
self.assertTrue(output_dataset.exists())
self.assertTrue(output_schema.exists())
self.assertTrue(coverage_path.exists())
```

Corrupt the source hash in a second test and assert `ValueError` contains `source hash`.

- [ ] **Step 5: Run dataset tests and compile the package**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest rank_model.tests.test_dataset -v
.\.venv\Scripts\python.exe -m compileall -q rank_model
```

Expected: all dataset tests pass and compileall returns exit code 0.

- [ ] **Step 6: Commit the isolated dataset foundation**

```powershell
git add rank_model/__init__.py rank_model/config.toml rank_model/pipeline.py rank_model/requirements.txt rank_model/stages/__init__.py rank_model/stages/dataset.py rank_model/tests/test_dataset.py
git commit -m "feat: build 10d rank dataset"
```

---

### Task 2: Feature Contract and Train-Only Preprocessing

**Files:**
- Create: `rank_model/stages/preprocessing.py`
- Create temporarily: `rank_model/tests/test_preprocessing.py`
- Modify: `rank_model/stages/dataset.py`
- Modify: `rank_model/config.toml`

**Interfaces:**
- Consumes: schema fields `continuous_feature_columns` and `industry_column` from Task 1.
- Produces: `RankPreprocessor.fit(frame: pd.DataFrame, continuous_columns: list[str], industry_column: str) -> RankPreprocessor`
- Produces: `RankPreprocessor.transform(frame: pd.DataFrame, scale_continuous: bool) -> np.ndarray`
- Produces: `equal_date_weights(dates: pd.Series) -> np.ndarray`
- Produces: `predicted_percentiles(scores: np.ndarray, dates: pd.Series) -> np.ndarray`

- [ ] **Step 1: Write failing preprocessing tests**

Copy the behavioral coverage of `ContextPreprocessor` into temporary tests. Use training industries `Bank`, `Tech`, and a missing value; validation includes unseen `Energy`. Assert:

```python
preprocessor = RankPreprocessor.fit(
    train,
    continuous_columns=["factor_a", "log_market_cap"],
    industry_column="industry",
)
self.assertEqual(preprocessor.industry_categories, ["Bank", "Tech", "UNKNOWN"])
matrix = preprocessor.transform(validation, scale_continuous=True)
self.assertEqual(matrix.shape[1], 5)
self.assertTrue(np.isfinite(matrix).all())
self.assertEqual(matrix[0, -1], 1.0)
```

For dates containing two and four rows, assert each date's weights sum to `1.0`. For scores `[3, 1, 2]` on one date, assert predicted percentiles `[1.0, 0.0, 0.5]`.

- [ ] **Step 2: Run tests and verify missing interfaces**

```powershell
.\.venv\Scripts\python.exe -m unittest rank_model.tests.test_preprocessing -v
```

Expected: missing `RankPreprocessor` or helper functions.

- [ ] **Step 3: Implement preprocessing by narrowing the proven context code**

Copy `_normalized_industries`, `ContextPreprocessor`, and `fit_context_preprocessor` behavior from `model/stages/context_training.py`; rename the public class to `RankPreprocessor`. Preserve these rules:

```python
def equal_date_weights(dates: pd.Series) -> np.ndarray:
    normalized = pd.to_datetime(dates, errors="raise").dt.normalize()
    counts = normalized.groupby(normalized).transform("size").to_numpy(dtype="float64")
    if np.any(counts <= 0):
        raise ValueError("date weights require non-empty groups")
    return 1.0 / counts


def predicted_percentiles(scores: np.ndarray, dates: pd.Series) -> np.ndarray:
    frame = pd.DataFrame({"date": pd.to_datetime(dates), "score": scores})
    if not np.isfinite(frame["score"]).all():
        raise ValueError("prediction scores must be finite")
    return frame.groupby("date", sort=False)["score"].transform(
        lambda x: (x.rank(method="average") - 1.0) / max(len(x) - 1, 1)
    ).to_numpy(dtype="float64")
```

Validate the schema feature whitelist before fitting. Reject duplicate feature names and every forbidden name from the design. Fit the scaler and industry vocabulary using training rows only. Map missing and unseen industries to `UNKNOWN`.

- [ ] **Step 4: Run preprocessing and dataset tests**

```powershell
.\.venv\Scripts\python.exe -m unittest rank_model.tests.test_dataset rank_model.tests.test_preprocessing -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit preprocessing**

```powershell
git add rank_model/config.toml rank_model/stages/dataset.py rank_model/stages/preprocessing.py rank_model/tests/test_preprocessing.py
git commit -m "feat: add rank feature preprocessing"
```

---

### Task 3: Immutable Run Contract and Ridge Rank Regression

**Files:**
- Create: `rank_model/stages/training.py`
- Create temporarily: `rank_model/tests/test_ridge_training.py`
- Modify: `rank_model/pipeline.py`
- Modify: `rank_model/config.toml`

**Interfaces:**
- Produces dataclass: `TrainingOutcome(score_validation: np.ndarray, model_objects: dict[str, Any], metadata: dict[str, Any])`
- Produces: `train_registered_model(config: dict[str, Any], config_path: Path, model_name: str, run_id: str) -> Path`
- Produces registry entry: `ridge_rank_regression`
- Produces CLI: `python -m rank_model.pipeline train --model ridge_rank_regression --run-id <id>`

- [ ] **Step 1: Write failing immutable-run and Ridge tests**

Create a synthetic three-date training set and one-date validation set with two continuous features, one industry column, and a monotonic rank target. Assert:

```python
outcome = train_ridge_rank_regression(train, validation, schema, params)
self.assertEqual(outcome.score_validation.shape, (len(validation),))
self.assertTrue(np.isfinite(outcome.score_validation).all())
self.assertEqual(outcome.metadata["lambda"], 1.0)
```

Create an existing temporary run directory and assert `train_registered_model` raises `FileExistsError` instead of overwriting it.

- [ ] **Step 2: Run the Ridge tests and verify failure**

```powershell
.\.venv\Scripts\python.exe -m unittest rank_model.tests.test_ridge_training -v
```

Expected: missing training module or Ridge trainer.

- [ ] **Step 3: Implement the common training contract and Ridge**

Implement a registry of callables with one model name per specification. For Ridge:

```python
weights = equal_date_weights(train["date"])
preprocessor = RankPreprocessor.fit(
    train,
    continuous_columns=schema["continuous_feature_columns"],
    industry_column=schema["industry_column"],
)
x_train = preprocessor.transform(train, scale_continuous=True)
x_validation = preprocessor.transform(validation, scale_continuous=True)
y_train = train["rank_target_10d"].to_numpy(dtype="float64")
alpha = float(params["lambda"]) * float(weights.sum())
model = Ridge(alpha=alpha, fit_intercept=True, solver="cholesky")
model.fit(x_train, y_train, sample_weight=weights)
score_validation = model.predict(x_validation)
```

Before training, filter only rows with finite rank labels; keep every validation prediction key and attach true targets only where available. Publish the run through a temporary directory and atomic rename. Write `config_snapshot.toml`, `feature_schema.json`, preprocessor, model, prediction Parquet, and manifest. Reload the preprocessor and model and require reloaded predictions to match before publication.

- [ ] **Step 4: Wire train CLI and run tests**

The CLI choices must contain all seven final names, although only Ridge dispatch succeeds in this task. Unknown or not-yet-registered names must fail with an explicit message.

```powershell
.\.venv\Scripts\python.exe -m unittest rank_model.tests.test_ridge_training -v
.\.venv\Scripts\python.exe -m compileall -q rank_model
```

Expected: all tests pass and the synthetic run contains no `accept`, `reject`, or `champion` field.

- [ ] **Step 5: Commit Ridge training**

```powershell
git add rank_model/config.toml rank_model/pipeline.py rank_model/stages/training.py rank_model/tests/test_ridge_training.py
git commit -m "feat: train ridge rank regression"
```

---

### Task 4: XGBoost and LightGBM Rank Regression

**Files:**
- Modify: `rank_model/stages/training.py`
- Modify: `rank_model/config.toml`
- Create temporarily: `rank_model/tests/test_tree_rank_regression.py`

**Interfaces:**
- Adds registry entry: `xgboost_rank_regression`
- Adds registry entry: `lightgbm_rank_regression`
- Both return the Task 3 `TrainingOutcome` contract.

- [ ] **Step 1: Write failing tree-regression tests**

Build 20 training dates and two validation dates with 30 rows per date. Assert each trainer returns finite scores for every validation row and metadata exactly records fixed rounds and objective:

```python
self.assertEqual(xgb_result.metadata["objective"], "reg:squarederror")
self.assertEqual(xgb_result.metadata["boosting_rounds"], 81)
self.assertEqual(lgb_result.metadata["objective"], "regression")
self.assertEqual(lgb_result.metadata["boosting_rounds"], 21)
```

- [ ] **Step 2: Run the tests and verify missing registry entries**

```powershell
.\.venv\Scripts\python.exe -m unittest rank_model.tests.test_tree_rank_regression -v
```

Expected: missing tree trainer functions or registry keys.

- [ ] **Step 3: Implement fixed tree rank regression**

Use unscaled continuous values plus the shared train-fitted industry encoding. Pass `equal_date_weights` as sample weights. Use the exact fixed parameters from the design:

```python
XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "eta": 0.05,
    "max_depth": 4,
    "min_child_weight": 100.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 10.0,
    "alpha": 0.1,
    "seed": 42,
}

LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 5,
    "min_data_in_leaf": 1000,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "feature_fraction": 0.8,
    "lambda_l2": 10.0,
    "lambda_l1": 0.1,
    "deterministic": True,
    "force_col_wise": True,
    "seed": 42,
}
```

Do not use validation-driven early stopping. Save native XGBoost JSON and LightGBM text artifacts, reload both, and verify prediction equality.

- [ ] **Step 4: Run all regression-model tests**

```powershell
.\.venv\Scripts\python.exe -m unittest rank_model.tests.test_ridge_training rank_model.tests.test_tree_rank_regression -v
```

Expected: Ridge, XGBoost, and LightGBM tests pass.

- [ ] **Step 5: Commit tree regression**

```powershell
git add rank_model/config.toml rank_model/stages/training.py rank_model/tests/test_tree_rank_regression.py
git commit -m "feat: add tree rank regressors"
```

---

### Task 5: MLP Rank Regression

**Files:**
- Modify: `rank_model/stages/training.py`
- Create temporarily: `rank_model/tests/test_mlp_rank_regression.py`

**Interfaces:**
- Adds registry entry: `mlp_rank_regression`
- Produces saved PyTorch state dictionary plus architecture metadata.

- [ ] **Step 1: Write failing deterministic MLP test**

Use five dates with 20 rows each and set NumPy/PyTorch seed `42`. Train twice and assert finite, equal predictions within `1e-6`. Assert architecture and epoch metadata:

```python
self.assertEqual(result.metadata["hidden_layers"], [128, 64, 32])
self.assertEqual(result.metadata["epochs"], 12)
self.assertEqual(result.metadata["loss"], "mse")
```

- [ ] **Step 2: Run the test and verify missing MLP entry**

```powershell
.\.venv\Scripts\python.exe -m unittest rank_model.tests.test_mlp_rank_regression -v
```

Expected: missing `mlp_rank_regression` trainer.

- [ ] **Step 3: Implement the fixed MLP**

Copy the small feed-forward network construction and deterministic seed handling from `model/stages/training.py`. Use scaled features, MSE, AdamW, batch size 8192, exactly 12 epochs, no validation early stopping, gradient clipping `1.0`, and date-equal row weights inside the loss:

```python
weighted_loss = (row_weights * torch.square(prediction - target)).sum()
weighted_loss = weighted_loss / row_weights.sum()
```

Save architecture JSON and `state_dict`, reload a new model instance, and require validation predictions to match within `1e-6`.

- [ ] **Step 4: Run deterministic MLP and prior tests**

```powershell
.\.venv\Scripts\python.exe -m unittest rank_model.tests.test_mlp_rank_regression rank_model.tests.test_ridge_training rank_model.tests.test_tree_rank_regression -v
```

Expected: all four rank-regression families pass.

- [ ] **Step 5: Commit MLP rank regression**

```powershell
git add rank_model/stages/training.py rank_model/tests/test_mlp_rank_regression.py
git commit -m "feat: add mlp rank regression"
```

---

### Task 6: Native XGBoost Pairwise and LightGBM LambdaRank

**Files:**
- Create: `rank_model/stages/ranking.py`
- Modify: `rank_model/stages/training.py`
- Modify: `rank_model/config.toml`
- Create temporarily: `rank_model/tests/test_native_tree_ranking.py`

**Interfaces:**
- Produces: `sorted_group_layout(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]`
- Produces: `lightgbm_relevance(rank_target: pd.Series) -> np.ndarray`
- Adds registry entry: `xgboost_pairwise_rank`
- Adds registry entry: `lightgbm_lambdarank`

- [ ] **Step 1: Write failing group and relevance tests**

Shuffle a frame containing three dates and assert `sorted_group_layout` returns contiguous dates and group sizes `[3, 2, 4]`. Assert no key is lost. Test relevance conversion exactly:

```python
actual = lightgbm_relevance(pd.Series([0.0, 0.009, 0.01, 0.999, 1.0]))
np.testing.assert_array_equal(actual, [0, 0, 1, 99, 99])
```

Train both native tree models on a toy grouped dataset and assert one finite score per validation key.

- [ ] **Step 2: Run tests and verify missing ranking helpers**

```powershell
.\.venv\Scripts\python.exe -m unittest rank_model.tests.test_native_tree_ranking -v
```

Expected: missing ranking module or native registry entries.

- [ ] **Step 3: Implement exact group contracts and native objectives**

`sorted_group_layout` must stable-sort by `date,stock_code`, return integer group sizes, and verify `sum(group_sizes) == len(frame)`. XGBoost receives an integer query ID derived only from date and uses:

```python
{
    "objective": "rank:pairwise",
    "eval_metric": "ndcg@100",
    "tree_method": "hist",
    "eta": 0.05,
    "max_depth": 4,
    "min_child_weight": 100.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 10.0,
    "alpha": 0.1,
    "seed": 42,
}
```

LightGBM receives integer relevance labels, date group sizes, 21 fixed rounds, linear `label_gain = list(range(100))`, `lambdarank_truncation_level = 100`, and the same tree-capacity parameters as its regression counterpart. Neither trainer may form a query across dates.

- [ ] **Step 4: Run native and regression tree tests**

```powershell
.\.venv\Scripts\python.exe -m unittest rank_model.tests.test_native_tree_ranking rank_model.tests.test_tree_rank_regression -v
```

Expected: all tests pass; saved and reloaded native models reproduce scores.

- [ ] **Step 5: Commit native tree ranking**

```powershell
git add rank_model/config.toml rank_model/stages/ranking.py rank_model/stages/training.py rank_model/tests/test_native_tree_ranking.py
git commit -m "feat: add native tree ranking"
```

---

### Task 7: Native MLP Pairwise Ranking

**Files:**
- Modify: `rank_model/stages/ranking.py`
- Modify: `rank_model/stages/training.py`
- Create temporarily: `rank_model/tests/test_mlp_pairwise_rank.py`

**Interfaces:**
- Produces: `sample_date_pairs(dates: pd.Series, targets: np.ndarray, pairs_per_stock: int, adjacent_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]`
- Produces: `pairwise_logistic_loss(scores: torch.Tensor, left: torch.Tensor, right: torch.Tensor, direction: torch.Tensor) -> torch.Tensor`
- Adds registry entry: `mlp_pairwise_rank`

- [ ] **Step 1: Write failing Pair sampler tests**

Use two dates with ten rows each. Assert every pair stays inside one date, every stock appears as a left anchor exactly eight times, the returned direction is only `-1` or `1`, and repeated seed `42` returns identical arrays. Assert changing the seed changes at least one opponent.

Test the loss:

```python
good = pairwise_logistic_loss(
    torch.tensor([2.0, 0.0]),
    torch.tensor([0]),
    torch.tensor([1]),
    torch.tensor([1.0]),
)
bad = pairwise_logistic_loss(
    torch.tensor([0.0, 2.0]),
    torch.tensor([0]),
    torch.tensor([1]),
    torch.tensor([1.0]),
)
self.assertLess(float(good), float(bad))
```

- [ ] **Step 2: Run tests and verify missing Pair interfaces**

```powershell
.\.venv\Scripts\python.exe -m unittest rank_model.tests.test_mlp_pairwise_rank -v
```

Expected: missing sampler or pairwise loss.

- [ ] **Step 3: Implement deterministic date batches and pairwise MLP**

For each date and each valid stock anchor, sample eight opponents: four uniformly from the date and four from neighboring percentile bands. Exclude self-pairs and exact target ties. Raise a clear error when a date cannot provide a valid opponent. Use:

```python
signed_margin = direction * (scores[left] - scores[right])
loss = torch.nn.functional.softplus(-signed_margin).mean()
```

Use eight dates per batch, the same `[128, 64, 32]` network, dropout, optimizer, 12 epochs, clipping, and seed as `mlp_rank_regression`. Save sampler parameters and pair counts in the manifest. Reload and reproduce validation scores within `1e-6`.

- [ ] **Step 4: Run all native-ranking tests**

```powershell
.\.venv\Scripts\python.exe -m unittest rank_model.tests.test_mlp_pairwise_rank rank_model.tests.test_native_tree_ranking -v
```

Expected: Pair constraints, deterministic training, and model reload checks pass.

- [ ] **Step 5: Commit native MLP ranking**

```powershell
git add rank_model/stages/ranking.py rank_model/stages/training.py rank_model/tests/test_mlp_pairwise_rank.py
git commit -m "feat: add pairwise mlp ranking"
```

---

### Task 8: Evaluation, Reports, and Cross-Run Comparison

**Files:**
- Create: `rank_model/stages/evaluation.py`
- Modify: `rank_model/stages/training.py`
- Modify: `rank_model/pipeline.py`
- Create temporarily: `rank_model/tests/test_evaluation.py`

**Interfaces:**
- Produces: `evaluate_predictions(predictions: pd.DataFrame, hac_lag: int = 10, top_k: int = 100) -> EvaluationBundle`
- Produces: `write_evaluation(bundle: EvaluationBundle, run_dir: Path) -> None`
- Produces: `compare_runs(run_dirs: list[Path], output_path: Path) -> pd.DataFrame`
- Produces CLI: `evaluate --run-id <id> --split validation`
- Produces CLI: `compare --run-ids <comma-separated ids>`

- [ ] **Step 1: Write failing perfect, reverse, tie, and missing-target tests**

Create two dates with 1000 rows each. For perfect scores, assert RankIC, Kendall Tau, pairwise accuracy, Precision@100, Recall@100, and NDCG@100 equal `1.0`; rank MAE equals `0.0`. For reversed scores, assert RankIC equals `-1.0` and Top100 overlap is zero. For constant scores, assert the date is counted as invalid RankIC and `constant_score_date = 1`. Include missing targets and assert prediction keys are retained while target metrics use finite targets only.

- [ ] **Step 2: Run evaluation tests and verify missing module**

```powershell
.\.venv\Scripts\python.exe -m unittest rank_model.tests.test_evaluation -v
```

Expected: missing evaluation interfaces.

- [ ] **Step 3: Implement metric primitives**

Adapt `_ndcg_at_k` and `hac_mean_test` from `model/stages/context_training.py`, replacing return-prediction fields with rank fields. Compute per date:

```text
rank_ic, kendall_tau, pairwise_accuracy, rank_mae, rank_rmse,
ndcg_100, precision_100, recall_100, jaccard_100,
top100_mean_return, top100_median_return, universe_mean_return,
top100_excess_return, bottom100_mean_return, top_bottom_spread,
top100_valid_return_count, unique_score_count, tie_ratio, valid_target_count
```

Compute pairwise accuracy from Kendall Tau without materializing all pair
combinations. Compute ten equal-count predicted-score groups and preserve all
ten returns. Aggregate daily metrics to monthly and yearly tables without
weighting dates by row count. Use Newey-West lag 10 for mean daily RankIC,
Top100 excess return, and Top-Bottom spread. Report ICIR without annualization.

- [ ] **Step 4: Implement standard output files and comparison**

Write:

```text
predictions_10d.parquet
metrics_summary.json
daily_metrics.csv
monthly_metrics.csv
yearly_metrics.csv
decile_returns.csv
top100_detail.parquet
manifest.json
```

`compare_runs` reads only completed manifests and summaries, writes one row per run and one column per metric, and omits all automatic-decision fields. A missing or duplicate run ID must fail explicitly.

- [ ] **Step 5: Run evaluation and full temporary test suite**

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s rank_model/tests -v
.\.venv\Scripts\python.exe -m compileall -q rank_model
```

Expected: every test passes; no report contains `accept`, `reject`, or `champion`.

- [ ] **Step 6: Commit evaluation**

```powershell
git add rank_model/pipeline.py rank_model/stages/evaluation.py rank_model/stages/training.py rank_model/tests/test_evaluation.py
git commit -m "feat: evaluate rank model runs"
```

---

### Task 9: Doctor, Documentation, Real-Data Acceptance, and Test Cleanup

**Files:**
- Create: `rank_model/README.md`
- Modify: `rank_model/pipeline.py`
- Modify: `rank_model/config.toml`
- Delete before delivery: `rank_model/tests/`

**Interfaces:**
- Produces CLI: `python -m rank_model.pipeline doctor`
- Documents every production command and artifact.
- Leaves no temporary tests or test data in the delivered tree.

- [ ] **Step 1: Implement doctor checks before documentation**

`doctor` must verify and print:

1. Source dataset and schema exist.
2. Source and published hashes match.
3. Development end is no later than `2023-12-31`.
4. Every source date has exactly 1000 keys.
5. Keys are unique and non-null.
6. All schema-declared features exist and are finite after source preparation.
7. Forbidden features are absent from the model feature list.
8. Training and validation dates and exit dates respect boundaries.
9. All nine runtime dependencies import successfully.
10. Output paths resolve under `rank_model` and source paths resolve under `model/data`.

- [ ] **Step 2: Write the README with exact PowerShell commands**

Document this sequence:

```powershell
.\.venv\Scripts\python.exe -m rank_model.pipeline doctor
.\.venv\Scripts\python.exe -m rank_model.pipeline prepare
.\.venv\Scripts\python.exe -m rank_model.pipeline train --model ridge_rank_regression --run-id ridge_rank_regression_10d
.\.venv\Scripts\python.exe -m rank_model.pipeline evaluate --run-id ridge_rank_regression_10d --split validation
```

List all seven model names, the source/feature/target contracts, immutable run behavior, output schemas, no-automatic-winner policy, and the distinction between ordinal scores and expected returns.

- [ ] **Step 3: Run the complete temporary test suite one final time**

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s rank_model/tests -v
.\.venv\Scripts\python.exe -m compileall -q rank_model
```

Expected: all tests pass and compileall exits 0.

- [ ] **Step 4: Run real-data doctor and prepare**

```powershell
.\.venv\Scripts\python.exe -m rank_model.pipeline doctor
.\.venv\Scripts\python.exe -m rank_model.pipeline prepare
```

Verify the prepared dataset has the same key count and date count as the source, no date after 2023, exactly 1000 rows per date, monotonic rank direction, finite rank labels wherever raw targets are finite, and schema hashes matching the output file.

- [ ] **Step 5: Run one end-to-end synthetic command for every registered model**

Use a temporary configuration whose paths and run directory are under the system temporary directory. Invoke `train` and `evaluate` for all seven registry names against a small grouped synthetic Parquet dataset. Assert every run writes all standard artifacts and reload checks pass. Remove the temporary directory after the command exits.

- [ ] **Step 6: Remove temporary tests and Python caches**

Delete only paths resolved beneath `rank_model`:

```powershell
Remove-Item -LiteralPath .\rank_model\tests -Recurse -Force
Get-ChildItem -LiteralPath .\rank_model -Recurse -Directory -Filter __pycache__ | ForEach-Object { Remove-Item -LiteralPath $_.FullName -Recurse -Force }
```

Then verify:

```powershell
Test-Path .\rank_model\tests
Get-ChildItem .\rank_model -Recurse -Directory -Filter __pycache__
```

Expected: `False` and no cache-directory output.

- [ ] **Step 7: Verify scope and working tree**

```powershell
git diff --check
git status --short
git diff --name-only HEAD~9..HEAD
```

Expected: code changes are confined to `rank_model`; unrelated user changes remain untouched.

- [ ] **Step 8: Commit documentation and cleanup**

```powershell
git add rank_model
git commit -m "docs: finish rank model pipeline"
```

Do not stage unrelated repository files or generated model run artifacts.

---

## Final Acceptance Checklist

- [ ] `rank_model` reads but never writes `model/data`.
- [ ] Rank labels match the documented formula and retain raw returns.
- [ ] No future-status or target column enters features.
- [ ] Date groups never cross trading dates.
- [ ] All dates receive equal total training weight.
- [ ] Seven registered models train and reload successfully on synthetic data.
- [ ] Development commands do not read 2024-2025.
- [ ] Every run emits predictions and all requested metric tables.
- [ ] Cross-run comparison contains no automatic winner fields.
- [ ] Real-data `doctor` and `prepare` pass.
- [ ] Temporary tests, temporary data, and `__pycache__` are absent at delivery.
- [ ] Existing `model` source and artifacts are unchanged.
