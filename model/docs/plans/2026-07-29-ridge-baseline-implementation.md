# Ridge Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible Ridge baseline that predicts 1-day, 5-day, and 10-day absolute open-to-open stock returns using only 2019-2022 training data and 2023 validation data.

**Architecture:** `model.pipeline` remains the only public command entry. A focused preprocessing module converts each complete T-day CSI1000 cross-section into 17 clipped z-scores plus 17 missing flags, and a training module joins horizon-specific sample membership, tunes Ridge, and atomically publishes one run bundle. The training command filters Parquet inputs at 2023-12-31 before materialization and never uses execution fields.

**Tech Stack:** Python 3.11+, pandas, NumPy, PyArrow, scikit-learn, joblib, TOML, Parquet

## Global Constraints

- All new model work and generated run artifacts stay under `model/`.
- Signal date is T close and the target is absolute T+1-open entry to future open return.
- Train on 2019-2022 and select hyperparameters on 2023.
- Do not read, predict, or evaluate 2024-2025 in this command.
- Do not use `stock_code`, industry, market cap, target columns, or `entry_*` columns as features.
- Do not filter training samples with `entry_tradeable`.
- Transform all 1000 stocks on each date before horizon-specific sample filtering.
- One run may train horizons 1, 5, 10, or all; do not create one Python file per model or horizon.
- Temporary tests live in `model/.ridge_checks/` and must be deleted before delivery.
- Existing run directories must never be overwritten.

---

### Task 1: Dependency And Configuration Contract

**Files:**
- Create: `model/requirements.txt`
- Modify: `model/config.toml`
- Modify: `model/pipeline.py`
- Test temporarily: `model/.ridge_checks/test_config.py`

**Interfaces:**
- Consumes: `load_config(path: Path) -> dict[str, Any]`
- Produces: validated `[preprocessing]` and `[models.ridge]` sections and the CLI shape `train --model ridge --horizon {1,5,10,all} --run-id ID`

- [ ] **Step 1: Write failing configuration and parser checks**

```python
from pathlib import Path
from model.pipeline import load_config, make_parser

config = load_config(Path("model/config.toml"))
assert config["preprocessing"] == {
    "mad_width": 3.0,
    "date_chunk_size": 60,
    "dtype": "float32",
    "expected_cross_section_size": 1000,
}
assert config["models"]["ridge"]["lambda_grid"] == [
    1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0
]
args = make_parser().parse_args(
    ["train", "--model", "ridge", "--horizon", "all", "--run-id", "ridge_baseline"]
)
assert (args.command, args.model, args.horizon, args.run_id) == (
    "train", "ridge", "all", "ridge_baseline"
)
```

- [ ] **Step 2: Run the check and verify it fails**

Run:

```powershell
.\.venv\Scripts\python.exe -B .\model\.ridge_checks\test_config.py
```

Expected: failure because the model sections and `train` parser do not exist.

- [ ] **Step 3: Add exact configuration**

Append to `model/config.toml`:

```toml
[preprocessing]
mad_width = 3.0
date_chunk_size = 60
dtype = "float32"
expected_cross_section_size = 1000

[models.ridge]
lambda_grid = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
solver = "cholesky"
rank_ic_tolerance = 0.000000000001
```

Create `model/requirements.txt`:

```text
pandas
numpy
pyarrow
rqdatac
scikit-learn
joblib
```

Extend `validate_config` with positive numeric `mad_width`, positive integer chunk and cross-section sizes, exact `float32` dtype, a strictly increasing nonnegative lambda grid, `cholesky` solver, and nonnegative tolerance. Add the `train` parser but defer command execution to Task 3.

- [ ] **Step 4: Install dependencies and rerun the check**

Run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\model\requirements.txt
.\.venv\Scripts\python.exe -B .\model\.ridge_checks\test_config.py
```

Expected: check exits with code 0 and `sklearn` plus `joblib` import successfully.

### Task 2: Leakage-Safe Cross-Sectional Preprocessing

**Files:**
- Create: `model/stages/preprocessing.py`
- Test temporarily: `model/.ridge_checks/test_preprocessing.py`

**Interfaces:**
- Consumes: a DataFrame containing `date`, `stock_code`, and the ordered raw factor columns from `model_dataset_schema.json`
- Produces:

```python
def derived_feature_columns(factor_columns: list[str]) -> list[str]: ...

def transform_daily_cross_sections(
    frame: pd.DataFrame,
    factor_columns: list[str],
    *,
    mad_width: float,
    expected_cross_section_size: int,
    date_chunk_size: int,
) -> tuple[np.ndarray, list[str]]: ...
```

The returned matrix is `float32`, ordered as 17 standardized factor columns followed by 17 `<factor>__missing` flags.

- [ ] **Step 1: Write failing numerical checks**

Use a two-date fixture with four stocks per date and factors covering: ordinary values, one missing value, `inf`, an extreme outlier, all missing, MAD equal to zero, and standard deviation equal to zero. Assert:

```python
assert matrix.dtype == np.float32
assert columns == ["f1", "f2", "f1__missing", "f2__missing"]
assert np.isfinite(matrix).all()
assert matrix[missing_row, f1_index] == 0.0
assert matrix[missing_row, f1_missing_index] == 1.0
assert np.allclose(nonmissing_daily_zscores.mean(axis=0), 0.0, atol=1e-6)
```

Also assert that duplicate `(date, stock_code)` keys and a date with the wrong number of rows raise `ValueError`.

- [ ] **Step 2: Run the check and verify it fails**

Run:

```powershell
.\.venv\Scripts\python.exe -B .\model\.ridge_checks\test_preprocessing.py
```

Expected: import failure because `model.stages.preprocessing` does not exist.

- [ ] **Step 3: Implement the daily transform**

For each date and factor:

```text
finite x
missing = not finite
median = median(finite x)
MAD = median(abs(finite x - median))
clip bounds = median +/- mad_width * 1.4826 * MAD
z = (clipped x - mean(clipped finite x)) / population_std(clipped finite x)
```

If MAD is zero or invalid, leave finite values unclipped. If fewer than two finite values remain or standard deviation is zero/nonfinite, set finite z-scores to zero. Fill missing z-scores with zero and preserve the missing flag as one. Validate the full key set before processing and print progress once per configured date chunk.

- [ ] **Step 4: Run the check and verify it passes**

Run:

```powershell
.\.venv\Scripts\python.exe -B .\model\.ridge_checks\test_preprocessing.py
```

Expected: exit code 0.

### Task 3: Ridge Training, Selection, And Atomic Run Bundle

**Files:**
- Create: `model/stages/training.py`
- Create: `model/runs/.gitignore`
- Modify: `model/pipeline.py`
- Test temporarily: `model/.ridge_checks/test_training.py`

**Interfaces:**
- Consumes:
  - `model/data/model_dataset.parquet`
  - `model/data/model_dataset_schema.json`
  - `model/data/model_sample_index.parquet`
  - `model/data/model_split_summary.json`
  - validated model configuration
- Produces:

```python
def mean_daily_rank_ic(
    dates: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[float, int]: ...

def train_ridge_run(
    config: dict[str, Any],
    config_path: Path,
    *,
    horizons: list[int],
    run_id: str,
) -> Path: ...
```

- [ ] **Step 1: Write failing behavior checks**

Create a small Parquet fixture spanning train, validation, and test dates. Assert:

```python
assert selected_lambda == expected_lambda
assert manifest["development_data"]["max_date"] <= "2023-12-31"
assert manifest["uses_entry_tradeable"] is False
assert set(predictions["split"]) == {"validation"}
assert set(predictions["horizon"]) == {1, 5, 10}
assert predictions["date"].max() <= pd.Timestamp("2023-12-31")
```

Assert that the scaler means equal means from training rows only, that the same transformed cross-section matrix is reused by all horizons, that an existing run ID raises `FileExistsError`, and that an injected failure leaves no formal run directory.

- [ ] **Step 2: Run the check and verify it fails**

Run:

```powershell
.\.venv\Scripts\python.exe -B .\model\.ridge_checks\test_training.py
```

Expected: import failure because `model.stages.training` does not exist.

- [ ] **Step 3: Implement development-only loading**

Validate the dataset and sample-index hashes first. Read only these model columns through PyArrow with `date <= validation.end`:

```text
date
stock_code
17 schema feature_columns
target_1d
target_5d
target_10d
```

Read only these sample-index columns with the same date filter:

```text
date
stock_code
split_1d
split_5d
split_10d
```

Require exact one-to-one keys, sorted keys, exactly 1000 rows per date, no dates after validation end, and exact raw-feature ordering from the schema. Do not request metadata or `entry_*` columns from Parquet.

- [ ] **Step 4: Implement metric and deterministic tuning**

For every horizon and lambda:

```python
scaler = StandardScaler().fit(X_train)
alpha = candidate_lambda * len(X_train)
model = Ridge(alpha=alpha, solver="cholesky", fit_intercept=True)
model.fit(scaler.transform(X_train), y_train)
y_pred = model.predict(scaler.transform(X_validation))
```

Compute each date's Spearman correlation from average ranks only when at least two rows exist and both rank vectors are nonconstant. Average valid dates with equal date weights. Compute RMSE over all validation rows. Select by:

```text
highest mean daily Rank IC
then lower RMSE when Rank IC differs by <= 1e-12
then larger lambda when both remain tied
```

- [ ] **Step 5: Implement artifacts and atomic publication**

Reject run IDs outside `[A-Za-z0-9][A-Za-z0-9._-]{0,63}`. Write all artifacts to `model/runs/.<run_id>.<random>.tmp/`; only rename to `model/runs/<run_id>/` after all horizons succeed:

```text
config_snapshot.toml
manifest.json
preprocessor_<horizon>d.joblib
model_<horizon>d.joblib
predictions.parquet
```

The prediction schema is exactly:

```text
date,stock_code,split,horizon,y_true,y_pred
```

The manifest records source/config hashes, git commit, environment versions, raw/derived features, preprocessing settings, all lambda candidate metrics, selected lambda and sklearn alpha, train/validation counts and date ranges, Rank IC valid-date counts, the development read range, and `uses_entry_tradeable=false`.

Create `model/runs/.gitignore`:

```text
*
!.gitignore
```

- [ ] **Step 6: Wire the public train command**

Map `--horizon all` to `[1, 5, 10]` and a numeric value to one horizon. `main()` calls `train_ridge_run` only for `train --model ridge`; no evaluation or locked-test behavior is added.

- [ ] **Step 7: Run the behavior check**

Run:

```powershell
.\.venv\Scripts\python.exe -B .\model\.ridge_checks\test_training.py
```

Expected: exit code 0, no test fixture artifacts outside `model/.ridge_checks/`.

### Task 4: Documentation And Doctor

**Files:**
- Modify: `model/README.md`
- Modify: `model/pipeline.py`
- Test temporarily: `model/.ridge_checks/test_doctor.py`

**Interfaces:**
- Consumes: complete Ridge configuration and local model dependencies
- Produces: `doctor` validation for `scikit-learn` and `joblib`, plus exact install/train/output documentation

- [ ] **Step 1: Write the failing doctor check**

Assert `run_doctor` reports:

```text
dependencies: pandas, numpy, pyarrow, rqdatac, sklearn, joblib
```

and rejects an invalid lambda grid, unsafe run ID, or missing model artifact source bundle before training.

- [ ] **Step 2: Run the check and verify it fails**

Run:

```powershell
.\.venv\Scripts\python.exe -B .\model\.ridge_checks\test_doctor.py
```

Expected: failure because the dependency message and Ridge checks are incomplete.

- [ ] **Step 3: Update doctor and README**

Document these exact commands:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\model\requirements.txt
.\.venv\Scripts\python.exe -m model.pipeline doctor
.\.venv\Scripts\python.exe -m model.pipeline prepare-data
.\.venv\Scripts\python.exe -m model.pipeline train `
  --model ridge `
  --horizon all `
  --run-id ridge_baseline
```

Explain 34 features, train/validation boundaries, lambda selection, run output schema, no `entry_tradeable` filtering, no 2024-2025 access, and that final refit/test/evaluation are later commands.

- [ ] **Step 4: Run the doctor check**

Run:

```powershell
.\.venv\Scripts\python.exe -B .\model\.ridge_checks\test_doctor.py
.\.venv\Scripts\python.exe -m model.pipeline doctor
```

Expected: both commands exit with code 0.

### Task 5: Real-Data Acceptance And Cleanup

**Files:**
- Generate and retain: `model/runs/ridge_baseline/`
- Generate temporarily: `model/runs/ridge_baseline_repro/`
- Delete: `model/.ridge_checks/`
- Delete: `model/runs/ridge_baseline_repro/`
- Delete: all `model/**/__pycache__/`

**Interfaces:**
- Consumes: the existing 1,699,000-row real model dataset and sample index
- Produces: one accepted `ridge_baseline` run bundle

- [ ] **Step 1: Run syntax and configuration verification**

Run:

```powershell
.\.venv\Scripts\python.exe -B -m compileall -q .\model
.\.venv\Scripts\python.exe -m model.pipeline doctor
```

Expected: both exit with code 0.

- [ ] **Step 2: Run the real three-horizon baseline**

Run:

```powershell
.\.venv\Scripts\python.exe -m model.pipeline train `
  --model ridge `
  --horizon all `
  --run-id ridge_baseline
```

Expected: one complete run containing three scaler files, three model files, a manifest, snapshot, and validation predictions.

- [ ] **Step 3: Validate the run contract**

Read the manifest and predictions and assert:

```text
raw features = 17
derived features = 34
horizons = 1, 5, 10
prediction split = validation only
maximum prediction/read date <= 2023-12-31
no entry field in raw or derived features
uses_entry_tradeable = false
all predictions and y_true values finite
keys unique by date, stock_code, horizon
manifest source hashes equal current source hashes
```

- [ ] **Step 4: Verify reproducibility and overwrite protection**

Run the same fixed input as `ridge_baseline_repro`; require identical selected lambdas, scaler arrays, model coefficients/intercepts, and prediction values. Then rerun with `--run-id ridge_baseline` and require `FileExistsError` without changing the accepted run hashes.

- [ ] **Step 5: Remove temporary checks and caches**

Delete `model/.ridge_checks/`, `model/runs/ridge_baseline_repro/`, failed temporary run directories, and all `model/**/__pycache__/`. Keep only `model/runs/.gitignore` and the accepted `model/runs/ridge_baseline/` output.

- [ ] **Step 6: Final review**

Run:

```powershell
git diff --check
git status --short
```

Inspect only `model/` changes, confirm no unrelated file was modified, and confirm no temporary test file remains.
