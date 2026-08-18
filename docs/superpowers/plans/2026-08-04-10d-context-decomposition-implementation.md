# 10d Context and Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible development-only experiment that builds leakage-safe market and industry context, compares direct and decomposed 10-day absolute-return models with expanding OOF selection, calibrates them without using 2023, and evaluates all frozen candidates once on 2023.

**Architecture:** Keep `model/pipeline.py` as the only public entry. Add `model/stages/context.py` for deterministic local feature/label construction and `model/stages/context_training.py` for OOF selection, calibration, models, metrics, gating, and artifacts. Reuse the existing factor preprocessing, sample index, model registry, and run-directory conventions; do not read test-period rows during development.

**Tech Stack:** Python 3.11, pandas, NumPy, PyArrow, scikit-learn, LightGBM, PyTorch, joblib, standard-library `unittest` for temporary TDD tests.

## Global Constraints

- All production changes stay under `model/`; the implementation plan stays under `docs/superpowers/plans/`.
- The experiment supports only `target_10d = post_open(T+11) / post_open(T+1) - 1`.
- Development reads at most 2023-12-31; 2024-2025 cannot enter features, fitting, calibration, selection, or evaluation.
- OOF folds are 2020 from purged 2019, 2021 from purged 2019-2020, and 2022 from purged 2019-2021.
- 2023 is evaluated only after all candidate structures, parameters, and official raw/calibrated output choices are frozen.
- Training never filters on `entry_tradeable` or T+1 state.
- Target values remain in raw return units and validation targets are not clipped.
- Temporary tests are removed after all verification because the project owner does not want added test files retained.

---

### Task 1: Context Data Contract and Point-in-Time Aggregates

**Files:**
- Create: `model/stages/context.py`
- Create temporarily: `model/_tmp_context_tests.py`

**Interfaces:**
- Produces: `build_daily_context(market: pd.DataFrame, calendar: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]`
- Produces: `decompose_target(frame: pd.DataFrame, shrink_k: float) -> pd.DataFrame`
- Produces: `build_context_dataset(...) -> pd.DataFrame`

- [ ] **Step 1: Write failing pure-function tests**

Create synthetic dates, two industries, a changing industry label, missing market cap, invalid limits, and one missing target. Assert that stock returns use prior official dates, rolling market returns use historical daily members, cap coverage triggers equal-weight fallback below 95%, `UNKNOWN` receives zero industry effect, and valid rows satisfy:

```python
np.testing.assert_allclose(
    result.loc[valid, "target_10d"],
    result.loc[valid, "market_target_10d"]
    + result.loc[valid, "industry_target_10d"]
    + result.loc[valid, "alpha_target_10d"],
    rtol=0,
    atol=1e-12,
)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.\.venv\Scripts\python.exe -m unittest model._tmp_context_tests -v`

Expected: import failure because `model.stages.context` does not exist.

- [ ] **Step 3: Implement market and industry context**

Implement adjusted-close daily returns, equal/cap-weighted market return series, 1/5/20-day compounding, breadth, dispersion, quantiles, limit/ST/suspension rates, market-cap distribution/coverage, and point-in-time industry equivalents. Reindex each industry to the official calendar before rolling so missing industry dates cannot silently shorten a window.

- [ ] **Step 4: Implement target decomposition and dataset publication**

Use:

```python
market = target.groupby("date").transform("mean")
industry_raw = target.groupby(["date", "industry"]).transform("mean") - market
industry = n / (n + shrink_k) * industry_raw
alpha = target - market - industry
```

Set the `UNKNOWN` industry effect to zero, retain missing target rows, join existing 34 factor value/missing features, add log-cap handling, add context missing flags, verify unique keys and exact reconstruction, and atomically write Parquet plus a JSON schema with source hashes.

- [ ] **Step 5: Run tests and verify GREEN**

Run: `.\.venv\Scripts\python.exe -m unittest model._tmp_context_tests -v`

Expected: all Task 1 tests pass.

### Task 2: Fold-Safe Encoding, OOF Selection, and Calibration

**Files:**
- Create: `model/stages/context_training.py`
- Modify temporarily: `model/_tmp_context_tests.py`

**Interfaces:**
- Produces: `fit_preprocessor(frame, continuous_columns, industry_column) -> ContextPreprocessor`
- Produces: `iter_expanding_folds(dates, exit_dates, years=(2020, 2021, 2022))`
- Produces: `select_by_oof_rmse(records, complexity_key, prefer_larger=False) -> dict`
- Produces: `fit_calibrator(y_true, y_pred) -> CalibrationResult`

- [ ] **Step 1: Add failing tests for fold boundaries and preprocessing**

Assert each fold has `max(train_exit) < Jan-01(prediction_year)`, unknown validation industries map to the fixed `UNKNOWN` column, scalers only see training rows, Ridge ties choose larger lambda, LightGBM ties choose fewer rounds, and calibration with nonpositive slope locks raw predictions.

- [ ] **Step 2: Run tests and verify RED**

Run: `.\.venv\Scripts\python.exe -m unittest model._tmp_context_tests -v`

Expected: missing OOF/preprocessor/calibrator symbols.

- [ ] **Step 3: Implement deterministic preprocessing and OOF utilities**

Use training-only `StandardScaler` for continuous features and training categories plus fixed `UNKNOWN` for dense float32 One-Hot output. Return serializable preprocessors and explicit feature names. Reject overlapping folds, nonfinite selected labels, duplicate keys, and any selected date after 2023-12-31.

- [ ] **Step 4: Implement calibration**

Fit equal-row-weight OLS:

```python
slope, intercept = np.polyfit(y_pred, y_true, 1)
use_calibrated = bool(np.isfinite(slope) and slope > 0)
```

Store raw and calibrated predictions; the official version is fixed from the OOF slope before 2023 is loaded.

- [ ] **Step 5: Run tests and verify GREEN**

Run: `.\.venv\Scripts\python.exe -m unittest model._tmp_context_tests -v`

Expected: all Task 1-2 tests pass.

### Task 3: Metrics, HAC, and Acceptance Diagnostics

**Files:**
- Modify: `model/stages/context_training.py`
- Modify temporarily: `model/_tmp_context_tests.py`

**Interfaces:**
- Produces: `return_metrics(frame: pd.DataFrame) -> dict[str, object]`
- Produces: `hac_mean_test(values: np.ndarray, lag: int = 10) -> dict[str, float]`
- Produces: `compare_to_champion(candidate, champion) -> dict[str, object]`

- [ ] **Step 1: Add failing metric tests**

Use hand-computable predictions to assert RMSE, MAE, R2, MSE Skill, calibration, daily RankIC, positive IC rate, decile MAE, Top100 mean target, NDCG@100, `abs(target)<=0.5` robust RMSE, extreme MSE contribution, direction accuracy, and a finite Newey-West confidence interval.

- [ ] **Step 2: Run tests and verify RED**

Run: `.\.venv\Scripts\python.exe -m unittest model._tmp_context_tests -v`

Expected: metric functions are missing.

- [ ] **Step 3: Implement metrics and HAC without new dependencies**

Compute date-equal RankIC summaries and daily MSE differences. Implement Bartlett-weight Newey-West variance with `lag=10`, normal 95% interval, percent of dates improved, and month-level RMSE differences. Return JSON-safe Python scalars only.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `.\.venv\Scripts\python.exe -m unittest model._tmp_context_tests -v`

Expected: all Task 1-3 tests pass.

### Task 4: Stage-One Models and Decomposition

**Files:**
- Modify: `model/stages/context_training.py`
- Modify temporarily: `model/_tmp_context_tests.py`

**Interfaces:**
- Produces: `run_context_ridge(...) -> CandidateResult`
- Produces: `run_context_lightgbm(...) -> CandidateResult`
- Produces: `run_decomposed_ridge(...) -> CandidateResult`

- [ ] **Step 1: Add failing small-data OOF model tests**

Build three synthetic OOF years and assert Ridge evaluates lambdas `[0.1, 1, 10]`, LightGBM evaluates predictions at `[10, 20, 40, 80, 120, 200]` from one max-round model per fold, and decomposed raw predictions exactly equal market plus industry plus alpha predictions.

- [ ] **Step 2: Run tests and verify RED**

Run: `.\.venv\Scripts\python.exe -m unittest model._tmp_context_tests -v`

Expected: model runners are missing.

- [ ] **Step 3: Implement direct Ridge and LightGBM**

Select candidates only by pooled raw OOF RMSE. Fit selected models on purged 2019-2022, apply the OOF calibrator, and then predict 2023 once. LightGBM fixes `num_leaves=31`, `max_depth=5`, `min_data_in_leaf=1000` and does not early-stop on 2023.

- [ ] **Step 4: Implement decomposed Ridge**

Train market rows by date, industry rows by date-industry with `industry_target_count` sample weights, and stock alpha rows. Select each component lambda independently by component OOF RMSE, then report stock-level sum RMSE and exact component identity.

- [ ] **Step 5: Run tests and verify GREEN**

Run: `.\.venv\Scripts\python.exe -m unittest model._tmp_context_tests -v`

Expected: all Task 1-4 tests pass.

### Task 5: Champion OOF Gate and Fixed Residual MLPs

**Files:**
- Modify: `model/stages/context_training.py`
- Modify temporarily: `model/_tmp_context_tests.py`

**Interfaces:**
- Produces: `run_fixed_residual_mlp(...) -> CandidateResult`
- Produces: `run_context_experiment(config, config_path, run_id) -> Path`

- [ ] **Step 1: Add failing fixed-MLP and gate tests**

Assert the MLP uses `[128,64,32]`, dropout `0.3`, weight decay `0.001`, gamma `1`, six epochs, seed `42`, no target standardization, no pair loss, and training-fold-only `delta = 1.5 * 1.4826 * MAD(residual)`. Assert Stage 2 is skipped unless a Stage-1 raw OOF RMSE is below the regenerated champion OOF RMSE.

- [ ] **Step 2: Run tests and verify RED**

Run: `.\.venv\Scripts\python.exe -m unittest model._tmp_context_tests -v`

Expected: fixed residual and orchestration symbols are missing.

- [ ] **Step 3: Implement fixed residual training**

Train date batches with AdamW and Huber for exactly six epochs. Regenerate the locked factor-only Ridge plus residual MLP OOF baseline on the same fold/sample keys. Only after the OOF gate passes, run Context Residual MLP and Decomposed Alpha Residual MLP.

- [ ] **Step 4: Implement atomic run artifacts**

Each candidate directory saves `manifest.json`, `predictions.parquet`, `feature_schema.json`, `context_feature_summary.json`, `calibrator.joblib`, preprocessors/models, and Decomposition `component_predictions.parquet`. Validate reloaded 2023 predictions and raw component identity before atomically publishing the run directory.

- [ ] **Step 5: Run tests and verify GREEN**

Run: `.\.venv\Scripts\python.exe -m unittest model._tmp_context_tests -v`

Expected: all Task 1-5 tests pass.

### Task 6: Unified CLI, Configuration, Documentation, and Cleanup

**Files:**
- Modify: `model/pipeline.py`
- Modify: `model/config.toml`
- Modify: `model/README.md`
- Delete: `model/_tmp_context_tests.py`

**Interfaces:**
- Adds CLI: `python -m model.pipeline prepare-context`
- Adds registered model: `python -m model.pipeline train --model context_decomposition --horizon 10 --run-id <id>`

- [ ] **Step 1: Add failing CLI/config tests**

Assert config validation requires context paths and exact grids, `prepare-context` dispatches only local builders, the new model rejects horizons other than `10`, and existing model choices still parse unchanged.

- [ ] **Step 2: Run tests and verify RED**

Run: `.\.venv\Scripts\python.exe -m unittest model._tmp_context_tests -v`

Expected: parser/config assertions fail before wiring.

- [ ] **Step 3: Wire configuration and commands**

Add `context_dataset` and `context_schema` paths, shrink `10`, cap coverage `0.95`, OOF years, exact lambda/round grids, fixed LightGBM/MLP parameters, champion run path, and HAC lag `10`. Add config validation and registry dispatch without creating model-specific entry files.

- [ ] **Step 4: Update README**

Document the two PowerShell commands, 2019-2023-only development boundary, output layout, OOF/calibration rules, Stage-2 gate, and the 2024-2025 contamination disclosure.

- [ ] **Step 5: Run focused and integration verification**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest model._tmp_context_tests -v
.\.venv\Scripts\python.exe -m model.pipeline doctor
.\.venv\Scripts\python.exe -m compileall model
```

Then run a synthetic end-to-end context experiment and a development-only real-data `prepare-context`. Confirm max date is 2023-12-31, every date has 1,000 keys, factor/context schemas match, decomposition identity holds, and no entry fields appear in model features.

- [ ] **Step 6: Remove temporary tests and rerun non-test checks**

Delete `model/_tmp_context_tests.py` with `apply_patch`, remove generated `__pycache__`, and rerun `doctor`, `compileall`, artifact schema checks, and `git diff --check`. Do not remove or modify existing user run outputs.
