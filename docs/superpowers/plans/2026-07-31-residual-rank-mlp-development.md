# Residual Rank MLP Development Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and run a development-only `residual_rank_mlp` model that combines a fixed Ridge baseline, purged expanding OOF residual learning, and a same-date group-distance-weighted Pairwise loss without reading 2024-2025.

**Architecture:** Reuse the existing unified `model.pipeline train` entry and the existing feature preprocessing. Extend `model/stages/training.py` with deterministic OOF Ridge, decile, pair-sampling, residual-MLP, selection, artifact, and manifest helpers; keep every horizon inside one registered model implementation. Publish only a completed development run after every validation passes, and stop before the rank stage if `gamma=0` wins.

**Tech Stack:** Python 3.12, pandas, NumPy, scikit-learn Ridge/StandardScaler, PyTorch CPU, Parquet, TOML.

## Global Constraints

- All source changes stay under `model/`; documentation stays under `docs/superpowers/`.
- Preserve Ridge, XGBoost, LightGBM, and MLP baseline behavior and run directories.
- Use only 2019-2023 during development; never read 2024-2025.
- Keep raw absolute-return targets; `target_standardization=false`.
- Never use `entry_tradeable` or any T+1 tradeability field for training.
- Final Top100 score is `ridge_pred + gamma * residual_pred`; no second inference head.
- Fixed Ridge regularization is `lambda=1`, with `alpha=lambda*n_train`.
- Candidate grids are `gamma=[0,0.25,0.5,1]` and `lambda_rank=[0,0.1,0.25,0.5]`.
- Pair sampling uses complete dates, eight opponents per stock, and deterministic seeds.
- Temporary tests, temporary runs, and caches are removed after verification.
- Do not implement finalization or locked-test prediction until the development gate accepts a nonzero residual model.

---

### Task 1: Register and validate the model configuration

**Files:**
- Modify: `model/config.toml`
- Modify: `model/pipeline.py`

**Interfaces:**
- Consumes: existing `[models.mlp]` fixed optimizer and network conventions.
- Produces: validated `config["models"]["residual_rank_mlp"]` and `train --model residual_rank_mlp` dispatch.

- [ ] **Step 1: Run a failing registration check**

Run:

```powershell
@'
from pathlib import Path
from model.pipeline import load_config, make_parser
c = load_config(Path("model/config.toml"))
assert "residual_rank_mlp" in c["models"]
assert "residual_rank_mlp" in make_parser()._subparsers._group_actions[0].choices["train"]._actions[1].choices
'@ | .\.venv\Scripts\python.exe -B -
```

Expected: FAIL because the model section and CLI choice do not exist.

- [ ] **Step 2: Add the centralized configuration**

Add `[models.residual_rank_mlp]` with these exact values:

```toml
ridge_lambda = 1.0
gamma_grid = [0.0, 0.25, 0.5, 1.0]
rank_lambda_grid = [0.0, 0.1, 0.25, 0.5]
dates_per_batch = 8
pairs_per_stock = 8
adjacent_pair_fraction = 0.5
top_group_weight = 2.0
second_group_weight = 1.5
huber_mad_multiplier = 1.5
rank_ic_tolerance = 0.0005
max_epochs = 80
early_stopping_patience = 10
early_stopping_min_delta = 0.00000001
learning_rate = 0.001
lr_scheduler_patience = 3
lr_scheduler_factor = 0.5
min_learning_rate = 0.00001
gradient_clip_norm = 1.0
num_threads = 4
seed = 42
stability_seeds = [42, 43, 44]
target_standardization = false
candidates = [
  { name = "small", hidden_layers = [64, 32], dropout = 0.1, weight_decay = 0.0001 },
  { name = "balanced", hidden_layers = [128, 64, 32], dropout = 0.1, weight_decay = 0.0001 },
  { name = "regularized", hidden_layers = [128, 64, 32], dropout = 0.3, weight_decay = 0.001 },
]
```

- [ ] **Step 3: Validate every field in `validate_config`**

Reject missing grids, values outside the documented ranges, grids without zero, non-unique values, nonpositive date/pair counts, non-boolean target standardization, stability seeds other than three unique integers, and malformed candidate structures.

- [ ] **Step 4: Register the trainer**

Add `residual_rank_mlp` to the train CLI choices and dispatch it to:

```python
train_residual_rank_mlp_run(
    config: dict[str, Any],
    config_path: Path,
    *,
    horizons: list[int],
    run_id: str,
) -> Path
```

- [ ] **Step 5: Run registration and doctor checks**

Run:

```powershell
.\.venv\Scripts\python.exe -B -m model.pipeline --config .\model\config.toml doctor
```

Expected: all existing doctor checks remain `[ok]`.

### Task 2: Add purged OOF Ridge and decile foundations

**Files:**
- Modify: `model/stages/training.py`

**Interfaces:**
- Consumes: `_load_development_data`, `transform_daily_cross_sections`, `StandardScaler`, `Ridge`, sample-index `exit_date_{h}d`.
- Produces:
  - `_load_residual_rank_data(...)`
  - `build_true_deciles(dates, targets) -> np.ndarray`
  - `build_purged_oof_ridge(...) -> dict[str, Any]`
  - `residual_rank_metrics(...) -> dict[str, float | int]`

- [ ] **Step 1: Run failing helper checks**

Use an inline synthetic frame to import the three new helpers. Assert that tied returns receive the same decile, OOF fold training exits before its prediction year, and `gamma=0` equals Ridge. Expected: import failure before implementation.

- [ ] **Step 2: Load exit dates without changing baseline readers**

Implement `_load_residual_rank_data` by reusing published lineage checks while reading:

```text
date, stock_code
17 raw factors
target_{h}d
split_{h}d
exit_date_{h}d
```

Require unique keys, finite selected targets, `uses_entry_tradeable=false`, and a maximum development date no later than 2023-12-31.

- [ ] **Step 3: Implement deterministic true deciles**

For each date, descending-average-rank finite targets and assign:

```python
group = min(10, ceil(10 * average_rank / n))
```

Return `int8`; reject dates with fewer than ten finite rows; preserve ties.

- [ ] **Step 4: Implement expanding OOF Ridge**

For prediction years 2020, 2021, and 2022:

```text
train_mask = split=="train"
             and date<prediction_year_start
             and exit_date<prediction_year_start
predict_mask = split=="train"
               and date within prediction year
```

Fit a fold-local scaler and Ridge with `alpha=n_fold_train`; predict every OOF row exactly once. Also fit the development Ridge and scaler on all 2019-2022 `split=="train"` rows for 2023 validation.

Return OOF predictions, validation predictions, fold boundaries/counts, fitted development Ridge/scaler, and masks.

- [ ] **Step 5: Implement common diagnostics**

Given date, target, and final prediction, compute Rank IC, IC standard deviation, ICIR, positive IC rate, hard decile MAE, Top100 recall, Top100 mean target, MSE, RMSE, MAE, R2, and prediction standard deviation. Use deterministic `stock_code` only to break predicted Top100 ties, never true-return ties.

- [ ] **Step 6: Run synthetic and real-data boundary checks**

Expected real boundaries must match the approved design:

```text
1d residual dates/rows: 726 / 726000
5d residual dates/rows: 722 / 722000
10d residual dates/rows: 717 / 717000
```

Assert every fold `max(train_exit_date) < min(oof_prediction_date)`.

### Task 3: Add deterministic date batches and Pairwise loss

**Files:**
- Modify: `model/stages/training.py`

**Interfaces:**
- Consumes: OOF/validation final predictions, true deciles, raw targets, complete-date row slices.
- Produces:
  - `build_date_slices(dates) -> list[np.ndarray]`
  - `sample_group_weighted_pairs(...) -> dict[str, np.ndarray]`
  - `group_weighted_pairwise_loss(...) -> torch.Tensor`

- [ ] **Step 1: Run a failing pair sampler check**

Create two dates with 100 stocks each and assert the future helper never crosses dates, never emits same-group/equal-return pairs, includes every eligible anchor, emits finite positive weights, and reproduces exactly for the same seed/epoch/date.

- [ ] **Step 2: Build complete-date batches**

Shuffle date indices, not rows. Concatenate approximately eight complete dates per optimization batch. Require each date slice to contain one contiguous date and preserve all its stock rows.

- [ ] **Step 3: Implement vectorized stratified pair sampling**

Within each true group, vectorize opponent draws for every anchor:

- Four draws from groups at distance one when available.
- Four draws from groups at distance at least three when available.
- Fall back to any different group only when a requested pool is empty.
- Canonicalize and deduplicate pairs.
- Remove equal-return pairs.
- Seed from the fixed integer representation of date plus global seed and epoch; never use Python's randomized `hash()`.

- [ ] **Step 4: Implement pair weights and loss**

Compute group-distance and Top-group weights, normalize pair weights to mean one within each date, divide prediction differences by the train-only horizon temperature, use `softplus(-sign*prediction_difference)`, and divide the date loss by `log(2)`.

- [ ] **Step 5: Verify gradients and lambda zero**

Assert the loss and gradients are finite. Assert `lambda_rank=0` skips pair construction and pair gradients entirely. Benchmark one synthetic epoch and one real date to ensure pair generation is not the dominant memory allocation.

### Task 4: Implement staged residual and rank-aware training

**Files:**
- Modify: `model/stages/training.py`

**Interfaces:**
- Consumes: Tasks 2-3 helpers and `[models.residual_rank_mlp]`.
- Produces:
  - `_train_residual_candidate(...)`
  - `_train_rank_candidate(...)`
  - `choose_residual_candidate(...)`
  - `choose_rank_candidate(...)`
  - `train_residual_rank_mlp_run(...)`

- [ ] **Step 1: Run failing selector tests**

Create candidate dictionaries proving the exact order:

```text
Rank IC
then group MAE within 0.0005
then RMSE
then smaller lambda / earlier epoch
```

Expected: imports fail before implementation.

- [ ] **Step 2: Implement train-only loss constants**

For each horizon calculate from OOF rows only:

```text
huber_delta = 1.5 * 1.4826 * MAD(oof_residual)
ridge_huber_baseline
temperature = median daily cross-sectional target std
```

Reject zero/nonfinite constants.

- [ ] **Step 3: Implement Stage C**

Train each of the three architectures against OOF residual Huber loss. Initialize the output weight to zero and bias to the OOF residual mean. At every epoch evaluate every gamma on 2023 final predictions, checkpoint by the approved Rank-IC/group-MAE/RMSE/epoch order, and stop after ten non-improving epochs.

- [ ] **Step 4: Enforce the Stage C gate**

If the selected gamma is zero, publish a completed rejected-development manifest with Ridge-equivalent predictions and:

```json
{
  "development_decision": "reject_residual",
  "rank_stage_executed": false
}
```

Do not run Stage D.

- [ ] **Step 5: Implement Stage D**

Fix the Stage C architecture and gamma. Train four models from scratch using normalized Huber plus `lambda_rank * pair_loss`. Include `lambda_rank=0` as the strict control. Schedule learning rate on validation total loss and checkpoint using Rank IC, group MAE, RMSE, and epoch.

- [ ] **Step 6: Enforce the Stage D gate**

Accept a positive rank lambda only if it beats the zero control in Rank IC, lowers group MAE, does not lower Top100 recall, and has finite/non-explosive prediction standard deviation. Otherwise select the zero control and record `reject_rank_loss`.

- [ ] **Step 7: Add stability seeds**

Reuse the selected seed-42 checkpoint, retrain seeds 43 and 44 with the frozen selected configuration, and store metric mean/std and individual results. Reject a configuration whose Rank IC direction is not positive for all three seeds.

### Task 5: Publish auditable development artifacts

**Files:**
- Modify: `model/stages/training.py`

**Interfaces:**
- Consumes: selected Stage C/D result.
- Produces: one atomic `model/runs/<run_id>` development run.

- [ ] **Step 1: Save model components**

For each selected horizon save:

```text
ridge_model_{h}d.joblib
ridge_preprocessor_{h}d.joblib
residual_mlp_{h}d.pt
mlp_preprocessor_{h}d.joblib
```

The PyTorch payload includes architecture, feature order, output initialization, gamma, lambda, horizon constants, state dict, and `target_standardization=false`.

- [ ] **Step 2: Save compatible predictions**

Write `predictions.parquet` with:

```csv
date,stock_code,split,horizon,y_true,y_pred
```

Write `component_predictions.parquet` with:

```csv
date,stock_code,horizon,ridge_pred,residual_pred,gamma,final_pred
```

Assert `final_pred == ridge_pred + gamma*residual_pred` within float32 tolerance.

- [ ] **Step 3: Write the manifest**

Record source hashes, features, OOF folds, exit-date boundaries, all training constants/candidates/checkpoints, Stage gates, seed diagnostics, artifacts, sample counts, and:

```json
{
  "status": "completed",
  "model_name": "residual_rank_mlp",
  "test_data_read": false,
  "uses_entry_tradeable": false,
  "target_standardization": false
}
```

Use a temporary run directory and atomic publish; delete it on any exception.

- [ ] **Step 4: Restore artifacts**

Load every saved Ridge/scaler/MLP component, recompute the full 2023 predictions, and require equality with the published predictions at configured CPU thread count.

### Task 6: Document, verify, and run the full development experiment

**Files:**
- Modify: `model/README.md`
- Temporary only: inline tests and `model/runs/residual_rank_mlp_repro_check`

**Interfaces:**
- Consumes: the completed registered model.
- Produces: `model/runs/residual_rank_mlp_development` and a user-facing result comparison.

- [ ] **Step 1: Document the model and command**

Explain the fixed Ridge, OOF residual folds, same-date pair loss, Stage gates, artifact files, raw target, and development command.

- [ ] **Step 2: Run static and environment checks**

Run AST parsing, `doctor`, config validation, `git diff --check`, and all inline helper acceptance tests. Remove generated `__pycache__`.

- [ ] **Step 3: Run a short real-data smoke experiment**

Use one horizon and reduced epochs through an in-memory config override and a temporary run ID. Validate OOF counts, pair rules, progress logging, atomic cleanup, and artifact restore; remove the temporary run.

- [ ] **Step 4: Run the full experiment**

Run:

```powershell
.\.venv\Scripts\python.exe -B -m model.pipeline `
  --config .\model\config.toml train `
  --model residual_rank_mlp `
  --horizon all `
  --run-id residual_rank_mlp_development
```

Allow Stage C to stop the run before Stage D when the approved gate fails.

- [ ] **Step 5: Verify the official run**

Require:

```text
only 2019-2023 was read
all validation keys/targets equal the four baselines
no duplicate/nonfinite predictions
gamma=0 exactly reproduces Ridge when selected
component addition holds
saved models restore
manifest hashes and flags match
```

- [ ] **Step 6: Reproduce and clean**

If Stage C/D reaches a selected nonzero model, run the same configuration as `residual_rank_mlp_repro_check`, compare predictions, metrics, weights, preprocessors, and fold metadata exactly, then delete the repro run and all caches.

- [ ] **Step 7: Report the development gate**

Compare Ridge, MLP, residual-only, and residual-rank results by horizon. State explicitly whether residual learning and rank loss were accepted or rejected. Do not implement or run 2024-2025 finalization until this gate is reviewed.
