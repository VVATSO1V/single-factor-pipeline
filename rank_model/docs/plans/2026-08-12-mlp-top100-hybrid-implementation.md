# Top100-Aware Hybrid MLP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible `mlp_top100_hybrid_rank` candidate that optimizes broad rank fit plus Top100 boundary ordering and selects its best epoch on 2023 validation NDCG@100.

**Architecture:** Keep the existing rank-model pipeline and run-bundle contract. Put deterministic pair sampling and validation ranking metrics in `rank_model/stages/ranking.py`; put parameter validation, batch training, early stopping, metadata, and persistence integration in `rank_model/stages/training.py`; expose the model through the existing TOML configuration and CLI registry.

**Tech Stack:** Python 3.11, pandas, NumPy, SciPy, PyTorch, TOML, stdlib `unittest`.

## Global Constraints

- Preserve all seven existing model implementations and immutable real run directories.
- Use only 2019-2022 rows for gradient updates and only 2023 rows for early stopping.
- Do not read or use 2024-2025 data.
- Keep `predictions_10d.parquet` and evaluation output schemas unchanged.
- Use seed `42`, deterministic PyTorch algorithms, eight complete dates per batch, maximum 30 epochs, patience 4, and NDCG minimum delta `0.0001` by default.
- Use loss `mean_date_rank_mse + 0.1 * mean_date_top100_pairwise_logistic` by default.
- Sample eight opponents per realized Top100 stock: four from rank `[0.7, 0.9)` and four from rank `<0.9`.
- Temporary test modules and generated test artifacts are deleted after final verification.

---

### Task 1: Deterministic Top100 Ranking Primitives

**Files:**
- Modify: `rank_model/stages/ranking.py`
- Create temporarily: `rank_model/tests/__init__.py`
- Create temporarily: `rank_model/tests/test_mlp_top100_hybrid.py`

**Interfaces:**
- Produces: `sample_top100_pairs(dates, targets, boundary_pairs_per_positive, broad_pairs_per_positive, seed) -> tuple[np.ndarray, np.ndarray, np.ndarray]`.
- Produces: `mean_daily_ndcg_at_k(dates, stock_codes, scores, targets, top_k) -> float`.
- Produces: `mean_daily_spearman(dates, scores, targets) -> float`.

- [ ] **Step 1: Write failing tests for deterministic same-date pair sampling**

Create two dates with ten evenly spaced labels. Assert every label at or above `0.9` receives exactly eight directed pairs, all opponents are below `0.9`, half belong to `[0.7, 0.9)`, no pair crosses dates, and repeated calls with seed 42 are identical.

- [ ] **Step 2: Run the sampling test and verify RED**

Run:

```powershell
& $python -m unittest rank_model.tests.test_mlp_top100_hybrid.Top100PairSamplingTests -v
```

Expected: import failure because `sample_top100_pairs` does not exist.

- [ ] **Step 3: Implement strict deterministic sampling**

Normalize dates, validate finite labels and positive integer pair counts, factorize by date, define positives as `target >= 0.9`, and sample with `np.random.default_rng(seed)`. Raise `ValueError` if a date lacks positives, boundary opponents, or broad opponents.

- [ ] **Step 4: Run the sampling tests and verify GREEN**

Run the command from Step 2. Expected: all pair sampling tests pass.

- [ ] **Step 5: Write failing tests for validation metrics**

Use two complete dates and assert perfect ordering gives NDCG and Spearman equal to `1.0`; reverse ordering gives lower NDCG and Spearman equal to `-1.0`; reject non-finite input and cross-column length mismatch.

- [ ] **Step 6: Implement mean daily NDCG@K and Spearman**

Use the evaluator's exact gain `exp2(rank_target_10d) - 1`, stable stock-code tie-breaking, per-date calculation, and equal averaging across dates.

- [ ] **Step 7: Run Task 1 tests and commit**

```powershell
& $python -m unittest rank_model.tests.test_mlp_top100_hybrid -v
git add rank_model/stages/ranking.py rank_model/tests
git commit -m "feat: add Top100 ranking primitives"
```

### Task 2: Hybrid Loss, Date-Batch Updates, And Early Stopping

**Files:**
- Modify: `rank_model/stages/training.py`
- Modify temporarily: `rank_model/tests/test_mlp_top100_hybrid.py`

**Interfaces:**
- Consumes: Task 1 sampling and validation metric functions.
- Produces: `train_mlp_top100_hybrid_rank(train, validation, schema, params) -> TrainingOutcome`.
- Produces metadata keys `best_epoch`, `stopped_epoch`, `best_validation_ndcg_100`, `best_validation_rank_ic`, `optimizer_step_count`, and `training_history`.

- [ ] **Step 1: Write a failing equal-date hybrid-loss test**

Construct a tensor batch with two date groups of unequal row counts. Assert the MSE term is the mean of the two date means and the pairwise term is the mean of the two date pair losses rather than a row- or pair-count-weighted aggregate.

- [ ] **Step 2: Verify RED**

Run the specific loss test. Expected: failure because the hybrid loss helper does not exist.

- [ ] **Step 3: Implement the minimal hybrid loss helper**

Return total loss and detached component values. Reject empty date groups, absent date pairs, and non-finite component losses.

- [ ] **Step 4: Verify GREEN**

Run the loss test and the complete temporary module.

- [ ] **Step 5: Write a failing small-training integration test**

Build synthetic 20-stock complete dates, a minimal schema, configurable `max_epochs=3`, `patience=1`, and `dates_per_batch=2`. Assert finite validation scores, at least one optimizer step per processed date batch, best-state restoration, deterministic repeated predictions, and complete early-stopping history.

- [ ] **Step 6: Implement parameter parsing and the trainer**

Validate all TOML fields and build the existing `[128, 64, 32]` network through a generalized MLP builder. For each epoch, resample pairs with `seed + epoch`; for each complete-date batch, compute the equal-date hybrid loss, backpropagate, clip gradients, and immediately call `optimizer.step()`. Score 2023 after each epoch, save a cloned CPU state on NDCG improvement greater than `min_delta`, stop after `patience` failures, and restore the best state.

- [ ] **Step 7: Verify deterministic training and commit**

```powershell
& $python -m unittest rank_model.tests.test_mlp_top100_hybrid -v
git add rank_model/stages/training.py rank_model/tests/test_mlp_top100_hybrid.py
git commit -m "feat: train Top100-aware hybrid MLP"
```

### Task 3: Registry, Configuration, Persistence, And CLI Compatibility

**Files:**
- Modify: `rank_model/config.toml`
- Modify: `rank_model/stages/training.py`
- Modify: `rank_model/README.md`
- Modify temporarily: `rank_model/tests/test_mlp_top100_hybrid.py`

**Interfaces:**
- Registers model name `mlp_top100_hybrid_rank` in `MODEL_NAMES` and `MODEL_REGISTRY`.
- Persists the new architecture through the existing `pytorch_mlp` run-bundle path.
- Keeps the standard `train`, `evaluate`, and `compare` commands unchanged.

- [ ] **Step 1: Write failing registry and parser tests**

Assert the new name appears in `MODEL_NAMES`, maps to the new trainer, and is accepted by `make_parser()` while all old names remain present.

- [ ] **Step 2: Verify RED**

Run the registry test. Expected: failure because the model is not registered.

- [ ] **Step 3: Add explicit TOML parameters and registration**

Add `[models.mlp_top100_hybrid_rank]` with hidden layers, dropout, learning rate, weight decay, loss weight, pair counts, date batch size, maximum epochs, patience, minimum delta, gradient clipping, and seed. Register the trainer without changing old entries.

- [ ] **Step 4: Write failing persistence test**

Train a small synthetic outcome, stage it in a temporary run directory, reload it through `_load_persisted_model`, and assert predictions match within `1e-6`. Assert architecture metadata identifies the new model and prediction columns equal the current contract.

- [ ] **Step 5: Extend persistence reconstruction and verify GREEN**

Allow the new architecture name, persisted widths, dropout, and output bias to reconstruct the generalized network. Run the complete temporary test module.

- [ ] **Step 6: Document the model and commit**

Add the model name, objective, early-stopping behavior, and PowerShell train/evaluate examples to `rank_model/README.md`.

```powershell
git add rank_model/config.toml rank_model/stages/training.py rank_model/README.md rank_model/tests/test_mlp_top100_hybrid.py
git commit -m "feat: register Top100-aware hybrid MLP"
```

### Task 4: Full Verification And Test Cleanup

**Files:**
- Verify: `rank_model/stages/ranking.py`
- Verify: `rank_model/stages/training.py`
- Verify: `rank_model/config.toml`
- Verify: `rank_model/README.md`
- Delete: `rank_model/tests/test_mlp_top100_hybrid.py`
- Delete if empty: `rank_model/tests/__init__.py`, `rank_model/tests/`

**Interfaces:**
- Produces no new runtime interface; proves compatibility and leaves no test artifact.

- [ ] **Step 1: Run all temporary tests fresh**

```powershell
& $python -m unittest discover -s rank_model/tests -v
```

Expected: zero failures and zero errors.

- [ ] **Step 2: Run compile, doctor, and CLI checks**

```powershell
& $python -m compileall -q rank_model
& $python -m rank_model.pipeline --config rank_model\config.toml doctor
& $python -m rank_model.pipeline --help
& $python -m rank_model.pipeline train --help
```

Expected: all commands exit zero; doctor validates the existing dataset read-only; train help lists `mlp_top100_hybrid_rank` and all seven prior names.

- [ ] **Step 3: Run a deterministic reduced-epoch smoke run in a temporary directory**

Use synthetic data and the registered trainer twice with identical parameters. Assert identical predictions, finite metrics, `optimizer_step_count > stopped_epoch`, valid best/stopped epochs, and successful save/reload.

- [ ] **Step 4: Inspect repository scope**

```powershell
git diff --check
git status --short
git diff --stat HEAD~3
```

Confirm no existing run directory or model result changed.

- [ ] **Step 5: Delete temporary tests and caches, then re-run runtime checks**

Delete only files created under `rank_model/tests` for this implementation and generated `__pycache__` directories. Re-run compile, doctor, CLI help, and a direct import of the new registry entry.

- [ ] **Step 6: Commit cleanup**

```powershell
git add -u rank_model/tests
git commit -m "test: remove temporary hybrid MLP tests"
```

- [ ] **Step 7: Report the manual real-training commands**

Provide commands for `train`, `evaluate`, and `compare`. Do not claim performance improvement until the user's real 2019-2023 run has completed and the new result is compared with the seven immutable baselines.
