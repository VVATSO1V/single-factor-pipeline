# Rank Model Locked Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build static 2024-2025 predictions from the five frozen 2019-2023 rank models and produce one unified locked-test evaluation comparison.

**Architecture:** Add one focused locked-test stage inside `rank_model`, keep final model artifacts read-only, and reuse the established evaluation functions. The stage publishes a hash-sealed test dataset, immutable per-model prediction/evaluation directories, and a metric-only comparison table.

**Tech Stack:** Python, pandas, NumPy, pyarrow, joblib, scikit-learn, XGBoost, LightGBM, PyTorch, TOML.

## Global Constraints

- Modify only `rank_model` files.
- Do not train, refit, tune, or early-stop on 2024-2025.
- Do not filter model evaluation with `entry_tradeable` or any T+1 status.
- Preserve exactly 1000 `(date, stock_code)` keys per source date.
- Keep prediction and comparison outputs immutable.
- Delete temporary tests, test data, and caches before delivery.

---

### Task 1: Locked-Test Dataset Contract

**Files:**
- Create: `rank_model/stages/locked_test.py`
- Modify: `rank_model/config.toml`
- Test temporarily: `rank_model/tests/test_locked_test_dataset.py`

**Interfaces:**
- Produces: `prepare_locked_test_dataset(config, config_path) -> dict[str, object]`
- Produces: a test rank Parquet, schema JSON, and coverage CSV.

- [ ] Write synthetic failing tests for exact 2024-2025 selection, retained missing late labels, 1000-key validation, rank direction, forbidden-feature exclusion, and immutable publication.
- [ ] Run the focused tests and verify they fail because the locked-test stage does not exist.
- [ ] Implement local full-history point-in-time context construction and test-only publication with source/development hashes.
- [ ] Run focused tests and all temporary tests.

### Task 2: Frozen Static Inference

**Files:**
- Modify: `rank_model/stages/locked_test.py`
- Modify: `rank_model/pipeline.py`
- Test temporarily: `rank_model/tests/test_locked_test_inference.py`

**Interfaces:**
- Produces: `predict_locked_test_model(config, config_path, model_name) -> Path`
- Consumes: `final_runs/<model_name>` and the prepared locked-test dataset.

- [ ] Write failing tests that load Ridge and one neural artifact, preserve all prediction keys, reject feature-order drift, reject altered hashes, and reject existing output directories.
- [ ] Run the focused tests and verify the missing inference interface fails.
- [ ] Implement verified loading for joblib, XGBoost, LightGBM, and PyTorch artifacts, using each stored preprocessor without fitting.
- [ ] Add `prepare-test` and `predict-test --model` CLI commands and run the tests.

### Task 3: Locked Evaluation And Comparison

**Files:**
- Modify: `rank_model/stages/locked_test.py`
- Modify: `rank_model/pipeline.py`
- Test temporarily: `rank_model/tests/test_locked_test_evaluation.py`

**Interfaces:**
- Produces: `evaluate_locked_test_model(config, config_path, model_name) -> Path`
- Produces: `compare_locked_test_models(config, config_path) -> pd.DataFrame`
- Reuses: `evaluate_predictions`, `write_evaluation`, and `compare_runs`.

- [ ] Write failing tests for test-only split validation, combined and yearly outputs, hash checks, no automatic-decision fields, and all-five-model completeness.
- [ ] Run focused tests and verify the new interfaces are missing.
- [ ] Implement immutable evaluation and comparison publication without changing final models or parameters.
- [ ] Add `evaluate-test --model` and `compare-test` CLI commands and run all temporary tests.

### Task 4: Real-Data Acceptance And Documentation

**Files:**
- Modify: `rank_model/README.md`
- Delete before delivery: `rank_model/tests/`

**Interfaces:**
- Documents the four public locked-test commands and artifact contracts.

- [ ] Run syntax compilation and all temporary tests.
- [ ] Prepare the real 2024-2025 dataset and verify dates, rows, 1000 keys per date, labels, feature order, and hashes.
- [ ] Generate and evaluate all five static model predictions, then write the comparison table.
- [ ] Verify final model hashes are unchanged before and after the complete locked-test workflow.
- [ ] Review every test summary and yearly table for finite/expected diagnostics, without changing a model based on the result.
- [ ] Delete temporary tests and caches, rerun production doctor/CLI smoke checks, and inspect `git diff`.
