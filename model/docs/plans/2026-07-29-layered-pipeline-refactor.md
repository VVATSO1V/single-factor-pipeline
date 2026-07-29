# Model Layered Pipeline Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the five top-level model build scripts with one config-driven command entry and focused internal stages without changing formulas, fields, dates, sample membership, or generated values.

**Architecture:** `model.pipeline` reads one TOML configuration and calls internal `model.stages` functions. The existing market, feature, label, model-dataset, and sample-index logic is migrated before old entry scripts are removed; temporary equivalence checks compare old and new implementations. Training, evaluation, and backtest modules remain documented future stages and are not created as empty files.

**Tech Stack:** Python 3.11+, standard-library `argparse` and `tomllib`, pandas, NumPy, pyarrow, rqdatac, PowerShell.

## Global Constraints

- All changed production files stay under `model/`.
- Do not change any factor formula, target formula, feature whitelist, field name, time split, purge rule, or tradeability rule.
- `prepare-data` must never call Ricequant; only `fetch-market` may access it.
- Keep current `model/data/*` paths and file formats during this refactor.
- Do not use `entry_tradeable` to select model samples.
- Keep 2024-2025 as locked test and 2019-2023 as final retraining period.
- Temporary tests, fixture files, comparison outputs, and caches must be deleted before delivery.
- Do not edit or revert unrelated dirty working-tree files.

---

### Task 1: Add the package, central configuration, and command skeleton

**Files:**
- Create: `model/__init__.py`
- Create: `model/config.toml`
- Create: `model/pipeline.py`
- Create: `model/stages/__init__.py`
- Temporary test: `model/.refactor_checks/test_config_and_cli.py`

**Interfaces:**
- Consumes: project-root `.env`, existing factor paths, current `model/data` paths.
- Produces: `load_config(path: Path) -> dict[str, object]`, `resolve_config_path(config_path: Path, raw_path: str) -> Path`, and CLI commands `doctor`, `fetch-market`, `prepare-data`.

- [ ] **Step 1: Write the failing configuration and CLI test**

```python
from pathlib import Path
import tempfile
import unittest

from model.pipeline import load_config, make_parser


class ConfigAndCliTest(unittest.TestCase):
    def test_default_config_has_all_required_sections(self):
        config = load_config(Path("model/config.toml"))
        self.assertEqual(config["project"]["index_code"], "000852.XSHG")
        self.assertEqual(config["target"]["horizons"], [1, 5, 10])
        self.assertEqual(len(config["factors"]["paths"]), 17)

    def test_parser_exposes_only_implemented_commands(self):
        parser = make_parser()
        self.assertEqual(parser.parse_args(["doctor"]).command, "doctor")
        self.assertEqual(
            parser.parse_args(["fetch-market"]).command,
            "fetch-market",
        )
        self.assertEqual(
            parser.parse_args(["prepare-data"]).command,
            "prepare-data",
        )

    def test_relative_paths_are_resolved_from_config_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.toml"
            config_path.write_text(
                '[project]\nindex_code="000852.XSHG"\n',
                encoding="utf-8",
            )
            from model.pipeline import resolve_config_path
            self.assertEqual(
                resolve_config_path(config_path, "../factor.csv"),
                (Path(directory) / "../factor.csv").resolve(),
            )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify the new package is missing**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest model\.refactor_checks\test_config_and_cli.py -v
```

Expected: FAIL because `model.pipeline` does not exist.

- [ ] **Step 3: Add the exact configuration sections**

Create `model/config.toml` with:

```toml
[project]
name = "csi1000_multifactor"
index_code = "000852.XSHG"
start_date = "2019-01-01"
end_date = "2025-12-31"
industry_source = "citics_2019"
env_path = "../.env"

[paths]
market_panel = "data/market_panel.csv"
trading_calendar = "data/trading_calendar.csv"
factor_wide = "data/factor_wide.csv"
target = "data/target.csv"
model_dataset = "data/model_dataset.parquet"
model_schema = "data/model_dataset_schema.json"
sample_index = "data/model_sample_index.parquet"
split_summary = "data/model_split_summary.json"
runs_dir = "runs"

[factors]
paths = [
  "../rev_1d/data/factor.csv",
  "../rev_5d/data/factor.csv",
  "../rev_20d_skip1/data/factor.csv",
  "../ivol_20d/data/factor.csv",
  "../volume_price_divergence_20d/data/factor.csv",
  "../overnight_rev_5d/data/factor.csv",
  "../intraday_rev_5d/data/factor.csv",
  "../downside_ivol_20d/data/factor.csv",
  "../tail_strength_20d/data/factor.csv",
  "../eps_revision_60d/data/factor.csv",
  "../eps_revision_breadth_60d/data/factor.csv",
  "../sue_latest/data/factor.csv",
  "../cash_earnings_quality/data/factor.csv",
  "../gross_margin_stability/data/factor.csv",
  "../roe_quality/data/factor.csv",
  "../leverage_risk/data/factor.csv",
  "../reverse_20d/data/factor.csv",
]

[target]
horizons = [1, 5, 10]
type = "absolute_open_to_open_return"
entry_price = "post_open"

[split.train]
start = "2019-01-01"
end = "2022-12-31"

[split.validation]
start = "2023-01-01"
end = "2023-12-31"

[split.test]
start = "2024-01-01"
end = "2025-12-31"

[split.final_train]
start = "2019-01-01"
end = "2023-12-31"

[execution]
min_listing_days = 120
```

- [ ] **Step 4: Implement the minimal package and parser**

`model/pipeline.py` must define:

```python
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.toml"
COMMANDS = ("doctor", "fetch-market", "prepare-data")


def resolve_config_path(config_path: Path, raw_path: str) -> Path:
    return (config_path.resolve().parent / raw_path).resolve()


def load_config(path: Path) -> dict[str, object]:
    with Path(path).open("rb") as handle:
        config = tomllib.load(handle)
    validate_config(config, Path(path))
    return config


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CSI1000 model pipeline")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor")
    subparsers.add_parser("fetch-market")
    subparsers.add_parser("prepare-data")
    return parser
```

`validate_config` must reject a missing required section, a horizon other than `1`, `5`, or `10`, duplicate factor paths, an invalid date order, and a non-positive `min_listing_days`.

- [ ] **Step 5: Run the configuration and CLI test**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest model\.refactor_checks\test_config_and_cli.py -v
```

Expected: all three tests PASS.

- [ ] **Step 6: Commit only Task 1 production files**

```powershell
git add model/__init__.py model/config.toml model/pipeline.py model/stages/__init__.py
git commit -m "refactor: add model pipeline configuration"
```

Do not add `model/.refactor_checks`.

---

### Task 2: Migrate the market-panel builder

**Files:**
- Create: `model/stages/market.py`
- Modify: `model/pipeline.py`
- Temporary test: `model/.refactor_checks/test_market_equivalence.py`
- Keep until Task 6: `model/build_market_panel.py`

**Interfaces:**
- Consumes: explicit `env_path`, `start_date`, `end_date`, `index_code`, `industry_source`, and optional `sample_size`.
- Produces: `build_market_panel(...) -> tuple[pd.DataFrame, pd.DataFrame]` and `write_market_outputs(panel, calendar, panel_path, calendar_path) -> None`.

- [ ] **Step 1: Write a failing equivalence test using fake Ricequant outputs**

The temporary test must import the old and new modules, replace these functions in both modules with deterministic fakes:

```python
init_rqdatac
build_universe
build_prices
build_status
build_exposures
```

The fake data must include two trading dates, two stocks, one stock entering the universe on the second date, one missing price record, and one missing status record. Assert:

```python
pd.testing.assert_frame_equal(old_panel, new_panel)
pd.testing.assert_frame_equal(old_calendar, new_calendar)
```

- [ ] **Step 2: Run the market equivalence test**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest model\.refactor_checks\test_market_equivalence.py -v
```

Expected: FAIL because `model.stages.market` does not exist.

- [ ] **Step 3: Move the existing market logic without changing formulas**

Copy the existing helper functions into `model/stages/market.py`. Replace the CLI-bound build signature with:

```python
def build_market_panel(
    *,
    env_path: Path,
    start_date: str,
    end_date: str,
    index_code: str,
    industry_source: str,
    sample_size: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
```

Remove `parse_args`, `main`, and the `if __name__ == "__main__"` block. Keep the panel column order and calendar validation unchanged.

- [ ] **Step 4: Implement `fetch-market`**

`pipeline.py` must resolve paths from `config.toml`, call `market.build_market_panel`, and write the same UTF-8 BOM CSV outputs with `%Y-%m-%d` calendar dates. It must print resolved paths and shapes.

- [ ] **Step 5: Run the fake-source equivalence and CLI help checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest model\.refactor_checks\test_market_equivalence.py -v
.\.venv\Scripts\python.exe -m model.pipeline fetch-market --help
```

Expected: equivalence PASS; help exits with code 0 without accessing Ricequant.

- [ ] **Step 6: Commit the migrated market stage**

```powershell
git add model/stages/market.py model/pipeline.py
git commit -m "refactor: migrate model market stage"
```

---

### Task 3: Migrate factor-table and target builders

**Files:**
- Create: `model/stages/features.py`
- Create: `model/stages/labels.py`
- Modify: `model/pipeline.py`
- Temporary test: `model/.refactor_checks/test_feature_label_equivalence.py`
- Keep until Task 6: `model/build_factor_table.py`
- Keep until Task 6: `model/build_target.py`

**Interfaces:**
- Consumes: configured market panel, trading calendar, 17 factor files, and configured output paths.
- Produces: unchanged `build_factor_table(...) -> pd.DataFrame` and `build_target_table(...) -> pd.DataFrame`.

- [ ] **Step 1: Write a failing fixture equivalence test**

Create a temporary five-date, three-stock market panel with:

- `in_universe=True` for two stocks per date.
- Valid `post_open` values except one intentional missing value.
- Two temporary factor CSVs with `date,stock_code,factor_value`.
- One factor missing on one valid universe key.

Call the old and new functions with separate output paths and assert:

```python
pd.testing.assert_frame_equal(old_factor, new_factor)
pd.testing.assert_frame_equal(old_target, new_target)
assert old_factor_output.read_bytes() == new_factor_output.read_bytes()
assert old_target_output.read_bytes() == new_target_output.read_bytes()
```

- [ ] **Step 2: Run the test and verify the new stages are missing**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest model\.refactor_checks\test_feature_label_equivalence.py -v
```

Expected: FAIL on missing `model.stages.features` or `model.stages.labels`.

- [ ] **Step 3: Move factor-table logic**

Move all non-CLI functions from `build_factor_table.py` to `stages/features.py`. Remove hard-coded `FACTOR_PATHS`, default paths, `parse_args`, and `main`. Keep factor names inferred from the parent directory and preserve factor path order from `config.toml`.

- [ ] **Step 4: Move target logic**

Move all non-CLI functions from `build_target.py` to `stages/labels.py`. Keep:

```text
target_1d(T)  = post_open(T+2)  / post_open(T+1) - 1
target_5d(T)  = post_open(T+6)  / post_open(T+1) - 1
target_10d(T) = post_open(T+11) / post_open(T+1) - 1
```

Keep the official trading calendar as the only shift calendar and retain missing targets rather than skipping dates.

- [ ] **Step 5: Add local stage calls to `prepare-data`**

Resolve all configured factor and output paths once in `pipeline.py`, then call:

```python
features.build_factor_table(...)
labels.build_target_table(...)
```

The command must stop before either call if the market panel, trading calendar, or any factor file is missing.

- [ ] **Step 6: Run fixture equivalence**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest model\.refactor_checks\test_feature_label_equivalence.py -v
```

Expected: all DataFrame and byte comparisons PASS.

- [ ] **Step 7: Commit Task 3 production files**

```powershell
git add model/stages/features.py model/stages/labels.py model/pipeline.py
git commit -m "refactor: migrate model feature and label stages"
```

---

### Task 4: Consolidate model-dataset and sample-index stages

**Files:**
- Create: `model/stages/dataset.py`
- Modify: `model/pipeline.py`
- Temporary test: `model/.refactor_checks/test_dataset_equivalence.py`
- Keep until Task 6: `model/build_model_dataset.py`
- Keep until Task 6: `model/build_sample_index.py`

**Interfaces:**
- Consumes: factor wide table, target table, market panel, trading calendar, schema path, split dates, and minimum listing days.
- Produces: unchanged `build_model_dataset(...) -> pd.DataFrame` and a configurable `build_sample_index(...) -> pd.DataFrame`.

- [ ] **Step 1: Write the failing dataset equivalence test**

Build a temporary fixture that includes:

- Six dates and two stocks.
- One missing target.
- One T+1 ST observation.
- One T+1 suspension.
- One T+1 open at the limit-up price.
- One stock below 120 listing days.

Call old and new model-dataset builders and assert identical DataFrames, Parquet schemas, and JSON schemas after removing only the output-file SHA field before comparison.

Call old and new sample-index builders with the current fixed split dates and assert identical DataFrames and summaries after replacing path-dependent hashes with a fixed sentinel.

- [ ] **Step 2: Run the dataset equivalence test**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest model\.refactor_checks\test_dataset_equivalence.py -v
```

Expected: FAIL because `model.stages.dataset` does not exist.

- [ ] **Step 3: Merge non-CLI model-dataset functions**

Move the non-CLI contents of `build_model_dataset.py` into `stages/dataset.py`, preserving:

- the 17-feature whitelist derivation;
- T metadata columns;
- all `entry_*` audit columns;
- `entry_tradeable` calculation;
- Parquet metadata;
- JSON schema and SHA validation;
- atomic file writes.

- [ ] **Step 4: Merge sample-index functions and remove fixed split constants**

Move non-CLI sample-index functions into the same module. Change `build_sample_index` to receive:

```python
split_periods: dict[str, tuple[pd.Timestamp, pd.Timestamp]]
final_train_period: tuple[pd.Timestamp, pd.Timestamp]
```

Pass the exact current dates from `config.toml`. Keep exit offsets `2`, `6`, and `11` and keep target-validity and exit-boundary purge behavior unchanged.

- [ ] **Step 5: Complete `prepare-data`**

After features and labels, call:

```python
dataset.build_model_dataset(...)
dataset.build_sample_index(...)
```

Use only configured local paths and configured `min_listing_days`.

- [ ] **Step 6: Run fixture equivalence and compile checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest model\.refactor_checks\test_dataset_equivalence.py -v
.\.venv\Scripts\python.exe -m compileall -q model
```

Expected: equivalence PASS and compileall exits 0.

- [ ] **Step 7: Commit Task 4 production files**

```powershell
git add model/stages/dataset.py model/pipeline.py
git commit -m "refactor: consolidate model dataset stages"
```

---

### Task 5: Implement `doctor` and production data-contract checks

**Files:**
- Modify: `model/pipeline.py`
- Temporary test: `model/.refactor_checks/test_doctor.py`

**Interfaces:**
- Consumes: `config.toml`, Python environment, factor CSV headers, and optional existing model outputs.
- Produces: exit code 0 with a concise check summary, or a nonzero exit with exact failing paths and conditions.

- [ ] **Step 1: Write failing doctor tests**

The tests must verify:

1. A valid temporary config with valid factor headers passes.
2. A missing factor file fails and names the path.
3. A factor file missing `factor_value` fails.
4. Duplicate factor names inferred from parent directories fail.
5. Existing schema feature columns differing from configured factor names fail.

- [ ] **Step 2: Run doctor tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest model\.refactor_checks\test_doctor.py -v
```

Expected: FAIL because doctor checks are not implemented.

- [ ] **Step 3: Implement doctor without data downloads**

`doctor` must check:

- Python is at least 3.11.
- pandas, NumPy, pyarrow, and rqdatac are importable.
- every required config section and date is valid;
- every factor file exists and has exactly the required key/value columns available;
- factor names are unique;
- market panel and calendar presence is reported separately from factor readiness;
- existing schema and split summary hashes are checked when their source files exist.

It must not initialize rqdatac or write any file.

- [ ] **Step 4: Run temporary tests and the real doctor command**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest model\.refactor_checks\test_doctor.py -v
.\.venv\Scripts\python.exe -m model.pipeline doctor
```

Expected: tests PASS; real project reports the 17 configured factors and current data readiness without warnings caused by rqdatac login.

- [ ] **Step 5: Commit doctor**

```powershell
git add model/pipeline.py
git commit -m "feat: add model pipeline doctor checks"
```

---

### Task 6: Remove legacy entry scripts and update README

**Files:**
- Delete: `model/build_market_panel.py`
- Delete: `model/build_factor_table.py`
- Delete: `model/build_target.py`
- Delete: `model/build_model_dataset.py`
- Delete: `model/build_sample_index.py`
- Modify: `model/README.md`

**Interfaces:**
- Consumes: the implemented commands and configuration fields.
- Produces: newcomer instructions that describe only commands that actually exist, with future training commands clearly labeled as planned.

- [ ] **Step 1: Confirm all temporary equivalence tests pass while old modules still exist**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s model\.refactor_checks -p "test_*.py" -v
```

Expected: every old-versus-new equivalence assertion PASS.

- [ ] **Step 2: Remove the five old entry scripts**

Delete only the five listed files after Step 1 passes. Do not delete current `model/data` artifacts.

- [ ] **Step 3: Rewrite README around the new workflow**

The README must contain:

1. Pipeline purpose and current T-to-T+1-open timing.
2. Three user-facing files.
3. `config.toml` sections and factor add/remove example.
4. Exact `doctor`, `fetch-market`, and `prepare-data` PowerShell commands.
5. Explicit warning that `prepare-data` never downloads Ricequant data.
6. Current outputs and row-count expectations.
7. Future `train`, `evaluate`, and `backtest` interface design marked “尚未实现”.
8. Future-function and locked-test rules.
9. Rebuild-dependency table.

- [ ] **Step 4: Check README commands against parser help**

Run:

```powershell
.\.venv\Scripts\python.exe -m model.pipeline --help
.\.venv\Scripts\python.exe -m model.pipeline doctor --help
.\.venv\Scripts\python.exe -m model.pipeline fetch-market --help
.\.venv\Scripts\python.exe -m model.pipeline prepare-data --help
```

Expected: every implemented README command exits 0 and every future command is labeled unavailable.

- [ ] **Step 5: Commit code removal and documentation**

```powershell
git add model/README.md model/build_market_panel.py model/build_factor_table.py model/build_target.py model/build_model_dataset.py model/build_sample_index.py
git commit -m "docs: switch model workflow to unified pipeline"
```

---

### Task 7: Run final regression checks and remove all temporary files

**Files:**
- Verify: all production files under `model/`
- Delete: `model/.refactor_checks/`
- Delete: all `model/**/__pycache__/`
- Do not modify: existing `model/data/*`

**Interfaces:**
- Consumes: final layered pipeline and current real data artifacts.
- Produces: a clean working tree scope with no temporary test files and a written verification summary.

- [ ] **Step 1: Capture current real-data contracts**

Using read-only pandas/pyarrow checks, verify:

```text
factor_wide.csv: 1,699,000 rows, 19 columns
target.csv: 1,699,000 rows, 5 columns
model_dataset.parquet: 1,699,000 rows, 35 columns
model_sample_index.parquet: 1,699,000 rows, 12 columns
```

Also verify:

- `(date, stock_code)` uniqueness in all four tables;
- exactly 1,000 rows on every factor-wide date;
- 17 schema feature columns matching config order;
- target counts and split counts matching `model_split_summary.json`;
- `uses_entry_tradeable` remains false.

- [ ] **Step 2: Run all temporary checks one final time**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s model\.refactor_checks -p "test_*.py" -v
.\.venv\Scripts\python.exe -m model.pipeline doctor
.\.venv\Scripts\python.exe -m compileall -q model
```

Expected: all commands exit 0.

- [ ] **Step 3: Delete temporary tests and caches**

Delete:

```text
model/.refactor_checks/
model/__pycache__/
model/stages/__pycache__/
```

Search the complete `model` tree and remove any temporary fixture, `.tmp`, comparison output, or cache created by this refactor.

- [ ] **Step 4: Verify final file layout and no legacy imports**

Run:

```powershell
rg -n "build_market_panel|build_factor_table|build_target|build_model_dataset|build_sample_index" model --glob "*.py" --glob "*.md"
```

Expected: references appear only as internal function names or migration history in design documents, never as imports of deleted top-level modules or executable README commands.

- [ ] **Step 5: Review the final diff**

Confirm:

- no files outside `model/` changed by this refactor;
- no generated data files are staged;
- no factor, target, split, or tradeability formula changed;
- no empty future-stage Python files were created;
- README describes the implemented state exactly.

- [ ] **Step 6: Commit final cleanup if needed**

```powershell
git add model
git commit -m "chore: verify layered model pipeline"
```

Skip the commit when no production cleanup remains.
