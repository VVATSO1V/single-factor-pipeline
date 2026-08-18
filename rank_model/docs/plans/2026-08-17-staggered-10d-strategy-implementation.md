# Staggered 10-Trading-Day Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ten independent, full-capital 10-trading-day rebalance paths for each frozen rank model, publish path and timing-average reports, and compare them with the existing daily strategy without running the final five-model backtests.

**Architecture:** Keep the existing execution engine authoritative. Add one buy-suppression switch to its open transition, then implement scheduling, offset simulation, aggregation, and immutable publication in a focused `staggered_strategy.py` stage. The unified pipeline receives two new commands and three new contained output paths; existing daily strategy artifacts and formulas remain unchanged.

**Tech Stack:** Python 3.11+, pandas, NumPy, PyArrow, TOML, `unittest`, SHA-256 artifact manifests, atomic directory/file publication.

## Global Constraints

- Work only in `rank_model`; do not modify factor directories, `model` code, or existing real artifacts.
- Use exactly the five names in `rank_model.stages.strategy.LOCKED_MODEL_NAMES`.
- Use only `score_raw` from 2024-01-02 through 2025-12-16 for selection; never use target columns or test metrics.
- Use T close for signal, T+1 adjusted open for entry/transition, and T+11 adjusted open for a complete 10-trading-day horizon.
- Partition all 474 eligible signal dates exactly once across offsets 1 through 10.
- The common ten-path average NAV interval must end on 2025-12-18.
- Retry blocked sells daily; never retry a failed buy before the next scheduled rebalance.
- Preserve current buy/sell restrictions, set-difference units, sizing, costs, marks, cash scaling, and no-replacement rules.
- Do not force terminal liquidation.
- Preserve existing `backtest-strategy`, `compare-strategy`, `strategy_runs`, and comparison artifacts byte-for-byte.
- Do not run any real `backtest-strategy-10d` command for the five frozen models.
- Temporary tests and caches must be deleted before delivery.

---

## File Map

- Modify `rank_model/stages/strategy.py`: add backward-compatible buy suppression and a shared sealed-input loader.
- Create `rank_model/stages/staggered_strategy.py`: schedule, offset simulation, path metrics, average NAV, publication, validation, and comparisons.
- Modify `rank_model/pipeline.py`: contained path validation, command adapters, parser registration, and error handling.
- Modify `rank_model/config.toml`: fixed staggered output paths and sealed scheduling constants.
- Modify `rank_model/README.md`: exact manual commands, semantics, and artifact layout.
- Create then delete `rank_model/tests/test_staggered_strategy.py`: temporary synthetic contract tests.

---

### Task 1: Shared Transition And Sealed Input Boundary

**Files:**
- Modify: `rank_model/stages/strategy.py`
- Create temporarily: `rank_model/tests/__init__.py`
- Create temporarily: `rank_model/tests/test_staggered_strategy.py`

**Interfaces:**
- Produces: `transition_at_open(state, desired, market, signal_date, execution_date, settings, *, allow_buys=True) -> TransitionResult`
- Produces: `LockedStrategyInputs`
- Produces: `load_locked_strategy_inputs(model_name, settings, source_paths) -> LockedStrategyInputs`
- Preserves: existing daily `simulate_strategy` behavior through the default `allow_buys=True`.

- [ ] **Step 1: Write failing transition tests**

Add synthetic tests proving that `allow_buys=False` still marks holdings and
retries holdings outside `desired`, but creates no buy attempts for missing
desired names:

```python
result = transition_at_open(
    state,
    ("KEEP", "MISSING"),
    market_for_day,
    pd.Timestamp("2024-01-02"),
    pd.Timestamp("2024-01-03"),
    settings,
    allow_buys=False,
)
self.assertFalse(result.trades["side"].eq("buy").any())
self.assertEqual(result.state.positions["KEEP"].units, keep_units)
self.assertTrue(result.trades.query("stock_code == 'EXIT'")["side"].eq("sell").all())
```

Also snapshot a normal `allow_buys=True` transition and assert its trade rows,
cash, units, costs, and end NAV remain unchanged.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
& $python -B -m unittest rank_model.tests.test_staggered_strategy.TransitionModeTests -v
```

Expected: failure because `allow_buys` is not accepted.

- [ ] **Step 3: Implement buy suppression minimally**

Change only the candidate line in `transition_at_open`:

```python
def transition_at_open(
    state: PortfolioState,
    desired: tuple[str, ...],
    market: pd.DataFrame,
    signal_date: Any,
    execution_date: Any,
    settings: StrategySettings,
    *,
    allow_buys: bool = True,
) -> TransitionResult:
    if not isinstance(allow_buys, bool):
        raise ValueError("allow_buys must be boolean")
    result_state = _validated_state(state)
    desired_codes = _desired_stock_codes(desired)
    market_rows = _market_rows_by_stock(market)
    # Keep the existing mark and sell blocks unchanged.
    entry_candidates = (
        sorted(desired_set - set(result_state.positions)) if allow_buys else []
    )
```

Do not branch sell logic or marking logic.

- [ ] **Step 4: Extract the sealed input loader with behavior snapshots**

Move the read-and-validation portion of `backtest_locked_strategy` into:

```python
@dataclass(frozen=True)
class LockedStrategyInputs:
    predictions: pd.DataFrame
    market_panel: pd.DataFrame
    trading_calendar: pd.DataFrame
    source_paths: dict[str, Path]
    source_hashes: dict[str, str]


def load_locked_strategy_inputs(
    model_name: str,
    settings: StrategySettings,
    source_paths: Mapping[str, Path],
) -> LockedStrategyInputs:
    """Validate seals once and load prediction, calendar, and market frames."""
```

The loader must perform the current source hashes, frozen-spec, five-row
locked comparison, prediction, calendar, and market-panel checks. Refactor
`backtest_locked_strategy` to call it and pass its frames to `simulate_strategy`.
Tests compare loader failures for a bad prediction hash, target-independent
selection, and an invalid frozen model name with the existing error contract.

- [ ] **Step 5: Run tests and daily-regression checks**

Run the focused tests plus a synthetic daily `simulate_strategy` snapshot.
Assert exact equality for daily NAV, trades, positions, diagnostics, summary,
and source hashes before and after the refactor.

- [ ] **Step 6: Commit Task 1**

```powershell
git add rank_model/stages/strategy.py rank_model/tests
git commit -m "refactor(rank-model): share strategy execution inputs"
```

---

### Task 2: Offset Schedule And Daily Path Simulation

**Files:**
- Create: `rank_model/stages/staggered_strategy.py`
- Modify temporarily: `rank_model/tests/test_staggered_strategy.py`

**Interfaces:**
- Consumes: `LockedStrategyInputs`, `StrategySettings`, `transition_at_open`.
- Produces: `OffsetSchedule` and `build_offset_schedules`.
- Produces: `StaggeredPathBundle` and `simulate_staggered_offset`.

- [ ] **Step 1: Write failing schedule tests**

Define expected schedule assertions against a 485-date synthetic official
calendar:

```python
schedules = build_offset_schedules(calendar, settings)
self.assertEqual(len(schedules), 10)
self.assertEqual(sum(len(x.signal_dates) for x in schedules), 474)
self.assertEqual({d for x in schedules for d in x.signal_dates}, set(period[:-11]))
self.assertEqual(schedules[0].first_execution_date, period[1])
self.assertEqual(schedules[9].first_execution_date, period[10])
self.assertEqual(min(x.final_horizon_date for x in schedules), period[475])
```

Add rejection tests for duplicate calendars, fewer than twelve period dates,
wrong offset count, and any schedule overlap/gap injected into a validator.

- [ ] **Step 2: Run schedule tests and verify RED**

Expected: import failure because `staggered_strategy.py` does not exist.

- [ ] **Step 3: Implement immutable schedule types**

Create:

```python
STAGGERED_HORIZON = 10
STAGGERED_OFFSET_COUNT = 10
STAGGERED_COMPLETE_SHIFT = 11

@dataclass(frozen=True)
class OffsetSchedule:
    offset: int
    signal_dates: Sequence[pd.Timestamp]
    execution_dates: Sequence[pd.Timestamp]
    final_horizon_date: pd.Timestamp

def build_offset_schedules(
    calendar: Any,
    settings: StrategySettings,
) -> Sequence[OffsetSchedule]:
    official = _calendar_dates(calendar)
    period = official[(official >= settings.start) & (official <= settings.end)]
    eligible = period[:-STAGGERED_COMPLETE_SHIFT]
    schedules = []
    for offset in range(1, STAGGERED_OFFSET_COUNT + 1):
        indices = list(range(offset - 1, len(eligible), STAGGERED_OFFSET_COUNT))
        schedules.append(
            OffsetSchedule(
                offset=offset,
                signal_dates=tuple(eligible[index] for index in indices),
                execution_dates=tuple(period[index + 1] for index in indices),
                final_horizon_date=period[indices[-1] + STAGGERED_COMPLETE_SHIFT],
            )
        )
    return tuple(schedules)
```

Derive eligible signals from the official calendar, never from target values.
Validate 474 dates, exact modulo partition, T+1 execution, T+11 completion,
and the sealed first/last dates.

- [ ] **Step 4: Write failing path-state tests**

Use a 32-date, four-stock synthetic panel and `top_k=2` test settings. Cover:

- offset cash rows before first execution;
- Top100 refresh only on scheduled dates;
- daily marks on non-scheduled dates;
- blocked sell retry on the next day;
- failed buy not retried before the next schedule;
- pending sell cancelled when the next Top100 contains that stock;
- unchanged units for desired intersections;
- final row exactly on the schedule's `final_horizon_date`;
- no forced terminal sell.

- [ ] **Step 5: Implement the path simulator**

Create:

```python
@dataclass
class StaggeredPathBundle:
    offset: int
    schedule: OffsetSchedule
    strategy: StrategyBundle

def simulate_staggered_offset(
    predictions: pd.DataFrame,
    market: pd.DataFrame,
    calendar: Any,
    settings: StrategySettings,
    schedule: OffsetSchedule,
) -> StaggeredPathBundle:
    """Value one offset daily and update its desired set only on schedule."""
```

The loop must value every official date from 2024-01-02 through the path's
final horizon. On scheduled execution dates, update `current_desired` from the
prior scheduled signal and call `transition_at_open` with `allow_buys=True`.
On other dates, keep `current_desired` and call with `allow_buys=False` so only
exits outside that set can retry. Record `offset`, `is_rebalance`, and
`active_signal_date` in the new daily and diagnostic reports. Use
`summarize_strategy` for each path's fixed metric formulas.

- [ ] **Step 6: Run Task 2 tests and commit**

```powershell
& $python -B -m unittest rank_model.tests.test_staggered_strategy.ScheduleTests -v
& $python -B -m unittest rank_model.tests.test_staggered_strategy.PathSimulationTests -v
git add rank_model/stages/staggered_strategy.py rank_model/tests
git commit -m "feat(rank-model): simulate staggered 10d paths"
```

---

### Task 3: Timing Aggregation And Immutable Model Publication

**Files:**
- Modify: `rank_model/stages/staggered_strategy.py`
- Modify temporarily: `rank_model/tests/test_staggered_strategy.py`

**Interfaces:**
- Consumes: ten `StaggeredPathBundle` instances.
- Produces: `build_offset_metrics`, `build_offset_summary`,
  `build_average_nav`, and `summarize_average_nav`.
- Produces: `backtest_staggered_strategy(model_name, *, destination, publication_root, settings, source_paths) -> Path`.

- [ ] **Step 1: Write failing aggregation tests**

Construct ten deterministic path NAV frames with different final dates. Assert:

```python
average = build_average_nav(paths, settings)
self.assertEqual(average.index.min(), pd.Timestamp("2024-01-02"))
self.assertEqual(average.index.max(), min(p.schedule.final_horizon_date for p in paths))
self.assertEqual(average.loc[date, "nav"], np.mean([p_nav(date) for p in paths]))
self.assertEqual(average.loc[date, "total_cost"], np.mean([p_cost(date) for p in paths]))
```

Verify aggregate return, volatility, Sharpe, drawdown, turnover, costs, and
cash ratio are recomputed from aggregate economic columns. Explicitly prove
that aggregate Sharpe is not the mean of path Sharpes. Assert the offset detail
has ten rows and summary has mean, median, sample std, min, and max for every
declared comparable metric.

- [ ] **Step 2: Implement average and distribution reports**

Define exact schemas:

```python
OFFSET_STATISTICS = ("mean", "median", "std", "min", "max")
AVERAGE_METRICS = (
    "elapsed_trading_observations", "cumulative_return", "cagr",
    "annualized_return", "annualized_volatility", "sharpe_ratio",
    "max_drawdown", "win_rate", "average_cash_ratio",
    "gross_turnover", "one_way_turnover", "average_gross_turnover",
    "average_one_way_turnover", "average_turnover",
    "annualized_gross_turnover", "annualized_one_way_turnover",
    "annualized_turnover", "total_cost", "ending_nav",
)
```

Build the average only on the exact date intersection of all paths. Require its
end to equal the earliest final horizon and prohibit forward filling.

- [ ] **Step 3: Write failing publication and tamper tests**

Cover a full synthetic model publication with ten offset directories and five
model-level outputs. Verify:

- sibling staging and one atomic model-directory rename;
- an existing destination fails without overwrite;
- all seven offset files and five model-level files are hashed;
- the model manifest binds ten offset manifests, source hashes, schedule dates,
  formulas, row counts, and output hashes;
- path traversal and symlink escape are rejected;
- a changed path file, model summary, source manifest, or calendar fails
  validation;
- interrupted staging leaves no published partial directory.

- [ ] **Step 4: Implement atomic model publication**

Create:

```python
def backtest_staggered_strategy(
    model_name: str,
    *,
    destination: Path,
    publication_root: Path,
    settings: StrategySettings,
    source_paths: Mapping[str, Path],
) -> Path:
    """Publish all ten offset paths and aggregate reports atomically."""
```

Require `destination` to be exactly `<publication_root>/<locked model>`. Load
inputs once with `load_locked_strategy_inputs`, simulate ten paths, build
aggregates, write everything under one sibling staging directory, validate all
hashes, and atomically rename to `strategy_10d_runs/<model>`. Use a distinct
`.strategy-10d.lock`. Do not call a real five-model input in tests.

- [ ] **Step 5: Run Task 3 tests and commit**

```powershell
& $python -B -m unittest rank_model.tests.test_staggered_strategy.AggregationTests -v
& $python -B -m unittest rank_model.tests.test_staggered_strategy.PublicationTests -v
git add rank_model/stages/staggered_strategy.py rank_model/tests
git commit -m "feat(rank-model): publish staggered strategy reports"
```

---

### Task 4: Config, CLI, Five-Model Comparison, And Daily Comparison

**Files:**
- Modify: `rank_model/config.toml`
- Modify: `rank_model/pipeline.py`
- Modify: `rank_model/stages/staggered_strategy.py`
- Modify temporarily: `rank_model/tests/test_staggered_strategy.py`

**Interfaces:**
- Produces CLI: `backtest-strategy-10d --model <LOCKED_MODEL_NAMES>`.
- Produces CLI: `compare-strategy-10d`.
- Produces immutable `strategy_10d_comparison.csv` and manifest.
- Produces immutable `daily_vs_10d_comparison.csv` and manifest.

- [ ] **Step 1: Add exact config contracts and failing validation tests**

Add paths:

```toml
strategy_10d_runs_dir = "strategy_10d_runs"
strategy_10d_comparison = "strategy_10d_comparison.csv"
daily_vs_10d_comparison = "daily_vs_10d_comparison.csv"
```

Add fixed settings:

```toml
[strategy_10d]
horizon = 10
offset_count = 10
last_complete_signal = "2025-12-16"
retry_blocked_sells_daily = true
retry_failed_buys_daily = false
```

Tests reject every changed value, missing key, output outside `rank_model`,
output inside `model/data`, or output path that aliases a daily artifact.

- [ ] **Step 2: Implement path validation and command adapters**

Add `_validate_staggered_strategy_paths`,
`command_backtest_strategy_10d`, and `command_compare_strategy_10d` in
`pipeline.py`. Build the same exact eight-source mapping used by the daily
command. The comparison command requires all five completed 10-day model
manifests plus the existing valid daily strategy comparison and sidecar.

- [ ] **Step 3: Implement immutable comparisons**

`strategy_10d_comparison.csv` has one row per model and includes every average
line metric plus `offset_<stat>_<metric>` fields for the five distribution
statistics. `daily_vs_10d_comparison.csv` has one row per model and paired
`daily_<metric>`, `strategy_10d_<metric>`, and `difference_<metric>` columns for:

```python
DAILY_VS_10D_METRICS = (
    "cumulative_return", "cagr", "annualized_volatility", "sharpe_ratio",
    "max_drawdown", "average_cash_ratio", "annualized_turnover", "total_cost",
)
```

Publish each CSV with an adjacent manifest binding logical/physical hashes,
column schemas, all five source model manifests, the daily comparison manifest,
and sealed formulas. A valid existing pair is verified without rewrite;
missing partners, tampering, or changed provenance fail closed.

- [ ] **Step 4: Register CLI and write parser tests**

Add parser choices exactly as designed. Route `FileNotFoundError`,
`ValueError`, and `FileExistsError` through `parser.error`. Tests assert:

```text
backtest-strategy-10d --help  -> success
compare-strategy-10d --help   -> success
unknown model                 -> argparse failure
```

Run comparison only against disposable synthetic report directories.

- [ ] **Step 5: Run Task 4 tests and commit**

```powershell
& $python -B -m unittest rank_model.tests.test_staggered_strategy.ConfigAndCliTests -v
& $python -B -m unittest rank_model.tests.test_staggered_strategy.ComparisonTests -v
git add rank_model/config.toml rank_model/pipeline.py rank_model/stages/staggered_strategy.py rank_model/tests
git commit -m "feat(rank-model): expose staggered strategy pipeline"
```

---

### Task 5: Documentation, Verification, And Temporary-Test Cleanup

**Files:**
- Modify: `rank_model/README.md`
- Delete: `rank_model/tests/test_staggered_strategy.py`
- Delete: `rank_model/tests/__init__.py` if the package is otherwise empty.

**Interfaces:**
- Delivers manual PowerShell commands only; does not produce real 10-day runs.

- [ ] **Step 1: Document exact manual workflow**

Add the five-model loop and comparison command:

```powershell
$python = ".\.venv\Scripts\python.exe"
$models = @(
  "ridge_rank_regression",
  "xgboost_rank_regression",
  "lightgbm_rank_regression",
  "lightgbm_lambdarank",
  "mlp_top100_hybrid_rank"
)
foreach ($model in $models) {
  & $python -m rank_model.pipeline --config rank_model\config.toml `
    backtest-strategy-10d --model $model
}
& $python -m rank_model.pipeline --config rank_model\config.toml `
  compare-strategy-10d
```

Document the 474 signals, ten offsets, failure-order policy, path end dates,
common average interval, 50 path reports, comparison semantics, immutability,
and the rule that 2024-2025 results cannot trigger retuning.

- [ ] **Step 2: Run the full temporary suite without real backtests**

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
& $python -B -m unittest discover -s rank_model\tests -p "test_*.py" -v
& $python -B -m rank_model.pipeline --config rank_model\config.toml doctor
& $python -B -m rank_model.pipeline --config rank_model\config.toml `
  backtest-strategy-10d --help
& $python -B -m rank_model.pipeline --config rank_model\config.toml `
  compare-strategy-10d --help
```

Also run a read-only schedule audit against the real trading calendar and
prediction metadata. It may read dates and hashes but must not call
`backtest_staggered_strategy` or create `strategy_10d_runs`.

- [ ] **Step 3: Verify unchanged existing artifacts**

Hash the existing daily `strategy_runs`, `strategy_comparison.csv`, and
`strategy_comparison.manifest.json` before and after all verification. Require
exact path counts, sizes, and SHA-256 equality. Run `git diff --check` and scan
for forbidden target access in `staggered_strategy.py`.

- [ ] **Step 4: Delete temporary tests and caches**

Remove `rank_model/tests/test_staggered_strategy.py`, the marker-only
`rank_model/tests/__init__.py`, empty test directories, `__pycache__`, `.pyc`,
and `.pyo`. Confirm that no `strategy_10d_runs`, 10-day comparison, or real
model output was created.

- [ ] **Step 5: Re-run final non-writing checks**

Run `compileall` only in a disposable temporary destination or use direct
module imports with `-B`; then delete any cache. Re-run CLI help, `doctor`,
`git diff --check`, manifest source validation, placeholder scans, and a clean
tracked status check. Do not execute the final commands from README.

- [ ] **Step 6: Commit documentation and cleanup**

```powershell
git add rank_model/README.md rank_model/tests
git commit -m "docs(rank-model): document staggered strategy workflow"
```

- [ ] **Step 7: Independent final review**

Review the complete range from the design commit through HEAD for future
functions, schedule off-by-one errors, stale-buy retries, sell retry behavior,
average-line math, path containment, immutability, and any change to the daily
strategy. Resolve Critical and Important findings, repeat focused review, and
provide the user with the exact PowerShell commands without running them.
