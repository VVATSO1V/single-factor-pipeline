# Static Strategy Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one reproducible strategy engine that runs identical 2024-2025 T+1 open, strict set-difference Top100 backtests for all five frozen rank models.

**Architecture:** Extend the existing `rank_model.pipeline` entry point and add one internal `rank_model.stages.strategy` module. The module validates sealed predictions and market data, simulates deterministic daily portfolio transitions, and atomically publishes immutable per-model reports plus one comparison table. It never reads test labels or metrics to form orders and never writes model artifacts.

**Tech Stack:** Python 3.11, pandas, NumPy, PyArrow, TOML configuration, unittest, existing rank-model hashing and process-lock helpers.

## Global Constraints

- Run all five names in `LOCKED_MODEL_NAMES` under one implementation and one configuration.
- Use only the T signal's `score_raw`; never use `target_10d`, `rank_target_10d`, or test metrics in trading logic.
- Select Top100 by `score_raw` descending and `stock_code` ascending.
- Execute only at the next official trading day's open and only inside 2024-2025.
- Use strict set-difference rebalancing; intersection units never change.
- Failed buys are not replaced; blocked sells remain held and are retried.
- Buy rules: non-ST, non-suspended, listing age at least 120, valid prices/status, and open at neither limit.
- Sell rules: block suspension, missing execution data, and open limit-down; ST, listing age, and limit-up do not block.
- Value with `post_open`, test limits with unadjusted open/limit fields, and carry the last valid mark during suspension.
- Initial NAV is 1.0, fractional units are allowed, and there is no lot or minimum-commission rule.
- Costs are 1 bp commission each side, 5 bps slippage each side, and 5 bps sell-only stamp duty.
- Do not liquidate at 2025 year-end; reserve the 2025-12-31 signal for 2026 forward simulation.
- Production outputs are immutable and must not update configuration, frozen models, final runs, predictions, or locked-test reports.
- Temporary tests and caches must be removed after real-data acceptance.

---

### Task 1: Seal Strategy Configuration And Test Conclusion

**Files:**
- Modify: `rank_model/config.toml`
- Create: `rank_model/locked_test_conclusion.json`
- Create temporarily: `rank_model/tests/test_strategy.py`
- Modify: `rank_model/stages/strategy.py`

**Interfaces:**
- Consumes: the existing frozen specification, locked-test comparison, locked-test schema, five prediction files, and five prediction manifests.
- Produces: `StrategySettings`, `load_strategy_settings(config)`, and `validate_locked_test_conclusion(...)`.

- [ ] **Step 1: Write failing configuration and conclusion tests**

```python
class StrategyContractTests(unittest.TestCase):
    def test_fixed_settings_are_loaded_exactly(self):
        settings = load_strategy_settings(VALID_CONFIG)
        self.assertEqual(settings.top_k, 100)
        self.assertEqual(settings.min_listing_days, 120)
        self.assertEqual(settings.buy_cost_rate, 0.0006)
        self.assertEqual(settings.sell_cost_rate, 0.0011)

    def test_conclusion_rejects_changed_prediction_hash(self):
        with self.assertRaisesRegex(ValueError, "prediction hash"):
            validate_locked_test_conclusion(
                conclusion=CONCLUSION,
                frozen_spec_path=frozen_spec,
                comparison_path=comparison,
                prediction_paths={"ridge_rank_regression": changed_prediction},
                locked_test_schema_path=locked_test_schema,
                prediction_manifest_paths=prediction_manifests,
            )
```

- [ ] **Step 2: Verify the tests fail before implementation**

Run: `& $python -m unittest rank_model.tests.test_strategy.StrategyContractTests -v`

Expected: import failure because `rank_model.stages.strategy` does not exist.

- [ ] **Step 3: Add fixed config paths and settings**

Add paths for `locked_test_conclusion`, `strategy_runs_dir`, and
`strategy_comparison`, plus this exact TOML section:

```toml
[strategy]
start = "2024-01-01"
end = "2025-12-31"
top_k = 100
expected_cross_section_size = 1000
min_listing_days = 120
initial_nav = 1.0
commission_bps = 1.0
slippage_bps = 5.0
sell_stamp_duty_bps = 5.0
limit_tolerance = 1e-8
annualization_days = 252
```

- [ ] **Step 4: Implement the immutable settings object and conclusion checks**

```python
@dataclass(frozen=True)
class StrategySettings:
    start: pd.Timestamp
    end: pd.Timestamp
    top_k: int
    expected_cross_section_size: int
    min_listing_days: int
    initial_nav: float
    commission_rate: float
    slippage_rate: float
    sell_stamp_duty_rate: float
    limit_tolerance: float
    annualization_days: int

    @property
    def buy_cost_rate(self) -> float:
        return self.commission_rate + self.slippage_rate

    @property
    def sell_cost_rate(self) -> float:
        return self.commission_rate + self.slippage_rate + self.sell_stamp_duty_rate
```

`validate_locked_test_conclusion` must require exactly the five frozen models,
the exact 2024-2025 boundary, `selection_policy = "no_test_based_selection"`,
and exact SHA-256 matches for `frozen_models.json`, the locked-test comparison,
the locked-test schema, every prediction file, and every locked-test prediction
manifest. `artifact_sha256.prediction_manifests` and the supplied manifest-path
mapping must each contain exactly all five frozen model names.

- [ ] **Step 5: Populate the tracked conclusion snapshot**

Record the existing five test metric rows and artifact hashes without adding a
winner or recommendation field. The artifact hashes must include
`locked_test_schema`, all five `predictions`, and all five
`prediction_manifests`. Include:

```json
{
  "schema_version": 1,
  "period": {"start": "2024-01-01", "end": "2025-12-31"},
  "selection_policy": "no_test_based_selection",
  "retuning_allowed": false,
  "strategy_models": ["ridge_rank_regression", "xgboost_rank_regression", "lightgbm_rank_regression", "lightgbm_lambdarank", "mlp_top100_hybrid_rank"]
}
```

- [ ] **Step 6: Run tests and commit**

Run: `& $python -m unittest rank_model.tests.test_strategy.StrategyContractTests -v`

Expected: all contract tests pass.

Commit: `feat(rank-model): seal static strategy contract`

---

### Task 2: Build Deterministic Signals And Execution Decisions

**Files:**
- Modify: `rank_model/stages/strategy.py`
- Modify temporarily: `rank_model/tests/test_strategy.py`

**Interfaces:**
- Consumes: normalized prediction and market-panel frames plus `StrategySettings`.
- Produces: `select_daily_top(predictions, settings, calendar)`, `buy_decision(row, settings)`, and `sell_decision(row, settings)`.

- [ ] **Step 1: Add failing selection and rule tests**

```python
def test_top100_uses_score_then_stock_code(self):
    selected = select_daily_top(tied_predictions, SETTINGS, calendar)
    self.assertEqual(selected[first_date][:3], ("A", "B", "C"))

def test_buy_and_sell_rules_are_asymmetric(self):
    st_row = market_row(is_st=True)
    self.assertEqual(buy_decision(st_row, SETTINGS), (False, "st"))
    self.assertEqual(sell_decision(st_row, SETTINGS), (True, "eligible"))

def test_limit_down_blocks_both_sides(self):
    row = market_row(raw_open=9.0, limit_down=9.0)
    self.assertEqual(buy_decision(row, SETTINGS)[1], "limit_down")
    self.assertEqual(sell_decision(row, SETTINGS)[1], "limit_down")
```

- [ ] **Step 2: Run the focused tests and confirm failure**

Run: `& $python -m unittest rank_model.tests.test_strategy.SignalAndRuleTests -v`

Expected: failures for missing functions.

- [ ] **Step 3: Implement prediction and calendar validation**

Require unique `(date, stock_code)` keys, finite `score_raw`, exactly 1,000 rows
per date, all signal dates in the official calendar, and exact model/test split
metadata. Sort with stable mergesort:

```python
ordered = group.sort_values(
    ["score_raw", "stock_code"],
    ascending=[False, True],
    kind="mergesort",
)
selected = tuple(ordered.head(settings.top_k)["stock_code"])
```

- [ ] **Step 4: Implement conservative execution decisions**

Both decision functions return `(eligible: bool, reason: str)`. Missing
`has_price_record`, suspension status, `raw_open`, `post_open`, or required
limit data blocks execution with one explicit reason. Buy additionally checks
ST, listing age, limit-up, and limit-down. Sell checks only suspension,
execution-data validity, and limit-down.

- [ ] **Step 5: Run tests and commit**

Run: `& $python -m unittest rank_model.tests.test_strategy.SignalAndRuleTests -v`

Expected: all signal and execution-rule tests pass.

Commit: `feat(rank-model): add deterministic strategy signals`

---

### Task 3: Implement One-Day Set-Difference Portfolio Transition

**Files:**
- Modify: `rank_model/stages/strategy.py`
- Modify temporarily: `rank_model/tests/test_strategy.py`

**Interfaces:**
- Consumes: `PortfolioState`, one desired Top100 tuple, one execution-date market slice, and settings.
- Produces: `transition_at_open(state, desired, market, signal_date, execution_date, settings) -> TransitionResult`.

- [ ] **Step 1: Add failing state-transition tests**

Cover all confirmed invariants:

```python
def test_intersection_units_never_change(self):
    result = transition_at_open(state_with_A_B, ("B", "C"), market, T, T1, SETTINGS)
    self.assertEqual(result.state.positions["B"].units, state.positions["B"].units)

def test_blocked_sale_is_carried_and_buy_is_not_replaced(self):
    result = transition_at_open(state_with_A, ("B",), blocked_A_and_B, T, T1, SETTINGS)
    self.assertIn("A", result.state.positions)
    self.assertNotIn("B", result.state.positions)
    self.assertEqual(result.diagnostics["replacement_buy_count"], 0)

def test_cash_shortage_scales_all_eligible_buys_equally(self):
    result = transition_at_open(low_cash_state, ("A", "B"), market, T, T1, SETTINGS)
    self.assertAlmostEqual(result.trades.loc["A", "gross_notional"], result.trades.loc["B", "gross_notional"])
```

- [ ] **Step 2: Confirm the transition tests fail**

Run: `& $python -m unittest rank_model.tests.test_strategy.TransitionTests -v`

- [ ] **Step 3: Implement marking, exits, entries, and costs**

Use these exact state types:

```python
@dataclass
class Position:
    units: float
    last_mark: float

@dataclass
class PortfolioState:
    cash: float
    positions: dict[str, Position]
    previous_nav: float
```

At the open: update valid non-suspended marks, compute pre-trade NAV, sell only
`held - desired`, then buy only `desired - held_after_sales`. Requested gross
per eligible entry is `pre_trade_nav / top_k`; apply one common scale no larger
than 1 so gross buys plus buy costs fit available cash. Record one attempted
trade row for every exit and entry candidate, including blocked reasons.

- [ ] **Step 4: Assert accounting identities**

Raise on negative cash beyond `1e-12`, non-positive units, duplicate order keys,
intersection trades, or:

```python
abs(end_nav - (pre_trade_nav - total_cost)) > 1e-10
```

- [ ] **Step 5: Run tests and commit**

Run: `& $python -m unittest rank_model.tests.test_strategy.TransitionTests -v`

Expected: all transition and accounting tests pass.

Commit: `feat(rank-model): simulate set-difference open execution`

---

### Task 4: Run The Full Static Period And Compute Reports

**Files:**
- Modify: `rank_model/stages/strategy.py`
- Modify temporarily: `rank_model/tests/test_strategy.py`

**Interfaces:**
- Consumes: validated signals, official calendar, filtered market panel, and settings.
- Produces: `StrategyBundle`, `simulate_strategy(...)`, `summarize_strategy(...)`, and `write_strategy_run(...)`.

- [ ] **Step 1: Add failing period, suspension, and metric tests**

```python
def test_signal_executes_on_next_official_date(self):
    bundle = simulate_strategy(predictions, market, calendar, SETTINGS)
    self.assertEqual(bundle.trades.iloc[0]["execution_date"], pd.Timestamp("2024-01-03"))

def test_missing_suspended_mark_is_carried_until_reopen(self):
    bundle = simulate_strategy(suspension_fixture, market, calendar, SETTINGS)
    self.assertEqual(bundle.daily_nav.loc[suspended_day, "gross_return"], 0.0)
    self.assertNotEqual(bundle.daily_nav.loc[reopen_day, "gross_return"], 0.0)

def test_target_columns_cannot_change_strategy_output(self):
    first = simulate_strategy(predictions, market, calendar, SETTINGS)
    changed = predictions.assign(target_10d=999.0, rank_target_10d=-999.0)
    second = simulate_strategy(changed, market, calendar, SETTINGS)
    pd.testing.assert_frame_equal(first.daily_nav, second.daily_nav)
```

- [ ] **Step 2: Confirm focused tests fail**

Run: `& $python -m unittest rank_model.tests.test_strategy.PeriodAndReportingTests -v`

- [ ] **Step 3: Implement the full-period loop**

Publish an initial NAV row on the first signal date, then process each later
official trading open from the immediately prior official signal. Do not
execute the final 2025-12-31 signal. Save end-of-open positions after every
transition and retain the final positions without liquidation.

- [ ] **Step 4: Implement exact metrics and immutable publication**

Compute cumulative return, CAGR using elapsed trading observations, annualized
volatility, zero-rate Sharpe, max drawdown, win rate, average cash ratio, gross
and one-way turnover, annualized turnover, total cost, fill rates, blocked-sale
days, and ending NAV. Write through a sibling temporary directory, hash every
artifact, write the manifest last, then atomically rename. Reject an existing
destination and protect publication with a process lock. Require exactly eight
source paths: prediction, prediction manifest, locked-test schema, market
panel, trading calendar, frozen models, locked-test conclusion, and locked-test
comparison. Before trusting the manifest or schema contents, require their
actual hashes to match the schema and exact five-model manifest anchors in the
sealed conclusion; then validate the manifest-to-schema-to-market/calendar
provenance chain.

- [ ] **Step 5: Run tests and commit**

Run: `& $python -m unittest rank_model.tests.test_strategy.PeriodAndReportingTests -v`

Expected: all full-period and report tests pass.

Commit: `feat(rank-model): publish immutable strategy reports`

---

### Task 5: Wire Unified CLI And Five-Model Comparison

**Files:**
- Modify: `rank_model/pipeline.py`
- Modify: `rank_model/stages/strategy.py`
- Modify temporarily: `rank_model/tests/test_strategy.py`

**Interfaces:**
- Consumes: `backtest_locked_strategy(...)` and `compare_strategy_runs(...)`.
- Produces: public `backtest-strategy --model` and `compare-strategy` commands.

- [ ] **Step 1: Add failing parser and comparison tests**

```python
def test_parser_accepts_only_frozen_strategy_models(self):
    args = make_parser().parse_args(["backtest-strategy", "--model", "lightgbm_lambdarank"])
    self.assertEqual(args.model, "lightgbm_lambdarank")

def test_comparison_has_five_rows_and_no_decision_column(self):
    frame = compare_strategy_runs(run_root, output, LOCKED_MODEL_NAMES)
    self.assertEqual(frame["model_name"].tolist(), list(LOCKED_MODEL_NAMES))
    self.assertTrue({"winner", "accept", "champion"}.isdisjoint(frame.columns))
```

- [ ] **Step 2: Confirm CLI tests fail**

Run: `& $python -m unittest rank_model.tests.test_strategy.CliAndComparisonTests -v`

- [ ] **Step 3: Validate strategy config in `load_config` and resolve paths**

Require exact fixed dates, top count, costs, listing days, initial NAV, and all
strategy paths. Ensure outputs resolve under `rank_model/strategy_runs`, while
source market data resolves under `model/data` and sealed inputs resolve under
`rank_model`.

- [ ] **Step 4: Add command dispatch**

```python
strategy_parser = subparsers.add_parser("backtest-strategy")
strategy_parser.add_argument("--model", required=True, choices=LOCKED_MODEL_NAMES)
subparsers.add_parser("compare-strategy")
```

Both commands catch `FileNotFoundError`, `ValueError`, and `FileExistsError` and
report them through `argparse` consistently with existing pipeline commands.

- [ ] **Step 5: Run tests and commit**

Run: `& $python -m unittest rank_model.tests.test_strategy.CliAndComparisonTests -v`

Expected: parser, path, dispatch, and comparison tests pass.

Commit: `feat(rank-model): expose static strategy commands`

---

### Task 6: Document, Run Real Acceptance, And Remove Temporary Tests

**Files:**
- Modify: `rank_model/README.md`
- Delete: `rank_model/tests/test_strategy.py`
- Delete if empty: `rank_model/tests/__init__.py`

**Interfaces:**
- Consumes: completed CLI and the real local locked-test artifacts.
- Produces: five immutable real strategy runs and `strategy_comparison.csv`.

- [ ] **Step 1: Document commands, formulas, boundaries, and artifacts**

Add the five-model loop:

```powershell
$models = @(
  "ridge_rank_regression",
  "xgboost_rank_regression",
  "lightgbm_rank_regression",
  "lightgbm_lambdarank",
  "mlp_top100_hybrid_rank"
)
foreach ($model in $models) {
  & $python -m rank_model.pipeline --config rank_model\config.toml `
    backtest-strategy --model $model
}
& $python -m rank_model.pipeline --config rank_model\config.toml compare-strategy
```

- [ ] **Step 2: Run all temporary tests together**

Run: `& $python -m unittest rank_model.tests.test_strategy -v`

Expected: all tests pass.

- [ ] **Step 3: Run read-only CLI and source acceptance**

Run:

```powershell
& $python -m rank_model.pipeline --config rank_model\config.toml doctor
& $python -m rank_model.pipeline --config rank_model\config.toml --help
```

Expected: doctor passes and help lists both strategy commands.

- [ ] **Step 4: Run the five real 2024-2025 backtests and comparison**

Expected per model: 485 NAV rows including the initial state, 484 execution
dates, no execution beyond 2025-12-31, no duplicate trade/position keys,
non-negative cash within tolerance, and an ending-position snapshot.

- [ ] **Step 5: Perform real-output invariants**

Verify all five manifests have identical strategy parameters and source period;
every trade's signal and execution dates are adjacent in the official calendar;
intersection stocks have no orders; no failed buy has a replacement; total cost
matches trade-level costs; daily NAV accounting reconciles; comparison has five
rows and no decision field; and hashes validate after reopening every artifact.

- [ ] **Step 6: Remove temporary tests and caches**

Delete the temporary test module and all `__pycache__` directories created by
this task. Do not delete production reports or tracked documentation.

- [ ] **Step 7: Run final production checks and commit**

Run:

```powershell
& $python -m compileall -q rank_model
& $python -m rank_model.pipeline --config rank_model\config.toml --help
git diff --check
git status --short
```

Expected: compilation and CLI help pass, no temporary test or cache remains,
and only intended production files are tracked.

Commit: `docs(rank-model): document static strategy workflow`

## Self-Review

- Spec coverage: every data boundary, trade rule, cost, valuation rule, output,
  freeze rule, and 2026 handoff is assigned to a task.
- Placeholder scan: the plan contains no deferred implementation markers.
- Type consistency: Tasks 1-6 consistently use `StrategySettings`,
  `PortfolioState`, `Position`, `TransitionResult`, and `StrategyBundle`.
