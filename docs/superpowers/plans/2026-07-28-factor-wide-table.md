# Factor Wide Table Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `model` workspace that builds a historical CSI1000 factor-wide table from the existing market panel and all 17 factor CSV files.

**Architecture:** Copy one existing market panel into `model/data/market_panel.csv`. A single `model/build_factor_table.py` script derives factor names from each path in `FACTOR_PATHS`, filters the market panel to point-in-time constituents, aligns each factor to the unique `(date, stock_code)` master index, and writes one CSV without filling missing factor values.

**Tech Stack:** Python 3, pandas, NumPy, CSV.

## Global Constraints

- Do not modify any existing factor builder, factor CSV, market panel, README, or pipeline file.
- Keep factor configuration in one `FACTOR_PATHS` list; adding or removing a factor requires editing only that list.
- Preserve one row for every historical CSI1000 constituent-date pair selected by `in_universe`.
- Fail on duplicate keys, duplicate inferred factor names, missing required columns, or row-count changes.
- Preserve missing factor values as empty CSV fields.
- Temporary acceptance-test files must be deleted after verification.

---

### Task 1: Build and verify the factor-wide table

**Files:**
- Create: `model/build_factor_table.py`
- Create: `model/data/market_panel.csv`
- Create on full run: `model/data/factor_wide.csv`
- Temporary test: `tmp/model_factor_table_acceptance.py`

**Interfaces:**
- Consumes: `model/data/market_panel.csv` and paths listed in `FACTOR_PATHS`.
- Produces: `build_factor_table(market_panel_path, factor_paths, output_path) -> pandas.DataFrame`.

- [ ] Write a temporary failing acceptance test covering historical-universe filtering, automatic factor naming, left alignment, missing values, duplicate-key rejection, and stable column order.
- [ ] Run the acceptance test and confirm it fails before `model/build_factor_table.py` exists.
- [ ] Implement strict CSV loaders, factor-name inference, index alignment, progress logging, coverage logging, and the command-line entry point.
- [ ] Copy `rev_5d/data/market_panel.csv` to `model/data/market_panel.csv`.
- [ ] Run the temporary acceptance test and confirm all checks pass.
- [ ] Run the script against the copied market panel and all 17 factors.
- [ ] Verify output rows equal 1,699,000, keys are unique, dates contain 1,000 rows each, and all 17 factor columns are present.
- [ ] Delete the temporary acceptance test and its fixture files.
