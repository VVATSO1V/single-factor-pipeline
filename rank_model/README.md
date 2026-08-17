# CSI1000 10-Day Rank Model

This package trains fixed, cross-sectional ranking specifications on the
prepared CSI1000 10-day development dataset. It only reads the source bundle
under `model/data`; it publishes prepared data and immutable model runs under
`rank_model`.

## Production Workflow

Run these commands from the repository root after the package is integrated:

```powershell
.\.venv\Scripts\python.exe -m rank_model.pipeline doctor
.\.venv\Scripts\python.exe -m rank_model.pipeline prepare
.\.venv\Scripts\python.exe -m rank_model.pipeline train --model ridge_rank_regression --run-id ridge_rank_regression_10d
.\.venv\Scripts\python.exe -m rank_model.pipeline evaluate --run-id ridge_rank_regression_10d --split validation
```

`doctor` is read-only. It verifies source presence and hashes, date and exit
boundaries, one thousand unique non-null keys per source date, finite declared
continuous features, forbidden-feature exclusions, runtime imports, published
hashes when present, and path containment. `prepare` writes the rank dataset.
`train` creates an immutable run: reusing a run ID fails rather than replacing
existing artifacts. `evaluate` only accepts the `validation` split. To compare
completed runs without selecting a winner, use:

```powershell
.\.venv\Scripts\python.exe -m rank_model.pipeline compare --run-ids ridge_rank_regression_10d,xgboost_rank_regression_10d
```

## Registered Models

- `ridge_rank_regression`
- `xgboost_rank_regression`
- `lightgbm_rank_regression`
- `mlp_rank_regression`
- `xgboost_pairwise_rank`
- `lightgbm_lambdarank`
- `mlp_pairwise_rank`
- `mlp_top100_hybrid_rank`

The four regression models fit the continuous percentile label. The three
native rankers and the hybrid MLP group observations strictly by trading date;
no training pair or ranking query crosses dates. Training rows from 2019-2022
and their exit dates remain in that period. Validation rows and exits remain in
2023. All training dates receive equal total loss weight.

`mlp_top100_hybrid_rank` combines equal-date rank MSE with a weighted pairwise
loss between each realized training Top100 stock and lower-ranked opponents.
It updates once per complete-date batch and restores the epoch with the highest
2023 mean daily NDCG@100, subject to the configured patience and minimum delta.
Rank IC is recorded at the selected epoch as a broad-ordering diagnostic. This
is model selection on validation data, not a 2024-2025 test result.

Run the hybrid candidate and compare it with the immutable baselines:

```powershell
& $python -m rank_model.pipeline --config rank_model\config.toml train `
  --model mlp_top100_hybrid_rank `
  --run-id real-2023-mlp-top100-hybrid

& $python -m rank_model.pipeline --config rank_model\config.toml evaluate `
  --run-id real-2023-mlp-top100-hybrid `
  --split validation

& $python -m rank_model.pipeline --config rank_model\config.toml compare `
  --run-ids real-2023-ridge,real-2023-xgb-reg,real-2023-lgb-reg,real-2023-xgb-rank,real-2023-lgb-rank,real-2023-mlp-reg,real-2023-mlp-rank,real-2023-mlp-top100-hybrid
```

## Data Contracts

The source schema declares the keys `date` and `stock_code`, continuous feature
columns, categorical `industry`, raw target `target_10d`, split `split_10d`,
and exit date `exit_date_10d`. Source dates have exactly 1000 unique,
non-null keys. The preparation step preserves raw returns and adds
`rank_target_10d`:

```text
(average_rank(target_10d) - 1) / (finite_count - 1)
```

Ranks are ascending, so a larger finite raw return has a larger rank label.
Rows without a raw target retain their keys and have a missing rank label. The
model feature list rejects keys, targets, split fields, exit dates,
`entry_tradeable`, and future-status fields. Continuous source features must be
finite after source preparation; industry is fitted as a train-only categorical
vocabulary and unknown values map to `UNKNOWN`.

`score_raw` is an ordinal model score, not a calibrated expected return. Its
scale and score differences must not be read as return forecasts. Evaluation
converts scores to daily percentiles for ordering metrics while retaining raw
`target_10d` for return, Top100, decile, and spread reporting.

## Freeze And Final Refit

After choosing specifications with the 2023 validation results, freeze the
five selected candidates before using the locked 2024-2025 test period:

```powershell
& $python -m rank_model.pipeline --config rank_model\config.toml freeze
```

`frozen_models.json` records each source validation run, its manifest and
feature-schema SHA-256, transformed feature order, fixed model parameters,
source-data hashes, a logical audit of the 2023 labels, and the exact physical
and full-table logical hashes of the sealed 2019-2023 rank dataset. It is immutable;
deleting it is an explicit research decision, not part of a normal rerun.

Refit each frozen specification on all eligible 2019-2023 labels:

```powershell
& $python -m rank_model.pipeline --config rank_model\config.toml refit --model ridge_rank_regression
& $python -m rank_model.pipeline --config rank_model\config.toml refit --model xgboost_rank_regression
& $python -m rank_model.pipeline --config rank_model\config.toml refit --model lightgbm_rank_regression
& $python -m rank_model.pipeline --config rank_model\config.toml refit --model lightgbm_lambdarank
& $python -m rank_model.pipeline --config rank_model\config.toml refit --model mlp_top100_hybrid_rank
```

Final refits include finite labels whose signal date is in 2019-2023 and whose
10-day target exit is no later than 2023-12-31. This reclaims observations that
were purged only at the old 2022/2023 train-validation boundary, while still
excluding labels that need 2024 prices. No validation split, early stopping,
test prediction, or strategy rule is used. The selected hybrid MLP runs a fixed
12 epochs. Final artifacts are immutable under
`rank_model/final_runs/<model-name>` and intentionally contain no prediction
file; 2024-2025 prediction is a later, separate locked-test step. `refit`
verifies the sealed rank file before opening it and does not read upstream
source data.

## 2024-2025 Locked Test

After all five final refits exist, build test features and labels from the
already prepared local `model/data` files. This does not access Ricequant:

```powershell
& $python -m rank_model.pipeline --config rank_model\config.toml prepare-test
```

The result retains all 485 CSI1000 signal dates from 2024-01-02 through
2025-12-31 and exactly 1000 keys per date. Signals after 2025-12-16 do not have
a complete 10-day exit inside 2025, so their keys and predictions remain while
their target and rank label are missing. The builder reads 20 prior trading
days only to warm up point-in-time market and industry features.

Run static inference and evaluation for each frozen model:

```powershell
$models = @(
  "ridge_rank_regression",
  "xgboost_rank_regression",
  "lightgbm_rank_regression",
  "lightgbm_lambdarank",
  "mlp_top100_hybrid_rank"
)

foreach ($model in $models) {
  & $python -m rank_model.pipeline --config rank_model\config.toml predict-test `
    --model $model
  & $python -m rank_model.pipeline --config rank_model\config.toml evaluate-test `
    --model $model
}

& $python -m rank_model.pipeline --config rank_model\config.toml compare-test
```

`predict-test` loads the fitted 2019-2023 preprocessor and model without
refitting either one. It verifies final-model, frozen-spec, source-data, schema,
feature-order, feature-code, and feature-parameter contracts before inference.
`evaluate-test` reuses the same
metric definitions and evaluation code as the 2023 validation stage, then
writes combined, daily, monthly, yearly, decile, and Top100 outputs.
`compare-test` writes one five-row metric table without choosing a winner.

The dataset, prediction file, and comparison are immutable. A run directory
allows one lock-protected evaluation append after prediction, then becomes
sealed; a normal rerun fails instead of overwriting it. Dataset creation,
per-model inference, evaluation, and comparison all exclude concurrent writers.
`entry_tradeable` and all T+1 status filters remain excluded here and enter only
the later strategy backtest.

## 2024-2025 Static Strategy

After all five sealed locked-test prediction runs exist, run the same static
Top100 strategy for every frozen model and publish the descriptive comparison:

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

Both commands preserve immutable outputs. `backtest-strategy` fails when its
model directory already exists. `compare-strategy` validates an existing CSV
and sidecar against all five current run manifests and returns without rewriting
either file when the pair is valid. A legacy CSV without a sidecar gains one
only after its exact schema and values are proven equal to freshly derived run
summaries. Tampered output, changed provenance, or any other incomplete pair
fails without overwrite. Delete neither output as part of a normal rerun.

The strategy period is the 485 official dates from 2024-01-02 through
2025-12-31. A Top100 signal observed after the close on date T executes only at
the next official open T+1, producing 484 execution observations plus one
initial NAV row. The 2025-12-31 signal is not executed because its next open is
outside the sealed period; dates from 2026 onward remain reserved for forward
simulation. Orders use only finite `score_raw`, with `stock_code` ascending as
the deterministic tie breaker. Targets, locked-test metrics, and comparison
results never enter selection, sizing, or execution.

Rebalancing uses strict set differences. Existing Top100 holdings are untouched,
holdings outside Top100 become sell attempts, and new Top100 names become buy
attempts. A blocked sell remains held and is retried while outside Top100. A
blocked buy is not replaced by rank 101 or any other name. Every eligible buy
requests `pre_trade_nav / 100`; when available cash cannot fund all eligible
buys plus costs, one common scale factor is applied to all of them.

A buy requires a non-ST, non-suspended stock with at least 120 listing days,
valid positive open, adjusted-open, and limit data, and an open at neither
limit. A sell is blocked only by suspension, missing valid execution or
valuation data, or an open at limit-down; ST status, listing age, and limit-up
do not block a sale. Limit checks use unadjusted `raw_open`, `limit_up`, and
`limit_down`; units and valuation use adjusted `post_open`. A holding without a
valid mark carries its last valid adjusted-open mark until trading resumes. The
period ends without forced liquidation.

The execution and report formulas are:

```text
pre_trade_nav = cash + sum(units * current_or_carried_post_open)
requested_buy_gross = pre_trade_nav / 100
buy_cost = executed_buy_gross * (0.0001 commission + 0.0005 slippage)
sell_cost = executed_sell_gross *
            (0.0001 commission + 0.0005 slippage + 0.0005 stamp duty)
end_nav = pre_trade_nav - total_cost
gross_return = pre_trade_nav / prior_execution_end_nav - 1
net_return = end_nav / prior_execution_end_nav - 1
gross_turnover = (executed_buy_gross + executed_sell_gross) / pre_trade_nav
one_way_turnover = gross_turnover / 2
cumulative_return = ending_nav / initial_nav - 1
cagr = (ending_nav / initial_nav) ** (252 / 484) - 1
annualized_volatility = sample_std(net_return) * sqrt(252)
sharpe_ratio = mean(net_return) / sample_std(net_return) * sqrt(252)
max_drawdown = max(1 - nav / running_max_nav)
win_rate = count(net_return > 0) / 484
fill_rate = executed_orders / attempted_orders, separately by side
```

The initial NAV row is included in maximum drawdown and average cash ratio, but
excluded from return, turnover, cost, and win-rate observations. Average
turnover uses the 484 execution rows; annualized turnover multiplies the daily
average by 252. `total_cost` is the sum of daily trade-level costs, and blocked
sale days count distinct execution dates with at least one blocked sell.

Each directory under `rank_model/strategy_runs/<model-name>` contains:

```text
daily_nav.csv
trades.parquet
positions.parquet
execution_diagnostics.csv
ending_positions.csv
metrics_summary.json
manifest.json
```

The manifest records fixed settings, formula and observation contracts, row
counts, and SHA-256 hashes for all eight sealed inputs and six report outputs.
`ending_positions.csv` contains the final positions and cash snapshot for
forward simulation. `rank_model/strategy_comparison.csv` contains exactly five
rows with `model_name` plus the exact strategy metric schema; it intentionally
has no winner, decision, acceptance, rejection, or parameter-update field.
The adjacent `rank_model/strategy_comparison.manifest.json` is the comparison
commit marker. It binds the CSV's physical and canonical logical SHA-256 hashes,
ordered column/type schema and row count, the relative paths and SHA-256 hashes
of all five source strategy manifests, and a sealed contract hash over common
settings, formulas, observation conventions, and source-input provenance. A
first publication stages both files and rolls the CSV back if the sidecar cannot
be committed, so readers never accept a CSV without its provenance seal.

## 2024-2025 锁定测试：十交易日错位换仓策略

十交易日策略用于检验“每十个交易日换仓”是否对固定模型的表现更稳健，同时避免
只选择某一个幸运起始日。它不是把一份资金拆成十份，而是为每个模型建立 **10 条彼此
独立、各自以 1.0 初始净值运行的满资金路径**。在某条路径首次交易前，该路径始终
保持全现金；十条路径只在研究汇总时等权平均。

静态测试只使用 2024-01-02 至 2025-12-16 的 474 个完整信号日。按照官方交易日历将
这些日期编号为 1 至 474；offset `k`（`k = 1..10`）使用编号
`k, k + 10, k + 20, ...` 的日期。因此，十条路径合起来恰好覆盖每个完整信号日一次，
不重叠也不遗漏。2025-12-17 至 2025-12-31 的信号没有完整的十日持有期，保留给
2026 年后的前向模拟，不进入这里的静态测试。

对任一排程信号日 `T`：

```text
T 收盘后：       只按当天有限的 score_raw 和 stock_code 的确定性顺序选出 Top100
T+1 复权开盘：   执行该次调仓
T+11 复权开盘：  该信号的完整十个交易日持有期终值；也是下一次同一路径调仓的执行时点
```

每条路径在自己的最后一个完整 `T+11` 开盘价处结束估值，不强制清仓。十条路径的
终止日期最多相差九个交易日。研究平均净值只使用十条路径都实际有净值的共同区间：
从 2024-01-02 的全现金初值到最早的完整终止日 2025-12-18，绝不为了对齐而向后填充
任何路径。平均净值、收益、波动率、Sharpe、回撤、换手和成本都由平均后的实际经济量
重新计算，不能把十条路径的 Sharpe 直接平均。

调仓沿用日频策略的同一执行状态机和交易限制，并采用集合差换仓：新 Top100 与现有
持仓的交集保留原有单位，不会先卖出再买回；仅尝试卖出不在新集合中的股票，并仅尝试
买入新集合中尚未持有的股票。被阻塞的卖出订单只要仍不在目标集合中就会在每个交易日
重试；买入失败后在本轮十交易日周期内不重试，也不以第 101 名或其他股票递补，所以
未成交部分保持现金。ST、停牌、上市天数、涨跌停、价格有效性、仓位缩放、交易成本和
持仓估值口径均与日频策略保持一致。

在仓库根目录执行以下命令，按顺序为五个已冻结模型生成十交易日结果，再生成五模型
汇总和日频对比。此工作流会运行真实的 2024-2025 锁定测试；不要把测试结果用于重新
调参、重新训练、增删特征或修改策略参数，否则该测试集将不再是独立测试集。

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

每个模型在 `rank_model/strategy_10d_runs/<model-name>/` 下生成十个
`offset_01` 至 `offset_10` 路径目录。每条路径都包含 `daily_nav.csv`、
`trades.parquet`、`positions.parquet`、`execution_diagnostics.csv`、
`ending_positions.csv`、`metrics_summary.json` 和 `manifest.json`，因此五个模型
共计 50 份路径报告。模型目录同时包含：

```text
offset_metrics.csv      # 十条路径各自的指标
offset_summary.csv      # 每项路径指标的均值、中位数、样本标准差、最小值、最大值
average_nav.csv         # 十条独立满资金路径的共同区间平均净值
average_metrics.json    # 由平均净值重新计算的整体指标
manifest.json           # 排程、来源、设置和全部输出的封存清单
```

五模型横向结果写入 `rank_model/strategy_10d_comparison.csv` 及其 manifest；
`rank_model/daily_vs_10d_comparison.csv` 及其 manifest 则对每个模型逐项给出
“十交易日策略指标 - 日频策略指标”，包括收益、波动率、Sharpe、回撤、现金占比、
年化换手和总成本。两张表都只呈现结果，不自动选择赢家。

所有十交易日输出都是不可变的。`backtest-strategy-10d` 若发现同一模型目录已经存在
会拒绝覆盖；`compare-strategy-10d` 若发现完整、哈希和来源均有效的既有 CSV/manifest
对，会验证后原样返回，不重写文件。缺失、部分写入、来源变化或任何篡改都会失败关闭，
而不是静默修补或覆盖旧结果。

## Artifacts

`prepare` creates the following files under `rank_model/data`:

```text
rank_dataset_10d.parquet
rank_dataset_10d_schema.json
rank_label_coverage.csv
```

The schema records source hashes, output hash, features, target formula, and
row/date counts. Every run under `rank_model/runs/<run-id>` contains:

```text
config_snapshot.toml
feature_schema.json
preprocessor.joblib
model artifact(s)
predictions_10d.parquet
metrics_summary.json
daily_metrics.csv
monthly_metrics.csv
yearly_metrics.csv
decile_returns.csv
top100_detail.parquet
manifest.json
```

Predictions use this schema:

```text
date,stock_code,split,horizon,target_10d,rank_target_10d,score_raw,pred_rank_pct,pred_rank_position
```

The comparison output has one row per completed run and metric columns only.
It intentionally contains no automatic `accept`, `reject`, or `champion`
field: specification choice remains a human decision.

Locked-test artifacts use the parallel structure:

```text
data/locked_test_rank_dataset_10d.parquet
data/locked_test_rank_dataset_10d_schema.json
data/locked_test_rank_label_coverage.csv
locked_test_runs/<model-name>/predictions_10d.parquet
locked_test_runs/<model-name>/metrics_summary.json
locked_test_runs/<model-name>/daily_metrics.csv
locked_test_runs/<model-name>/monthly_metrics.csv
locked_test_runs/<model-name>/yearly_metrics.csv
locked_test_runs/<model-name>/decile_returns.csv
locked_test_runs/<model-name>/top100_detail.parquet
locked_test_runs/<model-name>/manifest.json
locked_test_runs/locked_test_comparison.csv
```
