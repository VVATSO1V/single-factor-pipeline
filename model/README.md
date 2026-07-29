# 多因子模型流水线

`model` 将各单因子目录已经生成的 `factor.csv` 整理成统一的机器学习数据集。

当前阶段已经完成：

- 中证 1000 历史市场面板与官方交易日历。
- 17 个因子长表合并。
- T+1 开盘进入的 1、5、10 日绝对收益标签。
- 建模主表、特征白名单和 T+1 执行审计字段。
- 2019-2022 训练、2023 验证、2024-2025 锁定测试的样本索引。

模型训练、模型评价和策略回测尚未实现。本 README 中相应接口只描述后续约定，不代表当前已经可以运行。

## 核心口径

| 项目 | 口径 |
|---|---|
| 股票池 | 各交易日当时的中证 1000 成分股 |
| 信号时点 | T 日收盘后 |
| 计划成交 | T+1 开盘 |
| 收益价格 | 后复权开盘价 `post_open` |
| 预测目标 | 个股未来绝对收益 |
| 预测期限 | 1、5、10 个交易日 |
| 主键 | `date,stock_code` |

标签定义：

```text
target_1d(T)  = post_open(T+2)  / post_open(T+1) - 1
target_5d(T)  = post_open(T+6)  / post_open(T+1) - 1
target_10d(T) = post_open(T+11) / post_open(T+1) - 1
```

例如 `target_5d(T)` 表示 T+1 开盘买入、T+6 开盘卖出的绝对收益。

## 使用者需要了解的文件

```text
model/
├─ README.md
├─ config.toml
├─ pipeline.py
├─ stages/
│  ├─ market.py
│  ├─ features.py
│  ├─ labels.py
│  └─ dataset.py
├─ data/
├─ runs/
└─ docs/
```

日常只需要操作：

- `config.toml`：数据、因子、标签和时间划分的唯一配置。
- `pipeline.py`：统一命令入口。
- `README.md`：运行和数据口径说明。

`stages` 是内部实现，不应直接运行。以后增加预处理、训练、评价和策略时，也继续由 `pipeline.py` 统一调用。

## 配置

`config.toml` 集中维护：

- 股票池代码和数据日期。
- 米筐 `.env` 路径。
- 所有输入输出路径。
- 17 个因子路径及顺序。
- 1、5、10 日标签口径。
- train、validation、test 和 final_train 日期。
- 最低上市天数。

配置中的相对路径均以 `model/config.toml` 所在目录为基准，而不是以 PowerShell 当前目录为基准。

米筐账号继续保存在项目根目录 `.env`：

```text
RQDATA_USER=你的账号
RQDATA_PASSWORD=你的密码
```

账号密码不能写入 `config.toml`，也不能提交到 Git。

### 新增或删除因子

只修改 `config.toml`：

```toml
[factors]
paths = [
  "../rev_1d/data/factor.csv",
  "../new_factor/data/factor.csv",
]
```

因子文件必须遵守：

```text
<factor_name>/data/factor.csv
```

内容至少包含：

```csv
date,stock_code,factor_value
```

因子列名从目录名自动推断。配置中的路径顺序就是模型特征列顺序，不需要修改 Python 文件。

## 运行准备

在项目根目录执行：

```powershell
Set-Location "C:\Users\cbnb\Desktop\单因子测试流水线"
```

环境至少需要：

```text
Python 3.11
pandas
numpy
pyarrow
rqdatac
```

各因子目录的 `data/factor.csv` 必须先由单因子构建脚本生成。`model` 不会自动调用 17 个因子的构建脚本，也不会重复计算因子。

## 统一命令

### 1. 检查环境和数据契约

```powershell
.\.venv\Scripts\python.exe -m model.pipeline doctor
```

`doctor` 只读检查：

- Python 和依赖。
- 17 个因子文件及字段。
- 因子名称是否重复。
- 市场面板和交易日历。
- 现有主表与 schema 哈希。
- 现有样本索引与分割汇总哈希。
- 配置中的特征顺序和时间划分。

它不会登录米筐，也不会写入或覆盖数据。

### 2. 显式获取市场数据

只有市场面板不存在，或者确实需要更新日期和股票池时才运行：

```powershell
.\.venv\Scripts\python.exe -m model.pipeline fetch-market
```

已有文件时默认停止，防止误覆盖和消耗米筐额度。明确需要覆盖时：

```powershell
.\.venv\Scripts\python.exe -m model.pipeline fetch-market --force
```

输出：

```text
model/data/market_panel.csv
model/data/trading_calendar.csv
```

这是当前唯一会调用米筐的 `model` 命令。

### 3. 构建全部本地建模数据

```powershell
.\.venv\Scripts\python.exe -m model.pipeline prepare-data
```

执行顺序固定为：

```text
factor_wide
    ↓
target
    ↓
model_dataset + schema
    ↓
sample_index + split_summary
```

`prepare-data` 只读取本地市场面板和因子文件，绝不会调用米筐。缺少输入时会在写入任何输出前停止并列出缺失路径。

已有输出会使用同样路径覆盖。重新运行前不需要手动删除。Parquet、schema
和样本索引采用临时文件发布，较大的 CSV 按原有写入方式直接覆盖。

## 数据产物

| 文件 | 作用 |
|---|---|
| `data/market_panel.csv` | 完整交易日历 × 历史相关股票的市场面板 |
| `data/trading_calendar.csv` | 米筐官方交易日历 |
| `data/factor_wide.csv` | 每日 1000 只成分股 × 17 个因子 |
| `data/target.csv` | 1、5、10 日绝对收益标签 |
| `data/model_dataset.parquet` | 原始建模主表 |
| `data/model_dataset_schema.json` | 特征白名单、字段分组和主表哈希 |
| `data/model_sample_index.parquet` | 三个期限的时间划分和退出日期 |
| `data/model_split_summary.json` | 分割计数、日期边界和索引哈希 |

当前全量数据契约：

| 文件 | 行数 | 列数 |
|---|---:|---:|
| `factor_wide.csv` | 1,699,000 | 19 |
| `target.csv` | 1,699,000 | 5 |
| `model_dataset.parquet` | 1,699,000 | 35 |
| `model_sample_index.parquet` | 1,699,000 | 12 |

所有表的 `(date,stock_code)` 必须唯一。`factor_wide.csv` 每个交易日必须恰好有 1000 行。

## 市场面板和未来函数

市场面板先使用：

```text
官方交易日历 × 历史成分股并集
```

建立完整骨架，再合并行情、状态、行业和市值。这样即使某日停牌或行情源缺失，也不会把更晚日期误认为 T+1。

始终遵守：

1. T 日特征只能使用 T 日及以前可见的数据。
2. 历史股票池使用当时的指数成分股快照。
3. 标签从 T+1 开盘开始。
4. 财务和分析师因子按公告日或预测发布日期生效。
5. 标签和时间退出日期均以官方交易日历位移。
6. 特征缺失值在数据准备阶段保留，不填 0。
7. `target_*` 和 `entry_*` 不能进入模型特征。
8. `entry_tradeable` 不筛选模型训练样本，只在策略执行阶段使用。

## 时间划分

```text
2019-2022：train
2023：validation 和模型选择
2024-2025：锁定静态 test
2019-2023：选模后的 final_train
2026 以后：前向模拟和实盘观察
```

1、5、10 日分别根据真实退出日期 purge 跨越时期边界的样本：

```text
1 日退出：T+2
5 日退出：T+6
10 日退出：T+11
```

当前样本数量：

| 期限 | train | validation | test | final_train |
|---|---:|---:|---:|---:|
| 1 日 | 969,998 | 240,000 | 483,000 | 1,211,998 |
| 5 日 | 965,994 | 236,000 | 478,993 | 1,207,994 |
| 10 日 | 960,989 | 231,000 | 473,983 | 1,202,989 |

## 修改后从哪一步重跑

| 修改内容 | 需要执行 |
|---|---|
| 股票池、市场日期、市场字段 | `fetch-market --force`，再执行 `prepare-data` |
| 新增、删除或更新因子 | `prepare-data` |
| 标签口径或预测期限 | 修改代码并执行 `prepare-data` |
| 时间划分或最低上市天数 | 修改配置并执行 `prepare-data` |
| 模型参数 | 未来只重新 `train` |
| 评价指标 | 未来只重新 `evaluate` |
| Top N 或交易成本 | 未来只重新 `backtest` |

## 后续模型接口

以下命令是已经确定的接口设计，当前尚未实现：

```powershell
.\.venv\Scripts\python.exe -m model.pipeline train --model ridge --horizon all
.\.venv\Scripts\python.exe -m model.pipeline evaluate --run-id ridge_baseline --split validation
.\.venv\Scripts\python.exe -m model.pipeline backtest --run-id ridge_baseline
```

约定：

- 一个 `run_id` 对应一种模型配置，可以包含 1、5、10 日三个独立模型。
- `evaluation` 根据 `run_id` 读取模型清单和预测结果，不重新训练。
- 预测结果统一保存为：

```csv
date,stock_code,split,horizon,y_true,y_pred
```

- 开发阶段默认只评价 2023 validation。
- 模型选择完成后使用 2019-2023 重新训练，再生成 2024-2025 test 预测。
- 策略阶段才使用 T+1 `entry_tradeable`、交易成本、现金和无法成交状态。

## Git 和数据文件

`model/data` 和 `model/runs` 都是本地生成内容，不提交 GitHub。其他人复现时需要：

1. 生成配置中列出的各因子 `factor.csv`。
2. 生成或取得 `market_panel.csv` 和 `trading_calendar.csv`。
3. 运行 `doctor`。
4. 运行 `prepare-data`。
