# 多因子模型流水线

`model` 将各单因子目录已经生成的 `factor.csv` 整理成统一的机器学习数据集。

当前阶段已经完成：

- 中证 1000 历史市场面板与官方交易日历。
- 17 个因子长表合并。
- T+1 开盘进入的 1、5、10 日绝对收益标签。
- 建模主表、特征白名单和 T+1 执行审计字段。
- 2019-2022 训练、2023 验证、2024-2025 锁定测试的样本索引。
- Ridge、XGBoost、LightGBM 和 MLP 神经网络 1、5、10 日 baseline 的预处理、
  调参、训练和验证集预测。
- Residual Rank MLP 和 10 日 Context/Decomposition 实验入口及其开发期约束。

当前统一入口已经覆盖数据检查、市场数据获取、建模数据构建和开发期训练。
完整评价命令、2019-2023 最终重训、2024-2025 锁定测试和策略回测不在当前
`model.pipeline` 入口中实现，不能把 `train` 的 validation 输出当成锁定测试结果。

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
│  ├─ dataset.py
│  ├─ preprocessing.py
│  ├─ training.py
│  ├─ context.py
│  └─ context_training.py
├─ data/
├─ runs/
└─ docs/
```

日常只需要操作：

- `config.toml`：数据、因子、标签和时间划分的唯一配置。
- `pipeline.py`：统一命令入口。
- `README.md`：运行和数据口径说明。

`stages` 是内部实现，不应直接运行。以后增加预处理、训练、评价和策略时，也继续由 `pipeline.py` 统一调用。

各模块职责如下：

| 模块 | 作用 |
|---|---|
| `market.py` | 读取或构建市场面板、交易日历和 T+1 所需行情字段 |
| `features.py` | 将因子长表合并为宽表，并整理模型特征列 |
| `labels.py` | 根据 `post_open` 构建 1、5、10 日绝对开盘到开盘收益 |
| `dataset.py` | 合并特征和标签，生成时间切分、退出日期和审计字段 |
| `preprocessing.py` | 按日去极值、横截面标准化和缺失值处理 |
| `training.py` | 注册模型、训练候选、用 2023 validation 选模型并保存 run |
| `context.py` | 构造市场和行业环境特征 |
| `context_training.py` | 运行 10 日 Context/Decomposition 开发实验 |

## 配置

`config.toml` 集中维护：

- 股票池代码和数据日期。
- 米筐 `.env` 路径。
- 所有输入输出路径。
- 17 个因子路径及顺序。
- 1、5、10 日标签口径。
- train、validation、test 和 final_train 日期。
- 最低上市天数。
- 每日横截面去极值参数和完整截面行数。
- Ridge、XGBoost、LightGBM、MLP 和10日Context/Decomposition实验参数。

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
scikit-learn
joblib
xgboost
lightgbm
torch
```

安装或补齐依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\model\requirements.txt
```

当前神经网络baseline使用CPU版PyTorch。需要单独安装时：

```powershell
.\.venv\Scripts\python.exe -m pip install torch `
  --index-url https://download.pytorch.org/whl/cpu
```

各因子目录的 `data/factor.csv` 必须先由单因子构建脚本生成。`model` 不会自动调用 17 个因子的构建脚本，也不会重复计算因子。

## 快速运行流程

下面命令都在项目根目录执行，并显式传入配置文件，方便复现：

```powershell
Set-Location "C:\Users\cbnb\Desktop\单因子测试流水线"
$python = ".\.venv\Scripts\python.exe"
$config = ".\model\config.toml"

# 1. 只读检查输入、字段、因子顺序和时间划分
& $python -m model.pipeline --config $config doctor

# 2. 只有需要从米筐重新获取 market panel 时才运行
& $python -m model.pipeline --config $config fetch-market

# 3. 使用本地 market panel 和17个 factor.csv 构建建模数据
& $python -m model.pipeline --config $config prepare-data

# 4. 训练一个模型的1、5、10日开发期 baseline
& $python -m model.pipeline --config $config train `
  --model ridge `
  --horizon all `
  --run-id ridge_baseline
```

各命令的输入输出关系：

```text
doctor
  只读检查，不写数据，不访问米筐
        ↓
fetch-market（可选）
  米筐 → model/data/market_panel.csv + trading_calendar.csv
        ↓
prepare-data
  17 个 factor.csv + market panel
  → factor_wide.csv → target.csv → model_dataset.parquet
  → model_sample_index.parquet + model_split_summary.json
        ↓
train
  model_dataset + sample_index
  → model/runs/<run_id>/ 模型、预处理器、validation 预测和 manifest
```

`fetch-market` 是当前唯一会访问米筐的模型命令；`prepare-data`、`prepare-context`
和 `train` 只读取本地文件。已有 market panel 时不要重复执行 `fetch-market --force`，
避免无必要地消耗米筐额度。

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

### 4. 训练模型 baseline

`--model` 决定训练器，`--horizon` 决定训练 1、5、10 日中的一个或全部，
`--run-id` 是本次实验的唯一目录名。不同模型或参数必须使用不同的 `run-id`，
避免覆盖已有实验。

| 模型名 | 作用 | 支持期限 |
|---|---|---|
| `ridge` | 线性正则化基线，用于判断因子线性增量 | 1、5、10、`all` |
| `xgboost` | CPU 树模型，捕捉非线性和特征交互 | 1、5、10、`all` |
| `lightgbm` | CPU 轻量梯度提升树，捕捉非线性和交互 | 1、5、10、`all` |
| `mlp` | CPU PyTorch 神经网络基线 | 1、5、10、`all` |
| `residual_rank_mlp` | Ridge 预测之上的残差和排序开发实验 | 1、5、10、`all` |
| `context_decomposition` | 10 日市场/行业环境与个股分解实验 | 仅10 |

单独训练一个模型：

```powershell
.\.venv\Scripts\python.exe -m model.pipeline train `
  --model ridge `
  --horizon all `
  --run-id ridge_baseline
```

```powershell
.\.venv\Scripts\python.exe -m model.pipeline train `
  --model xgboost `
  --horizon all `
  --run-id xgboost_baseline
```

```powershell
.\.venv\Scripts\python.exe -m model.pipeline train `
  --model lightgbm `
  --horizon all `
  --run-id lightgbm_baseline
```

```powershell
.\.venv\Scripts\python.exe -m model.pipeline train `
  --model mlp `
  --horizon all `
  --run-id mlp_baseline
```

只训练 10 日模型时，把 `--horizon all` 改成 `--horizon 10`。例如：

```powershell
Set-Location "C:\Users\cbnb\Desktop\单因子测试流水线"
.\.venv\Scripts\python.exe -m model.pipeline --config .\model\config.toml train `
  --model lightgbm `
  --horizon 10 `
  --run-id lightgbm_10d_baseline
```

训练命令不会读取 2024-2025 作为模型选择依据，也不会使用 `entry_tradeable`
删除训练样本；当前输出主要用于 2023 validation 的开发期比较。

```powershell
.\.venv\Scripts\python.exe -m model.pipeline train `
  --model residual_rank_mlp `
  --horizon all `
  --run-id residual_rank_mlp_development
```

`--horizon` 可取 `1`、`5`、`10` 或 `all`。每个 `run_id` 只能使用一次；
目标目录已存在时命令会停止，不会覆盖已有实验。

训练命令只读取 2019-2023：

1. 在每天完整 1000 只股票上分别处理 17 个因子。
2. 无穷值转缺失，并保存 17 个缺失标记。
3. 使用 `median ± 3 × 1.4826 × MAD` 去极值。
4. 在当日非缺失股票间计算横截面 z-score，缺失值填 0。
5. 得到 17 个标准化因子和 17 个缺失标记，共 34 个特征。
6. Ridge 的 `StandardScaler` 只拟合 2019-2022 对应训练行；树模型不需要
   第二次标准化。
7. 2023 用于模型参数选择，主指标是每日横截面 Spearman Rank IC 的等权均值。

Rank IC 在 `1e-12` 内并列时先选择验证 RMSE 更低者，仍并列时选择更大的
lambda。配置中的 lambda 使用平均损失尺度，实际传给 sklearn 的
`alpha = lambda × n_train`。

XGBoost 使用 CPU `hist` 算法、固定随机种子和三组预先定义的树复杂度候选。
每组最多 300 轮，按 2023 RMSE 早停；Rank IC 距离最优值不超过 `0.0005`
时，选择 RMSE 更低者，再选择结构更简单的树。早停和参数选择都只使用
2023 validation。

LightGBM 使用确定性的 CPU `gbdt`，三组候选分别限制为 7、15、31 个叶子，
并同步提高叶节点最小样本数。其最大轮数、早停、Rank IC 容差和选择顺序
与 XGBoost 相同。

MLP 使用CPU版PyTorch和 `[64,32]`、`[128,64,32]` 两类网络容量，
第三组候选在后者基础上增加 Dropout 和 weight decay。输入的
`StandardScaler` 只拟合2019-2022；Target保持原始绝对收益，不进行
标准化。输出层偏置仅使用训练期Target均值初始化，损失和预测值始终保持
原始收益单位。

`residual_rank_mlp` 固定使用 `lambda=1` 的 Ridge 基准。它先按年份生成
2020、2021、2022 的扩展 OOF Ridge 预测，并用每个期限的真实退出日期
purge 边界；残差网络因此学习 `实际收益 - OOF Ridge预测`，而不是学习
Ridge 在自身训练样本上的拟合误差。

开发实验分为两段：

1. Stage C 比较三个残差 MLP 结构和 `gamma=[0,0.25,0.5,1]`。
2. 只有残差模型通过 Ridge 对照门槛，才进入 Stage D。
3. Stage D 固定结构和 gamma，从头比较
   `lambda_rank=[0,0.1,0.25,0.5]`。
4. 每个训练 batch 包含约 8 个完整交易日；收益损失使用当天全部股票，
   Pairwise 损失只在同日股票间抽样。
5. `lambda_rank=0` 是严格控制组，不构造 Pair，也不产生排序梯度。
6. 最终候选用 seed 42、43、44 检查稳定性，正式开发预测仍使用 seed 42，
   不做三模型 ensemble。

最终预测始终满足：

```text
final_pred = ridge_pred + gamma * residual_pred
```

主指标是2023日均Rank IC。与最高Rank IC相差不超过 `0.0005` 的候选，
依次按更低的十分组 MAE、更低 RMSE、更小 gamma/lambda 和更早 epoch
选择。若残差或排序损失没有通过门槛，运行仍会完整发布，并在 manifest
中记录拒绝原因和回退到 Ridge 或纯残差模型的决定。

该模型额外保存 `component_predictions.parquet`、每个期限的 Ridge 模型与
预处理器、残差 MLP 与其预处理器。发布前会重新加载全部组件并复算2023
预测，确认落盘模型可以恢复。

本命令不会读取模型字段中的 2024-2025 样本，不生成 test 预测，也不会使用
`entry_tradeable` 筛选训练样本。

### 5. 10日 Context 与收益分解实验

先从已有本地数据构建开发表：

```powershell
.\.venv\Scripts\python.exe -m model.pipeline prepare-context
```

该命令不访问米筐，只读取已有 `market_panel.csv`、模型数据集和样本索引，
输出 `context_dataset_10d.parquet` 和 `context_dataset_10d_schema.json`。
Context表最多构建到 `2023-12-31`，包括原有34个因子特征、T日市值与行业、
按每天当时成分股计算的市场状态，以及按每天当时行业成员计算的行业状态。

三个10日标签逐行满足：

```text
target_10d = market_target_10d + industry_target_10d + alpha_target_10d
```

训练完整实验：

```powershell
.\.venv\Scripts\python.exe -m model.pipeline train `
  --model context_decomposition `
  --horizon 10 `
  --run-id context_decomposition_10d
```

这套实验与前面的baseline不同：参数只由2020-2022扩展OOF选择，所有训练
样本的10日Target退出日必须早于对应预测年份。2023不用于lambda、树轮数、
网络结构或校准选择，只在候选冻结后统一评价一次。

程序先按Parquet过滤条件只读取到2022，完成全部OOF、阶段门控和候选冻结；
随后才读取2023。冻结参数在完整开发表上重新运行时必须复现完全相同的OOF
键和预测，复现误差会写入顶层manifest，之后才统一生成2023预测。

Stage 1固定比较 `context_ridge_10d`、`context_lightgbm_10d` 和
`decomposed_ridge_10d`。只有至少一个候选的raw OOF RMSE低于按相同折重建
的当前冠军，才训练两个固定Residual MLP。Residual MLP使用
`[128,64,32]`、Dropout `0.3`、weight decay `0.001`、六轮和seed 42，
不使用Pairwise Loss，也不标准化Target。

当前冠军的门控OOF使用嵌套残差口径：2021只使用2020年的Ridge OOF残差
训练残差网络，2022只使用2020-2021；2020因为没有更早的OOF残差，明确
退化为纯Ridge预测。候选还会逐年报告2020、2021、2022相对冠军的RMSE，
三个年份未保持同方向改善时不能通过验收。

每个候选子目录单独保存预测、模型、预处理器、校准器、特征schema、Context
数据诊断和manifest。预测表固定为：

```csv
date,stock_code,split,horizon,y_true,y_pred_raw,y_pred_calibrated
```

第一层验收指标是RMSE、MAE、R2、校准和预测波动；第二层才是Rank IC、
十分组误差、Top100平均真实收益与NDCG@100。与冠军的每日MSE差使用
Newey-West `lag=10`。训练与评价不使用 `entry_tradeable`。
10日Target每日重叠，因此只报告未年化Rank ICIR，不使用 `sqrt(252)` 放大。

注意：此前审计曾查看过2024测试期Target汇总，因此2024-2025只能作为带
披露的历史留出期，不能再称为完全未触碰测试；本命令仍不会读取该时期数据。

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
| `data/context_dataset_10d.parquet` | 2019-2023的10日Context特征与分解标签 |
| `data/context_dataset_10d_schema.json` | Context特征顺序、口径和输入哈希 |
| `runs/<run_id>/` | 模型、预处理器、验证预测、配置快照和运行清单 |

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
2023：baseline validation 和模型选择；Context实验只做冻结后评价
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
| Ridge 参数 | 使用新 `run_id` 重新执行 `train` |
| XGBoost 参数 | 使用新 `run_id` 重新执行 `train` |
| LightGBM 参数 | 使用新 `run_id` 重新执行 `train` |
| MLP 参数 | 使用新 `run_id` 重新执行 `train` |
| Residual Rank MLP 参数 | 使用新 `run_id` 重新执行 `train` |
| Context特征或分解口径 | `prepare-context`，再用新 `run_id` 训练 |
| Context模型参数 | 使用新 `run_id` 重新执行 `context_decomposition` |
| 评价指标 | 未来只重新 `evaluate` |
| Top N 或交易成本 | 未来只重新 `backtest` |

## 运行产物与后续接口

当前已经实现：

```powershell
.\.venv\Scripts\python.exe -m model.pipeline train `
  --model ridge `
  --horizon all `
  --run-id ridge_baseline
```

```powershell
.\.venv\Scripts\python.exe -m model.pipeline train `
  --model xgboost `
  --horizon all `
  --run-id xgboost_baseline
```

```powershell
.\.venv\Scripts\python.exe -m model.pipeline train `
  --model lightgbm `
  --horizon all `
  --run-id lightgbm_baseline
```

```powershell
.\.venv\Scripts\python.exe -m model.pipeline train `
  --model mlp `
  --horizon all `
  --run-id mlp_baseline
```

输出：

```text
model/runs/ridge_baseline/
├─ config_snapshot.toml
├─ manifest.json
├─ preprocessor_1d.joblib
├─ model_1d.joblib
├─ preprocessor_5d.joblib
├─ model_5d.joblib
├─ preprocessor_10d.joblib
├─ model_10d.joblib
└─ predictions.parquet
```

XGBoost 使用相同的目录协议，但模型文件为 `model_1d.json`、
`model_5d.json` 和 `model_10d.json`，不生成 `StandardScaler` 文件。
LightGBM 同样不生成 `StandardScaler`，模型文件使用对应期限的 `.txt`
原生格式。
MLP 使用 `model_1d.pt`、`model_5d.pt` 和 `model_10d.pt` 保存PyTorch
状态字典，并为每个期限保存只拟合训练集的 `preprocessor_*.joblib`。

预测表固定为：

```csv
date,stock_code,split,horizon,y_true,y_pred
```

当前 `split` 只会是 `validation`。`manifest.json` 记录数据和配置哈希、特征顺序、
预处理口径、全部候选参数的验证指标、最终选择、样本边界、环境版本和 Git 状态。

后续仍未实现：

```powershell
.\.venv\Scripts\python.exe -m model.pipeline evaluate --run-id ridge_baseline --split validation
.\.venv\Scripts\python.exe -m model.pipeline backtest --run-id ridge_baseline
```

约定：

- 一个 `run_id` 对应一种模型配置，可以包含 1、5、10 日三个独立模型。
- `evaluation` 根据 `run_id` 读取模型清单和预测结果，不重新训练。
- 开发阶段默认只评价 2023 validation。
- 模型选择完成后使用 2019-2023 重新训练，再生成 2024-2025 test 预测。
- 策略阶段才使用 T+1 `entry_tradeable`、交易成本、现金和无法成交状态。

## Git 和数据文件

`model/data` 和 `model/runs` 都是本地生成内容，不提交 GitHub。其他人复现时需要：

1. 生成配置中列出的各因子 `factor.csv`。
2. 生成或取得 `market_panel.csv` 和 `trading_calendar.csv`。
3. 运行 `doctor`。
4. 运行 `prepare-data`。
5. 研究Context/Decomposition时再运行 `prepare-context`。
