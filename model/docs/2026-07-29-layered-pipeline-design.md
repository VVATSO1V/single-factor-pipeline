# Model 分层流水线设计

## 1. 目标

将 `model` 目录整理为一套简洁、可复现、可继续扩展的多因子建模流水线：

- 使用者只需要理解 `README.md`、`config.toml` 和统一入口 `pipeline.py`。
- 因子清单、日期、标签、时间划分、模型及策略参数集中配置。
- 数据、预处理、训练、评价和策略职责分离。
- 不为每个模型或每个预测期限创建独立训练脚本。
- 每次训练保存配置、数据指纹、模型、预测和评价结果。
- 第一轮重构不得改变已有数据口径、公式、字段或时间边界。

## 2. 范围边界

`model` 流水线从各单因子目录已经生成的 `factor.csv` 开始。单因子构建脚本仍由各因子目录维护，`model` 不直接调用这些脚本。

米筐市场数据获取是显式在线阶段。默认本地完整运行不得自动重新下载市场数据，避免误耗账号额度。

本次重构包含：

1. 统一命令入口。
2. 集中配置。
3. 现有五个数据构建脚本的内部模块化。
4. 为未来预处理、训练、评价和策略阶段保留稳定接口。
5. 更新 `model/README.md`。

本次重构不包含实际 Ridge、XGBoost、LightGBM、神经网络训练，也不生成新的回测结果。

## 3. 目标目录

```text
model/
├─ __init__.py
├─ README.md
├─ config.toml
├─ pipeline.py
├─ stages/
│  ├─ __init__.py
│  ├─ market.py
│  ├─ features.py
│  ├─ labels.py
│  ├─ dataset.py
│  ├─ preprocessing.py
│  ├─ training.py
│  ├─ evaluation.py
│  └─ strategy.py
├─ data/
│  ├─ source/
│  └─ prepared/
├─ runs/
└─ docs/
```

第一轮仅创建已经具备实际功能的模块。尚未实现的预处理、训练、评价和策略文件不创建空壳；它们在对应阶段开始时再增加。

## 4. 现有代码迁移

| 当前文件 | 目标模块 | 职责 |
|---|---|---|
| `build_market_panel.py` | `stages/market.py` | 米筐初始化、交易日历、股票池、行情、状态、行业和市值 |
| `build_factor_table.py` | `stages/features.py` | 校验并合并配置中的因子长表 |
| `build_target.py` | `stages/labels.py` | 构造 T+1 开盘进入的 1、5、10 日绝对收益 |
| `build_model_dataset.py` | `stages/dataset.py` | 合并特征、标签、T 日信息和 T+1 执行审计字段 |
| `build_sample_index.py` | `stages/dataset.py` | 按信号日与退出日构造 train、validation、test 和 final_train |

`build_model_dataset.py` 与 `build_sample_index.py` 合并到同一个职责层，但保留独立函数，避免构建主表时被迫同时重建样本索引。

迁移后删除旧的五个顶层可执行脚本，避免同一逻辑存在两个入口。删除前必须完成新旧产物逐字段回归检查。

## 5. 集中配置

`config.toml` 是参数唯一来源，至少包含：

- 项目名称和中证 1000 指数代码。
- 数据起止日期。
- 米筐环境文件路径。
- 17 个因子文件路径。
- 标签期限及价格口径。
- train、validation、test 和 final_train 日期。
- 最低上市天数与涨跌停判断口径。
- 数据目录和运行目录。

米筐账号密码继续保存在项目根目录 `.env`，不得写入配置或运行记录。

路径默认相对 `model/config.toml` 解析，保证项目移动到其他电脑后仍可使用。

新增或删除因子只修改配置，不修改 Python 代码。

## 6. 统一入口

第一轮提供以下命令：

```powershell
python -m model.pipeline doctor
python -m model.pipeline fetch-market
python -m model.pipeline prepare-data
```

命令职责：

- `doctor`：检查 Python 依赖、配置、因子路径、输入字段及已有产物的一致性，不写数据。
- `fetch-market`：显式调用米筐，生成市场面板和官方交易日历。
- `prepare-data`：按依赖顺序生成因子宽表、标签、建模主表和样本索引。

`prepare-data` 默认复用已有市场面板，不得隐式调用米筐。缺少市场面板时应停止并提示先运行 `fetch-market`。

未来阶段增加：

```powershell
python -m model.pipeline train --model ridge --horizon all
python -m model.pipeline evaluate --run-id <run_id> --split validation
python -m model.pipeline backtest --run-id <run_id>
python -m model.pipeline run-all --model ridge
```

`run-all` 只运行本地阶段，不包含 `fetch-market`。

## 7. 数据产物

为降低第一轮重构风险，现有文件名和格式暂时保持不变：

```text
model/data/market_panel.csv
model/data/trading_calendar.csv
model/data/factor_wide.csv
model/data/target.csv
model/data/model_dataset.parquet
model/data/model_dataset_schema.json
model/data/model_sample_index.parquet
model/data/model_split_summary.json
```

`data/source` 和 `data/prepared` 的物理迁移放在后续独立变更中进行。只有在所有调用方均改用集中配置后才能迁移，避免同时改变代码组织和数据路径。

## 8. 模型注册与运行

一个 `run_id` 对应一种模型配置，但可以包含 1、5、10 日多个期限。

统一模型注册表根据模型名称创建实例。调用时只运行明确指定的组合：

```text
--model ridge --horizon 1
--model ridge --horizon all
--model lightgbm --horizon 5
```

三个期限使用相同训练流程，但分别拟合模型、预处理器和目标，不共享监督标签。

简单模型可以由 `training.py` 统一管理。神经网络实现明显变复杂时，可以迁入内部 `models` 子包，但外部命令和注册接口保持不变。

## 9. 预测与评价接口

训练输出统一使用长表：

```csv
date,stock_code,split,horizon,y_true,y_pred
```

`evaluation.py` 根据 `run_id` 读取运行清单与预测表，不重新训练模型，也不要求再次指定模型名称。

开发阶段默认只评价 2023 validation。模型和参数确定后，使用 2019 至 2023 的 `final_train` 重训，才允许生成并评价 2024 至 2025 test 预测。

## 10. 运行记录

每次训练写入：

```text
model/runs/<run_id>/
├─ config_snapshot.toml
├─ manifest.json
├─ preprocessor_<horizon>.joblib
├─ model_<horizon>.joblib
├─ predictions.parquet
├─ evaluation/
└─ backtest/
```

`manifest.json` 至少记录：

- 模型名称和参数。
- 预测期限。
- 特征白名单。
- 数据文件哈希。
- 配置哈希。
- Git commit。
- Python 和主要依赖版本。
- 训练、验证和测试日期。
- 运行状态。

## 11. 数据和未来函数约束

重构后继续强制：

1. T 日特征只能使用 T 日及以前可见数据。
2. 标签从 T+1 开盘开始。
3. 时间划分使用真实退出日 purge 边界。
4. `entry_*` 字段不能进入模型特征。
5. `entry_tradeable` 不筛选模型训练样本，只在策略阶段使用。
6. 验证和测试不能拟合缺失值、标准化或模型参数。
7. 2024 至 2025 锁定测试在模型选择完成前不得读取。

## 12. 验收标准

第一轮重构完成必须满足：

1. `doctor` 能检查配置、依赖、因子输入和已有数据。
2. `fetch-market` 与原市场面板命令参数和输出兼容。
3. `prepare-data` 能从已有市场面板生成全部后续产物。
4. 新旧 `factor_wide.csv`、`target.csv`、主表、schema、样本索引和汇总在排序统一后逐字段一致。
5. 行数、主键唯一性、日期边界、标签有效数和分割样本数与当前结果一致。
6. 运行临时测试文件和缓存清理后再交付。
7. `README.md` 只保留统一入口、配置说明、数据口径、产物说明和常见错误。

## 13. 实施顺序

1. 增加配置与统一入口。
2. 将现有逻辑迁移到 `stages`，不修改计算口径。
3. 通过统一入口生成临时输出，与现有产物对比。
4. 修复所有差异后移除旧入口。
5. 更新 README。
6. 执行语法、配置、数据契约和全流程回归检查。
7. 删除临时测试产物。
