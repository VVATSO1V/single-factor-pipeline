# Ridge Baseline 设计

## 1. 目标

在现有分层数据流水线之上增加可复现的 Ridge baseline：

- 使用 T 日收盘后可见的 17 个因子。
- 分别预测 T+1 开盘进入后的 1、5、10 日绝对收益。
- 只用 2019-2022 训练，用 2023 选择正则化强度。
- 不读取、不评价 2024-2025 锁定测试结果。
- 保存预处理器、模型、验证集预测和完整运行清单。

## 2. 新增文件

```text
model/stages/preprocessing.py
model/stages/training.py
```

修改：

```text
model/config.toml
model/pipeline.py
model/README.md
```

不创建单独的 `train_ridge_1d.py`、`train_ridge_5d.py` 或
`train_ridge_10d.py`。

## 3. 依赖

新增 `scikit-learn`。其依赖 `joblib` 用于保存预处理器和模型。

不自行实现 Ridge 数值求解。

## 4. 样本与特征边界

输入：

```text
model/data/model_dataset.parquet
model/data/model_dataset_schema.json
model/data/model_sample_index.parquet
model/data/model_split_summary.json
```

只从 schema 的 `feature_columns` 读取 17 个因子。禁止读取：

- `industry`
- `market_cap`
- `stock_code` 作为数值或类别特征
- `target_*` 作为特征
- 任意 `entry_*` 字段

每个期限分别使用：

```text
split_<horizon> == train       -> 训练
split_<horizon> == validation  -> 验证
```

`entry_tradeable` 不参与样本筛选。

## 5. 每日横截面预处理

预处理必须先在每个 T 日完整的 1000 只成分股上完成，再根据对应期限的
split 筛选训练和验证样本。不得先使用未来标签是否有效过滤股票后再计算
横截面统计量。

每个日期、每个因子按以下顺序处理：

1. 正负无穷转为缺失。
2. 记录原始缺失标记 `<factor>__missing`。
3. 在当日非缺失股票中计算中位数和 MAD。
4. 使用 `median ± 3 × 1.4826 × MAD` 去极值。
5. 对去极值后的当日非缺失值计算均值和总体标准差。
6. 标准差有效且大于 0 时计算 z-score。
7. 标准差为 0 或有效值不足 2 个时，非缺失值设为 0。
8. 原始缺失位置的 z-score 填 0。

MAD 为 0 或不可用时不裁剪；当日因子全部缺失时，标准化因子全部为 0，
缺失标记全部为 1。

该步骤只使用 T 日当时可见的完整股票横截面，不使用未来日期信息。

横截面处理后共有 34 个派生特征：

```text
17 个标准化因子
17 个缺失标记
```

为了让 Ridge 对缺失标记和标准化因子使用可比的惩罚尺度，再使用仅在对应
期限训练样本上拟合的 `StandardScaler`。验证样本只能调用训练期已经拟合
的 scaler。

## 6. 内存处理

不得为 1、5、10 日分别重复计算完整横截面变换。

一次 `train --horizon all`：

1. 只读取 2019-2023 的主键、17 个因子和三个目标。
2. 按日期块进行横截面处理，限制中间 DataFrame 大小。
3. 结果使用 `float32` 保存于内存。
4. 三个期限复用同一份横截面处理结果。
5. 每个期限再根据自己的 split 拟合 scaler 和 Ridge。

不生成长期保留的额外特征宽表。

## 7. Ridge 与参数选择

Ridge 目标保持原始绝对收益，不做排名或标准化。

为使正则化强度不依赖训练样本数，配置使用平均损失口径的 `lambda`：

```text
lambda_grid = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
```

传给 scikit-learn 的参数为：

```text
alpha = lambda × n_train
```

每个期限、每个 lambda：

1. 仅使用 2019-2022 训练。
2. 预测 2023 validation。
3. 按日期计算验证集横截面 Spearman Rank IC。
4. 使用所有有效日期的 Rank IC 均值作为主选择指标。
5. Rank IC 均值在 `1e-12` 内并列时，选择 RMSE 更低者。
6. 仍并列时，选择更大的 lambda，作为更保守的模型。

每天在至少有 2 个有效预测和真实收益、且两者排名均非常数时才计算
Rank IC。每个有效日期等权；RMSE 使用全部验证行。Ridge 使用确定性的
`cholesky` 求解器。

训练阶段只计算参数选择所需的 Rank IC 和 RMSE。完整评价指标留给后续
`evaluation.py`。

## 8. 命令

```powershell
.\.venv\Scripts\python.exe -m model.pipeline train `
  --model ridge `
  --horizon all `
  --run-id ridge_baseline
```

支持：

```text
--horizon 1
--horizon 5
--horizon 10
--horizon all
```

当前 `--model` 只接受 `ridge`。后续模型通过同一注册接口增加。

`run_id` 必须是安全目录名。目标运行目录已经存在时默认停止，防止覆盖既有
实验。

## 9. 运行产物

```text
model/runs/<run_id>/
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

选择单一期限时只保存对应文件。

预测表统一使用：

```csv
date,stock_code,split,horizon,y_true,y_pred
```

本阶段的 `split` 只能是 `validation`。

`manifest.json` 至少记录：

- 运行状态与创建时间。
- 模型名称。
- 期限。
- 原始特征和 34 个派生特征。
- MAD 宽度。
- lambda 网格、候选指标和最终选择。
- 训练、验证样本数和日期范围。
- 数据文件及 schema 哈希。
- 配置文件哈希。
- Git commit。
- Python、pandas、NumPy 和 scikit-learn 版本。

整个运行先写入 `runs` 下的临时目录，全部期限成功后再发布为正式
`run_id`，避免留下半成品。

## 10. 锁定测试

本阶段不得：

- 使用 2024-2025 样本拟合 scaler。
- 使用 2024-2025 样本选择 lambda。
- 生成 2024-2025 预测。
- 输出 test 指标。

模型选定后的 2019-2023 final_train 和 2024-2025 静态测试将在后续独立
命令中实现。

## 11. 验收

实施完成必须验证：

1. 每个日期横截面处理前仍有完整 1000 行。
2. 预处理不读取目标或 `entry_*` 决定横截面统计量。
3. 训练和验证主键与相应 split 完全一致。
4. 三个期限不会互用 target 或 split。
5. scaler 只在训练样本拟合。
6. 2024-2025 不出现在训练读取范围、预测表和 manifest。
7. 固定输入重复运行得到相同 lambda、系数和预测。
8. 运行目录已有时不会覆盖。
9. 失败运行不会留下正式 run 目录。
10. 临时测试文件和缓存清理后交付。
