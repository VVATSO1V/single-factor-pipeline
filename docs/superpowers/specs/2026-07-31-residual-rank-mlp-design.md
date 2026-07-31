# Ridge 残差与排序感知 MLP 设计

日期：2026-07-31  
状态：已确认，待实施  
范围：`model/` 内的模型训练、冻结和评价流程

## 1. 目标

在不改变现有因子、收益标签和时间划分的前提下，新增独立模型
`residual_rank_mlp`，检验两项增量：

1. MLP 能否预测 Ridge 在样本外无法解释的非线性残差。
2. 分组距离加权的排序损失能否提高最终收益预测头的横截面排序能力。

最终策略只使用一个连续预测收益：

\[
\hat r_{i,t,h}
=
\hat r^{Ridge}_{i,t,h}
+
\gamma_h\hat e^{MLP}_{i,t,h}
\]

Top100 只按 \(\hat r\) 从高到低选择。十分组只用于训练辅助损失和评价，
不产生第二套实盘分数。

## 2. 非目标

本设计不包含：

- 修改17个因子的计算公式。
- 修改 T+1 开盘进入的 target 口径。
- 使用相对收益代替绝对收益。
- 将 target 标准化。
- 使用 `entry_tradeable` 过滤模型训练样本。
- 使用2024-2025数据调参、早停或重新选择模型。
- 在本阶段引入 Transformer、文本因子或新的股票属性特征。
- 覆盖或删除现有 Ridge、XGBoost、LightGBM 和 MLP baseline。

## 3. 实验结构

保留现有模型并新增以下消融实验：

| 实验 | 说明 |
|---|---|
| A | Ridge baseline，固定 `lambda=1` |
| B | 当前原始 MLP baseline |
| C | Ridge + OOF 残差 MLP，`lambda_rank=0` |
| D | Ridge + OOF 残差 MLP + Pairwise 排序损失 |
| E | 仅在 D 有效后研究 Soft Rank，不属于第一版实施 |

模型实现仍通过统一 `model.pipeline` 入口注册，不为每个期限或候选参数创建
独立 Python 文件。

## 4. 数据契约

### 4.1 特征

沿用当前34列派生特征：

- 17个因子的当日横截面 MAD 缩尾和 z-score 值。
- 17个因子缺失标记。
- 横截面处理后的缺失因子值填0。

禁止进入特征的字段：

- `stock_code`
- `target_*`
- `entry_*`
- `entry_tradeable`
- T+1 停牌、ST、涨跌停状态
- 2024-2025期间生成的统计量

每日横截面预处理只使用同一交易日的数据。每个时间 fold 的全局
`StandardScaler` 只能拟合该 fold 的训练数据。

### 4.2 Target

信号时点为 T 日收盘后，计划成交为 T+1 后复权开盘价：

\[
target_{1d}(T)
=
\frac{post\_open(T+2)}{post\_open(T+1)}-1
\]

\[
target_{5d}(T)
=
\frac{post\_open(T+6)}{post\_open(T+1)}-1
\]

\[
target_{10d}(T)
=
\frac{post\_open(T+11)}{post\_open(T+1)}-1
\]

模型始终输出原始绝对收益率，不对 target 做均值方差标准化。

## 5. 开发期 OOF Ridge

### 5.1 时间划分

开发期固定为：

- 2019-2022：训练和 OOF 残差生成。
- 2023：验证、早停、参数和模型选择。
- 2024-2025：完全锁定，不读取。

每个 horizon 独立生成 expanding OOF 预测：

| OOF预测年份 | Ridge训练年份 |
|---|---|
| 2020 | 2019 |
| 2021 | 2019-2020 |
| 2022 | 2019-2021 |

Ridge 固定使用：

```text
lambda = 1
alpha = lambda * 当前fold训练样本数
```

### 5.2 Purge

fold 训练样本必须同时满足：

```text
date < OOF预测年份开始日
exit_date_h < OOF预测年份开始日
```

已核对的最大训练信号日：

| OOF预测年 | 1日 | 5日 | 10日 |
|---|---|---|---|
| 2020 | 2019-12-27 | 2019-12-23 | 2019-12-16 |
| 2021 | 2020-12-29 | 2020-12-23 | 2020-12-16 |
| 2022 | 2021-12-29 | 2021-12-23 | 2021-12-16 |

每个 fold 的 Ridge `StandardScaler` 只拟合该 fold 训练特征，再变换 OOF
预测年份。禁止使用2019-2022整体拟合的 scaler 生成早期 OOF 预测。

### 5.3 OOF 残差

\[
e^{OOF}_{i,t,h}
=
r_{i,t,h}
-
\hat r^{Ridge,OOF}_{i,t,h}
\]

开发期残差 MLP 的样本为：

| 期限 | 日期数 | 行数 |
|---|---:|---:|
| 1日 | 726 | 726,000 |
| 5日 | 722 | 722,000 |
| 10日 | 717 | 717,000 |

2019只用于第一个 Ridge warm-up，不进入残差 MLP。

## 6. 真实十分组

十分组按同一天、同一 horizon 的真实 target 从高到低生成：

```text
第1组：未来收益最高
第10组：未来收益最低
```

使用平均排名处理并列收益：

\[
rank_i
=
rank(r_i,\ method=average,\ descending=True)
\]

\[
g_i
=
\min
\left(
10,
\left\lceil
10\frac{rank_i}{N_t}
\right\rceil
\right)
\]

完全相同的收益使用相同排名和组别，不通过 `stock_code` 人为打破并列。
因此每组不强制恰好100只。

十分组由实际最终收益生成，而不是由 Ridge 残差生成。分组标签是训练 target
的派生标签，不得进入模型特征。

## 7. 残差 MLP

MLP 接收与 baseline 相同的34个特征，输出残差修正：

\[
\hat e=MLP(x)
\]

最终预测：

\[
\hat r=\hat r^{Ridge}+\gamma\hat e
\]

输出层初始化：

```text
weight = 0
bias = OOF训练残差均值
```

训练初始状态应近似 Ridge。当 `gamma=0` 时，最终预测必须逐条等于 Ridge。

候选结构沿用当前 MLP：

```text
small       = [64, 32], dropout=0.1, weight_decay=0.0001
balanced    = [128, 64, 32], dropout=0.1, weight_decay=0.0001
regularized = [128, 64, 32], dropout=0.3, weight_decay=0.001
```

## 8. 收益损失

收益误差使用 Huber Loss。每个 horizon 的阈值只根据开发期 OOF 残差计算：

\[
\delta_h
=
1.5\times1.4826\times MAD(e^{OOF}_h)
\]

Ridge OOF 基准损失：

\[
C_h
=
mean
\left[
Huber_{\delta_h}
(\hat r^{Ridge,OOF},r)
\right]
\]

归一化收益损失：

\[
L_{return}
=
\frac{
mean[Huber_{\delta_h}(\hat r,r)]
}{C_h}
\]

该归一化只平衡损失尺度，不改变 target 和预测值的收益率单位。

## 9. 分组距离加权 Pairwise Loss

股票对只允许来自同一日期和同一 horizon。

\[
s_{ij}=sign(r_i-r_j)
\]

温度参数只根据训练期 target 计算：

\[
\tau_h
=
median_t
\left[
std_{cross-section}(r_{i,t,h})
\right]
\]

\[
d_{ij}
=
\frac{\hat r_i-\hat r_j}{\tau_h}
\]

\[
\ell_{ij}
=
\log(1+\exp(-s_{ij}d_{ij}))
\]

真实组距离权重：

\[
w^{distance}_{ij}
=
\frac{|g_i-g_j|}{9}
\]

Top组权重：

\[
v_i=
\begin{cases}
2,&g_i=1\\
1.5,&g_i=2\\
1,&其他
\end{cases}
\]

\[
w_{ij}
=
w^{distance}_{ij}
\frac{v_i+v_j}{2}
\]

每个日期内将有效 pair 权重归一化到均值1。Pairwise Loss 再除以
\(\log2\)，使随机排序附近的损失约为1：

\[
L_{pair}
=
\frac{mean(w_{ij}\ell_{ij})}{\log2}
\]

最终联合损失：

\[
L_{total}
=
L_{return}
+
\lambda_{rank}L_{pair}
\]

第一版固定候选：

```text
lambda_rank = [0, 0.1, 0.25, 0.5]
```

## 10. 日期 Batch 与 Pair 采样

当前随机行 batch 改为按完整日期组织：

```text
约8个完整日期 / batch
约1000只股票 / 日期
约8000行 / batch
```

收益损失可以在整个 batch 计算，Pairwise Loss 必须逐日期计算后再对日期
平均。

每只股票每个 epoch 抽取8个同日对手：

- 4个来自相邻真实组，学习局部和 Top100 边界排序。
- 4个来自距离较远的真实组，学习明显强弱关系。
- 同组 pair 跳过。
- 真实收益完全相同的 pair 跳过。
- 采样随机数由 `seed + epoch + date` 确定。
- 去除重复 pair。

第一版不搜索配对数量、Top组权重和温度公式。

## 11. 两阶段开发实验

### 11.1 阶段 C：纯残差

三个网络结构分别预测 OOF 残差，损失为 Huber residual loss。每个结构训练
一次，再在2023验证以下修正强度：

```text
gamma = [0, 0.25, 0.5, 1.0]
```

不允许 `gamma>1`。选择一个网络结构和 gamma。如果 `gamma=0` 最优，
停止后续排序损失实验并保留 Ridge。

### 11.2 阶段 D：排序辅助

固定阶段 C 选出的：

- 网络结构
- gamma
- Huber delta 公式
- pair temperature 公式
- pair 采样和权重

从头训练四个 `lambda_rank` 候选。`lambda_rank=0` 是严格控制组。

每个 horizon 的训练数量：

```text
阶段C：3个结构
阶段D：4个lambda
合计：7个模型
```

## 12. Checkpoint 与候选选择

每个 epoch 在完整2023 validation 上计算：

- 日均 Rank IC
- IC标准差、ICIR、正IC率
- 分组 MAE
- Top100召回率
- Top100真实平均收益
- RMSE、MAE、R2
- 预测值标准差

checkpoint 顺序：

1. 更高日均 Rank IC。
2. Rank IC 差异不超过0.0005时，选择更低分组 MAE。
3. 再选择更低 RMSE。
4. 再选择更早 epoch。

训练参数：

```text
max_epochs = 80
early_stopping_patience = 10
```

学习率调度使用 validation total loss，最佳 checkpoint 使用上述排序规则。

不同 `lambda_rank` 的最终选择顺序：

1. 最高日均 Rank IC。
2. 与最高值相差不超过0.0005的候选进入近优集合。
3. 近优集合中选择最低分组 MAE。
4. 再选择最低 RMSE。
5. 再选择较小 `lambda_rank`。
6. 再选择更早 epoch。

Top100收益和召回率只作为诊断，不作为主要选模指标。

## 13. 随机种子稳定性

参数筛选统一使用 `seed=42`。最终配置使用 `seed=[42,43,44]` 重跑并报告：

- Rank IC 均值和标准差。
- 分组 MAE 均值和标准差。
- Top100召回率均值和标准差。

正式开发预测仍使用 `seed=42`，不将三seed平均包装成额外 ensemble。若指标
只在单一 seed 有效，则拒绝该配置。

## 14. 接受与回退

残差 MLP 需要同时满足：

- `gamma>0`。
- Rank IC 高于 Ridge，或处于0.0005近优范围。
- 分组 MAE 低于 Ridge。
- 月度 IC 不集中在极少月份。
- 三seed方向一致。

排序损失需要同时满足：

- `lambda_rank>0`。
- Rank IC 高于 `lambda_rank=0` 控制组。
- 分组 MAE下降。
- Top100召回率不下降。
- RMSE没有显著恶化。
- 预测值标准差没有异常放大。

回退规则：

| 结果 | 决策 |
|---|---|
| `gamma=0` 最优 | 拒绝残差 MLP，保留 Ridge |
| 残差 MLP 不如 Ridge | 不进入阶段 D |
| `lambda_rank>0` 不如控制组 | 保留纯残差模型 |
| Rank IC 提高但分组 MAE 恶化 | 不接受 |
| 只有一个 seed 有效 | 判定不稳定 |
| 2024-2025失效 | 记录衰减，不回头调参 |

## 15. 最终重训与冻结

开发完成后冻结全部结构、gamma、lambda、损失、配对、训练和 epoch 参数。

最终 expanding OOF 增加2023：

| OOF预测年份 | Ridge训练年份 |
|---|---|
| 2020 | 2019 |
| 2021 | 2019-2020 |
| 2022 | 2019-2021 |
| 2023 | 2019-2022 |

最终残差训练规模：

| 期限 | OOF日期 | OOF行数 |
|---|---:|---:|
| 1日 | 968 | 968,000 |
| 5日 | 964 | 964,000 |
| 10日 | 959 | 959,000 |

最终 Ridge 使用 `final_train_h`：

| 期限 | 行数 | 日期数 | 最大信号日 |
|---|---:|---:|---|
| 1日 | 1,211,998 | 1,212 | 2023-12-27 |
| 5日 | 1,207,994 | 1,208 | 2023-12-21 |
| 10日 | 1,202,989 | 1,203 | 2023-12-14 |

最终重训重新计算训练期 scaler、OOF residual、Huber delta、基准损失和
temperature，但不能改变计算公式或候选选择结果。使用开发阶段选定的固定
epoch，不使用2024-2025早停。

## 16. 命令边界

### 16.1 开发选模

```powershell
.\.venv\Scripts\python.exe -m model.pipeline train `
  --model residual_rank_mlp `
  --horizon all `
  --run-id residual_rank_mlp_development
```

只允许读取2019-2023。

### 16.2 冻结重训

```powershell
.\.venv\Scripts\python.exe -m model.pipeline finalize `
  --source-run residual_rank_mlp_development `
  --run-id residual_rank_mlp_frozen
```

使用2019-2023重训，不读取2024-2025。

### 16.3 锁定测试

```powershell
.\.venv\Scripts\python.exe -m model.pipeline predict-test `
  --run-id residual_rank_mlp_frozen
```

只能加载冻结模型，不能训练或修改参数。

## 17. 产物

每个 horizon 保存：

```text
ridge_model_{h}d.joblib
ridge_preprocessor_{h}d.joblib
residual_mlp_{h}d.pt
mlp_preprocessor_{h}d.joblib
```

主预测表保持现有契约：

```csv
date,stock_code,split,horizon,y_true,y_pred
```

额外审计表：

```csv
date,stock_code,horizon,ridge_pred,residual_pred,gamma,final_pred
```

必须逐条满足：

\[
final\_pred
=
ridge\_pred
+
\gamma\times residual\_pred
\]

development manifest 记录数据 hash、特征顺序、OOF 边界、purge、scaler
范围、分组和 pair 规则、全部候选、三seed结果，并明确：

```text
test_data_read = false
uses_entry_tradeable = false
target_standardization = false
```

frozen manifest 记录来源 run、冻结配置、最终训练范围、artifact hash 和：

```text
status = frozen
test_evaluated = false
```

## 18. 锁定测试评价

2024、2025和2024-2025合计分别报告：

- 日均 Rank IC、IC标准差、ICIR、正IC率和 HAC t。
- 分组 MAE。
- Top100召回率和真实平均收益。
- MSE、RMSE、MAE、R2。
- 预测值标准差。
- 月度 IC。
- 十分组收益单调性。

统一比较 Ridge、原始 MLP、残差 MLP、残差排序 MLP、XGBoost 和
LightGBM。测试结果不得触发重新选模。测试后修改的模型不能继续把2024-2025
称为锁定测试。

## 19. 策略边界

模型评价完成后才进入独立策略阶段：

```text
预测Top100
T+1开盘执行
无法买入则跳过且不递补
无法卖出则继续持有
剩余资金保留现金
计算交易成本和换手率
```

ST、停牌、涨跌停和 `entry_tradeable` 只能在策略执行阶段使用。

## 20. 验收

数据验收：

- `(date,stock_code)` 唯一。
- target 与现有模型逐条一致。
- OOF训练早于 OOF预测。
- `exit_date` 不跨 fold。
- development 和 finalize 未读取2024-2025。
- `entry_tradeable` 未进入训练。

数学验收：

- `gamma=0` 时逐条等于 Ridge。
- `lambda_rank=0` 时不计算 Pairwise 梯度。
- pair 不跨日期。
- 同组和相同收益 pair 被排除。
- loss 和 gradient 全部有限。
- `final_pred` 加法逐条成立。

模型验收：

- 保存后重新加载预测一致。
- 相同 seed 完整重跑结果一致。
- 三seed方向一致。
- prediction 行数与样本索引一致。
- artifact hash 匹配。

实施过程中创建的临时测试脚本、测试 run 和缓存必须在验证完成后删除。
