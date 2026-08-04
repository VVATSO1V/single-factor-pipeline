# 10日绝对收益 Context 与 Decomposition 模型设计

## 1. 研究目标

本阶段只研究模型预测能力，不进行策略回测，不设计换仓、持仓、交易成本、
可交易性过滤或资金管理。

第一目标是在每个信号日 T，为当日中证1000的每只股票预测未来10日绝对收益：

```text
target_10d(T) = post_open(T+11) / post_open(T+1) - 1
```

模型输出仍为原始收益率单位。主要选模指标依次为：

1. RMSE。
2. MAE。
3. 校准截距、校准斜率和预测值标准差。
4. Rank IC、ICIR和正Rank IC比例。
5. 十分组MAE、Top100平均真实收益和NDCG@100。

如果所有候选都不能稳定降低具体收益预测误差，则正式接受“当前数据不足以
稳定预测每只股票具体收益”的结论，再把研究目标降级为横截面排序和Top100
识别。不得因为回归结果不理想而无限增加模型复杂度。

## 2. 范围与非目标

本阶段：

- 只验证10日目标。
- 只使用当前已经落地的本地数据。
- 不新增米筐字段或外部数据源。
- 不读取2024-2025进行训练、参数选择、校准或开发评价。
- 不使用 `entry_tradeable` 或T+1状态筛选训练样本。
- 不修改1日和5日模型。
- 不训练Transformer。
- 不扩大现有Pairwise Loss参数搜索。
- 不做策略回测。

只有10日收益分解模型通过RMSE验收后，才讨论复制到5日和1日。

## 3. 当前冠军基准

当前10日开发冠军为 `residual_rank_mlp_development`：

```text
模型：Ridge + Residual MLP
Ridge lambda：1
MLP hidden_layers：[128, 64, 32]
dropout：0.3
weight_decay：0.001
gamma：1
lambda_rank：0
selected epoch：6
```

2023验证指标：

```text
RMSE：0.0752478258
MAE：0.0514607906
R2：0.0001945421
Rank IC：0.0916357765
预测值标准差：0.0072671172
```

新模型必须首先与该冠军比较RMSE，而不是只与零预测、历史均值或较弱模型
比较。

## 4. 已知测试集污染说明

在本设计的数据审计阶段，有一次Target极值审计未先限制日期，意外输出了
2024测试期的汇总和个别极值。没有使用这些信息修改代码、模型或候选参数，
本设计中的全部数据规则只依据2019-2023重新统计。

但按最严格定义，2024-2025已经被查看过，后续报告不得再称其为“完全未触碰
的锁定测试”。后续可以继续作为固定规则下的历史留出评价，但必须保留这项
污染披露；真正未观察的评价应依赖后续前向数据。

## 5. 实际数据审计

`model/data/market_panel.csv` 的实际审计结果：

```text
总行数：3,758,188
交易日：1,699
历史相关股票：2,212
重复(date, stock_code)：0
in_universe行数：1,699,000
每日in_universe股票数：固定1,000
```

2019-2023开发数据的主要问题：

| 问题 | 实际结果 |
|---|---:|
| 成分股行业缺失 | 0行 |
| 每日行业数 | 29-30 |
| 少于10只股票的日期行业组 | 约21% |
| 市值缺失 | 1,431行，主要集中在2只股票 |
| 成分股开收盘价缺失 | 0行 |
| 10日Target缺失 | 11行，均来自 `000043.XSHE` |
| 20日窗口不可用 | 2019年前20个交易日 |
| 非正或缺失涨跌停价 | 1,250行 |
| 10日收益范围 | 约-70%至+369% |

高缺失因子包括：

```text
sue_latest：约61.2%
eps_revision_breadth_60d：约39.6%
eps_revision_60d：约38.3%
cash_earnings_quality：约16.8%
```

第一版不从这些高缺失因子构造市场或行业聚合特征。

## 6. 收益标签分解

股票 i 在日期 t 的10日收益记为：

```text
y_i,t = target_10d(i, t)
```

当日等权市场收益：

```text
m_t = mean_i(y_i,t)
```

行业 j 的原始超额收益：

```text
g_raw_j,t = mean(y_i,t | industry(i,t)=j) - m_t
```

由于约21%的日期行业组少于10只股票，行业标签向市场收益收缩：

```text
shrink_j,t = N_j,t / (N_j,t + 10)
g_j,t = shrink_j,t * g_raw_j,t
```

个股Alpha标签：

```text
a_i,t = y_i,t - m_t - g_industry(i,t),t
```

因此每条有效样本必须严格满足：

```text
y_i,t = m_t + g_industry(i,t),t + a_i,t
```

等权市场标签与“预测每只股票、每只股票在RMSE中等权”的目标一致。它不是
官方中证1000指数收益，也不声称复制官方指数权重。

## 7. 数据异常处理

### 7.1 行业缺失

当前开发数据没有行业缺失，但流水线必须具备确定性回退：

- 使用T日当天的行业标签。
- 不使用未来标签回填。
- 不把最新行业标签覆盖到历史。
- 缺失行业归入 `UNKNOWN`。
- 增加 `industry_missing=1`。
- `UNKNOWN` 的行业收益预测固定为0。
- Context模型的行业数值特征填0并保留缺失标记。
- `UNKNOWN` 股票的Alpha标签为 `y_i,t - m_t`。

不进行历史行业前向填充，避免在真实行业调整后继续使用过期分类。

### 7.2 行业分类变化

- 每个日期使用该日期的点时行业分类。
- 行业日序列使用每一天当时的行业成员计算。
- 不使用今天的行业成员重新计算历史行业收益。
- 股票行业发生变化后，从生效日开始使用新行业。
- 行业类别使用One-Hot，验证期未知类别按 `UNKNOWN` 处理。
- 禁止Target Encoding。

### 7.3 市值缺失

个股市值先转换为：

```text
log_market_cap = log1p(market_cap)
```

缺失时：

- 填充当日中证1000横截面中位数。
- 增加 `market_cap_missing=1`。
- 不长期前向填充。

市值加权市场或行业特征：

- 排除缺失权重后重新归一化。
- 保存 `market_cap_coverage`。
- 覆盖不足95%时，该加权特征标记缺失并回退到等权特征。

### 7.4 滚动窗口不足

市场面板从2019-01-02开始，因此早期1、5、20日上下文不可用。主键不删除：

- 数值填0。
- 为每个窗口增加缺失/可用标记。
- Context、Decomposition和控制模型使用相同监督样本键。

### 7.5 Target缺失

- 不填0。
- 不填市场或行业均值。
- 不参与监督训练、RMSE、MAE、Rank IC。
- 不参与对应日期的市场或行业未来标签均值。
- 原始主表可以保留该行。
- 每个日期记录有效Target数量。

### 7.6 极端Target

- 验证Target不裁剪。
- 不删除涨跌超过50%或100%的样本。
- 正式RMSE、MAE和R2使用原始Target。
- 神经网络训练使用Huber降低极端梯度影响。
- 额外报告 `abs(target)<=50%` 的稳健RMSE。
- 报告极端样本对总MSE的贡献比例。

### 7.7 涨跌停与停牌

- 非正或缺失的涨跌停价格视为不可用，不视为真实0元价格。
- 涨跌停比例只在有效涨跌停数据上计算。
- 增加 `limit_data_missing_rate`。
- 停牌股票继续属于当日股票池。
- 停牌期间形成的0价格收益保留。
- 市场和行业环境增加停牌率。

### 7.8 现有因子缺失

- 不删除股票。
- 沿用现有横截面中性值填充和17个独立缺失标记。
- 第一版不计算基本面因子的市场或行业均值。

## 8. Context 特征

市场层特征：

- 成分股等权过去1、5、20日收益。
- 成分股市值加权过去1、5、20日收益。
- 上涨股票比例。
- 横截面收益标准差。
- 收益10%、50%、90%分位数。
- 涨停、跌停、停牌和ST比例。
- 市值均值、中位数和离散度。
- 市值有效覆盖率和涨跌停数据有效覆盖率。

行业层特征：

- 行业过去1、5、20日等权收益。
- 行业相对市场过去1、5、20日收益。
- 行业内上涨股票比例。
- 行业内收益标准差和分位差。
- 行业股票数量。
- 行业市值中位数。
- 行业在中证1000中的市值占比。
- 行业停牌率和数据覆盖率。

个股层输入：

- 现有17个因子。
- 17个因子缺失标记。
- `log_market_cap` 和市值缺失标记。
- 行业One-Hot和行业缺失标记。
- 当日市场环境。
- 当日所属行业环境。

所有特征只使用T日及以前数据。滚动市场序列按每天当时的历史成分股计算，
禁止使用T日的1000只成分股回算过去20日市场状态。

具体计算口径：

```text
r_i,t = post_close_i,t / post_close_i,t-1 - 1
r_mkt_equal,t = mean_i(r_i,t | i属于t日股票池)
r_mkt_cap,t = sum_i(weight_i,t * r_i,t)
R_mkt_k,t = product_s(1 + r_mkt,s) - 1, s=t-k+1...t
```

行业日收益同样先使用每个历史日期当时的行业成员计算，再复合成1、5、20日
收益；行业超额收益为同窗口行业复合收益减市场复合收益。上涨比例、收益分位
数、离散度、涨跌停率、停牌率和ST率均为T日横截面状态。涨停命中定义为
`raw_close >= limit_up - 1e-8`，跌停命中定义为
`raw_close <= limit_down + 1e-8`，分母只包含对应限价有效的股票。

Ridge和MLP的连续特征使用训练折拟合的标准化器；LightGBM使用原始连续
特征。缺失填充值、标准化参数和One-Hot类别字典均只能从当前训练折获得，
随后原样应用于OOF年份和2023。

## 9. 候选实验

### 9.1 阶段一：结构筛选

| 模型 | 目的 |
|---|---|
| `context_ridge_10d` | 检查Context的线性增量 |
| `context_lightgbm_10d` | 检查Context的非线性交互 |
| `decomposed_ridge_10d` | 检查显式分解本身 |

Context Ridge只允许：

```text
lambda = [0.1, 1, 10]
alpha = lambda * n_train
```

每个lambda生成完整的2020-2022扩展OOF预测，按合并后的原始预测OOF RMSE
选择。RMSE相同至 `1e-8` 时选择更大的lambda，以较强正则作为确定性破同分
规则。

Context LightGBM固定当前10日最佳树结构：

```text
num_leaves = 31
max_depth = 5
min_data_in_leaf = 1000
```

迭代轮数只允许：

```text
num_boost_round = [10, 20, 40, 80, 120, 200]
```

每个轮数生成完整的2020-2022扩展OOF预测，按合并后的原始预测OOF RMSE
选择；RMSE相同至 `1e-8` 时选择更少轮数。不重新搜索树结构，也不使用2023
Early Stopping。

Decomposed Ridge分别训练：

```text
market_ridge
industry_ridge
alpha_ridge
```

三个子模型只允许 `lambda=[0.1,1,10]`。行业模型按行业股票数加权。
三个组件分别按各自标签的合并OOF RMSE选择lambda，不枚举27种联合组合；
最终仍同时报告三个组件相加后的股票级OOF RMSE。

### 9.2 阶段二：固定非线性增强

只有阶段一至少一个候选的2020-2022原始OOF RMSE低于按相同折重新生成的
当前冠军OOF RMSE，才训练：

| 模型 | 结构 |
|---|---|
| `context_residual_mlp_10d` | Context Ridge + Residual MLP |
| `decomposed_residual_mlp_10d` | 市场Ridge + 行业Ridge + Alpha Residual MLP |

MLP配置固定为：

```text
hidden_layers = [128, 64, 32]
dropout = 0.3
weight_decay = 0.001
gamma = 1
lambda_rank = 0
epochs = 6
seed = 42
```

不重新搜索结构、gamma、epoch或Pairwise lambda。

Huber delta只用对应训练折计算。Residual MLP沿用当前实现：先计算训练折
Ridge残差的MAD，再使用：

```text
delta = 1.5 * 1.4826 * MAD(ridge_residual)
```

## 10. 时间切分与OOF

候选参数和校准只使用扩展OOF。当前冠军保留其已经锁定的特征与参数，但必须
按同一折和同一有效样本键重新生成OOF预测，作为阶段门控基准：

| OOF预测年份 | 允许训练数据 |
|---|---|
| 2020 | 2019且Target退出日早于2020-01-01 |
| 2021 | 2019-2020且Target退出日早于2021-01-01 |
| 2022 | 2019-2021且Target退出日早于2022-01-01 |

使用2020-2022合并后的原始OOF RMSE选择lambda和LightGBM轮数。阶段一OOF
门控完成后，符合条件时再生成阶段二OOF预测。全部候选结构在读取2023前
冻结。随后：

```text
2019-2022重新训练全部已冻结候选
一次性生成并评价全部候选的2023预测
```

2023不得用于拟合scaler、缺失填充值、lambda、轮数或校准参数。

## 11. 预测校准

使用2020-2022 OOF预测拟合：

```text
y_hat_calibrated = intercept + slope * y_hat_raw
```

截距和斜率使用合并OOF样本的普通最小二乘拟合，每条股票日期样本等权。

规则：

- 同时保存raw和calibrated预测。
- 模型参数先按raw OOF RMSE选择，再拟合校准器；不得用同一批OOF校准后
  RMSE反向选择模型参数。
- OOF校准后指标只作描述，不作为阶段门控；2023校准指标才是独立评价。
- `slope>0` 时，该候选的预先锁定官方预测为calibrated预测。
- `slope<=0` 时拒绝校准、记录方向异常，并预先锁定raw预测为官方预测。
- 不得观察2023后在raw与calibrated之间择优。
- 校准后仍输出原始收益率单位。
- 正斜率校准不改变横截面排序。
- Decomposition只校准最终相加预测，不分别重标三个组件。
- 2023只应用固定校准器，不重新拟合。

## 12. 损失函数

- Ridge使用平方损失，对应条件均值和RMSE目标。
- LightGBM使用L2回归损失。
- Residual MLP使用训练期确定delta的Huber Loss。
- Target不标准化。
- 可以使用训练期常数归一化损失尺度，但输出单位不变。
- 本阶段不加入Pairwise或Top100分类损失。

## 13. 评价与统计检验

### 13.1 第一层：绝对收益准确度

- MSE、RMSE、MAE、R2。
- 相对当前冠军的MSE Skill。
- 校准截距和斜率。
- 预测值标准差与真实收益标准差。
- 上涨方向准确率。
- 原始和校准预测分别报告。
- 验收比较使用OOF阶段预先锁定的官方预测版本。

```text
Skill = 1 - MSE_candidate / MSE_current_champion
```

### 13.2 第二层：横截面能力

- 日均Rank IC、IC标准差、ICIR和正IC率。
- 十分组MAE。
- Top100平均真实收益。
- NDCG@100。

### 13.3 重叠Target统计

每天计算：

```text
MSE_t = mean_i((y_hat_i,t - y_i,t)^2)
d_t = MSE_candidate,t - MSE_champion,t
```

对 `d_t` 使用Newey-West/HAC、`lag=10`，报告：

- 平均每日MSE差。
- HAC t值和95%置信区间。
- 候选每日MSE更低的日期比例。
- 月度RMSE差异。

## 14. 验收门槛

强接受：

- 2023 RMSE低于当前冠军。
- MSE Skill大于0。
- MAE不恶化。
- HAC置信区间上界小于0。
- 改善不只来自极少数极端样本。
- 预测值没有被压成近似常数。

暂定改善：

- RMSE和MAE改善。
- 但HAC置信区间包含0。

拒绝：

- RMSE没有改善。
- 改善只来自少数极端样本。
- OOF校准斜率非正的候选不能强接受，但保留raw预测作诊断。
- 预测值接近常数。
- 结果不能在OOF年份保持同方向。

Rank IC和Top100指标是第二层诊断，不得覆盖第一层RMSE结论。

## 15. 产物契约

所有新增工作必须留在 `model/`。候选run至少保存：

```text
manifest.json
predictions.parquet
component_predictions.parquet（Decomposition）
feature_schema.json
context_feature_summary.json
calibrator.joblib
各子模型和预处理器
```

`predictions.parquet`：

```csv
date,stock_code,split,horizon,y_true,y_pred_raw,y_pred_calibrated
```

Decomposition组件表：

```csv
date,stock_code,market_pred,industry_pred,alpha_pred,final_pred_raw,final_pred_calibrated
```

必须逐行验证：

```text
final_pred_raw = market_pred + industry_pred + alpha_pred
```

manifest记录数据hash、特征顺序、异常覆盖率、OOF边界、参数选择、校准器、
全部指标、HAC检验、组件恢复检查和测试集污染披露。

## 16. 实施顺序

1. 构建并审计本地市场和行业Context表。
2. 构建三层标签并验证逐行重构恒等式。
3. 生成统一10日训练接口和扩展OOF切分。
4. 训练Context Ridge、Context LightGBM和Decomposed Ridge。
5. 重新生成当前冠军OOF并运行阶段一OOF门控。
6. 只有阶段一通过才生成两种固定Residual MLP的OOF预测。
7. 冻结全部候选，使用2019-2022重训并一次性评价2023。
8. 保存并重新加载全部产物，对2023做一致性复算，不进行第二轮选模。
9. 输出与当前冠军的完整比较和HAC检验。
10. 不读取2024-2025进行开发决策。
