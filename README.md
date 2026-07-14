# 单因子测试流水线

这是一套标准化的 A 股横截面单因子测试流水线。使用者只需要准备两张标准 CSV：

1. `factor.csv`：待测因子值
2. `market_panel.csv`：股票池、行情、交易状态、行业和市值暴露

主程序会自动完成预处理、T/T+1 对齐、IC、十分组、多空净值、换手率和年度表现统计。

## 项目结构

```text
单因子测试流水线/
├─ single_factor_pipeline.py
├─ README.md
├─ .env.example
└─ reverse_20d/
   ├─ build_market_panel.py
   ├─ build_factor.py
   ├─ data/
   │  ├─ market_panel.csv
   │  └─ factor.csv
   └─ output/
      ├─ summary.csv
      ├─ ic_series.csv
      ├─ quantile_daily_returns.csv
      ├─ quantile_summary.csv
      ├─ long_short_nav.csv
      ├─ turnover.csv
      ├─ yearly_performance.csv
      ├─ data_quality.csv
      ├─ report.html
      ├─ figures/
      └─ run_config.json
```

`reverse_20d` 是一个 20 日反转因子的完整示例。以后测试新因子时，可以复制这个文件夹，改成新的因子名，再重写其中的 `build_factor.py`。

## 环境

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install pandas numpy rqdatac
```

也可以直接安装项目依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

根目录 `.env` 用来保存米筐账号密码，`.env.example` 是可以上传的空模板：

```text
RQDATA_USER=
RQDATA_PASSWORD=
```

本地运行前先复制一份 `.env.example` 为 `.env`：

```powershell
Copy-Item .env.example .env
```

然后打开 `.env`，把米筐账号密码填进去。`.env` 已经被 `.gitignore` 忽略，上传 GitHub 前仍建议再检查一遍不要包含真实密钥。

## 输入文件

### 1. 待测因子 `factor.csv`

```csv
date,stock_code,factor_value
2025-01-02,000012.XSHE,0.352
2025-01-02,000019.XSHE,-1.126
2025-01-03,000012.XSHE,0.417
```

含义：

| 字段 | 含义 |
|---|---|
| `date` | 因子截面日期 T |
| `stock_code` | 米筐股票代码 |
| `factor_value` | T 日可观测到的因子值 |

也支持宽表输入：

```csv
date,000012.XSHE,000019.XSHE,600123.XSHG
2025-01-02,0.352,-1.126,0.441
2025-01-03,0.417,-0.932,0.382
```

### 2. 市场面板 `market_panel.csv`

```csv
date,stock_code,in_universe,post_close,raw_open,raw_close,limit_up,limit_down,is_st,is_suspended,listing_days,industry,market_cap
2025-01-02,000012.XSHE,true,12.53,8.31,8.42,9.26,7.58,false,false,2150,建材,25600000000
```

含义：

| 字段 | 含义 |
|---|---|
| `in_universe` | T 日是否属于目标股票池，例如当期中证1000成分股 |
| `post_close` | 后复权收盘价，用来计算未来收益 |
| `raw_open` | 未复权开盘价，用来判断 T+1 开盘是否涨跌停 |
| `raw_close` | 未复权收盘价，保留用于检查 |
| `limit_up` / `limit_down` | 当日未复权涨跌停价 |
| `is_st` | 当日是否 ST |
| `is_suspended` | 当日是否停牌 |
| `listing_days` | 上市至当日的交易日数量 |
| `industry` | 行业分类，用于行业中性化 |
| `market_cap` | 总市值，用于市值中性化 |

`market_panel.csv` 的唯一键是 `(date, stock_code)`。建议覆盖“历史成分股并集股票 × 全部交易日”，而不是只保留股票在指数内的日期；否则股票在 T 日属于股票池、T+1 被调出时，T 日未来收益会缺少 T+1 价格。股票池必须使用历史成分股快照，不能用当前成分股回填历史。

## 核心对齐规则

因子值使用 T 日截面，收益从 T 日之后开始算：

```text
return_1d(T)  = post_close(T+1)  / post_close(T) - 1
return_5d(T)  = post_close(T+5)  / post_close(T) - 1
return_10d(T) = post_close(T+10) / post_close(T) - 1
```

交易过滤检查 T+1 的状态：

```text
T 日在股票池
T 日因子值有效
T+1 非 ST
T+1 非停牌
T+1 上市满 N 个交易日
T+1 开盘未涨停
T+1 开盘未跌停
```

默认 `N=120`，可用 `--min-listing-days` 修改。

## 预处理

每天在有效股票截面内执行：

```text
MAD 去极值
Z-score 标准化
行业 + log(market_cap) 中性化
中性化残差再次 Z-score
```

输出两个版本：

| 版本 | 含义 |
|---|---|
| `raw` | 去极值并标准化后的原始因子 |
| `neutralized` | 对行业和市值回归后的残差因子 |

中性化回归形式：

```text
factor_raw_i = a + b * log(market_cap_i) + 行业哑变量_i + residual_i
factor_neutralized_i = zscore(residual_i)
```

## 输出逻辑

### 1. IC 序列、IC 均值、ICIR

每天计算：

```text
Pearson IC = Corr(factor, future_return)
Rank IC    = Corr(rank(factor), rank(future_return))
ICIR       = IC均值 / IC标准差
```

输出文件：

| 文件 | 内容 |
|---|---|
| `ic_series.csv` | 每日 1/5/10 日 IC |
| `summary.csv` | IC 均值、ICIR、IC 为正比例等汇总 |

### 2. 十分组收益与单调性

每天按因子值从低到高分成 G1 到 G10，计算各组未来平均收益。

```text
monotonicity = SpearmanCorr(组号, 各组长期平均收益)
top_bottom_spread = G10平均收益 - G1平均收益
```

输出：

| 文件 | 内容 |
|---|---|
| `quantile_daily_returns.csv` | 每天每组平均收益 |
| `quantile_summary.csv` | 各组长期表现、单调性、头尾差 |

### 3. 多头、空头和多空组合净值

多空净值只使用 1 日收益，避免把重叠的 5 日、10 日收益逐日复利。

正向因子默认：

```text
Long = G10
Short = G1
LongShort = G10 - G1
```

如果因子方向相反，运行时设置 `--direction negative`。

输出：

| 文件 | 内容 |
|---|---|
| `long_short_nav.csv` | 多头净值、空头净值、多空净值、每日收益 |

注意：这里是不含交易成本、冲击成本、滑点和融券约束的诊断型净值。

### 4. 换手率

```text
factor_rank_turnover = 1 - 相邻两日因子排名相关性
long_turnover        = 1 - 相邻两日头部组合重合比例
short_turnover       = 1 - 相邻两日尾部组合重合比例
```

输出：

| 文件 | 内容 |
|---|---|
| `turnover.csv` | 因子排序换手、头部组合换手、尾部组合换手 |

### 5. 分年度表现与衰减

按自然年统计 IC、多空收益、波动、夏普、最大回撤、胜率和换手。

输出：

| 文件 | 内容 |
|---|---|
| `yearly_performance.csv` | 分年度表现，用于观察因子是否衰减 |

如果运行主流水线时添加 `--report`，程序还会生成：

| 文件 | 内容 |
|---|---|
| `report.html` | 自动汇总 IC、分组收益、多空净值、换手率和年度表现的 HTML 报告 |
| `figures/` | 报告中使用的图表，包括 IC、累计 IC、分组收益、多空净值、换手率和年度表现 |

## 运行 20 日反转示例

### 第一步：填写 `.env`

```powershell
Copy-Item .env.example .env
```

然后编辑 `.env`：

```text
RQDATA_USER=你的米筐账号
RQDATA_PASSWORD=你的米筐密码
```

### 第二步：下载市场面板

默认股票池是中证1000：

```powershell
.\.venv\Scripts\python.exe .\reverse_20d\build_market_panel.py `
  --start-date 2019-01-01 `
  --end-date 2025-12-31 `
  --index-code 000852.XSHG
```

如果要换成中证500：

```powershell
.\.venv\Scripts\python.exe .\reverse_20d\build_market_panel.py `
  --start-date 2019-01-01 `
  --end-date 2025-12-31 `
  --index-code 000905.XSHG
```

生成：

```text
reverse_20d/data/market_panel.csv
```

### 第三步：计算 20 日反转因子

```powershell
.\.venv\Scripts\python.exe .\reverse_20d\build_factor.py --lookback 20
```

生成：

```text
reverse_20d/data/factor.csv
```

### 第四步：运行主流水线

```powershell
.\.venv\Scripts\python.exe .\single_factor_pipeline.py `
  --factor-name reverse_20d `
  --factor-path .\reverse_20d\data\factor.csv `
  --market-panel-path .\reverse_20d\data\market_panel.csv `
  --output-dir .\reverse_20d\output `
  --return-windows 1,5,10 `
  --quantiles 10 `
  --min-listing-days 120 `
  --mad-width 3 `
  --direction positive `
  --report
```

如果测试新的因子，需要把 `--factor-name`、`--factor-path`、`--market-panel-path` 和 `--output-dir` 改成新因子文件夹对应的路径。

## 新增一个因子的推荐方式

1. 复制 `reverse_20d/` 文件夹，改名为你的因子名，例如 `volatility_20d/`
2. 保留或调整 `build_market_panel.py`
3. 重写 `build_factor.py`，输出同样格式的 `data/factor.csv`
4. 运行 `single_factor_pipeline.py`，把参数指向新文件夹的数据和输出目录

这样每个因子的输入、因子计算代码和输出结果都放在自己的文件夹里，后续整理和复现会比较清楚。

## 当前范围

当前版本暂不包含：

```text
与已有因子库的相关性
交易成本扣减
分钟级成交模拟
多因子合成
组合优化
机器学习预测
```
