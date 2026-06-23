# 功能清单

## 数据处理

- A 股数据湖接入：从 market-data-platform parquet 数据湖读取个股日线，支持 5500+ 标的
- 环境变量配置：`TIME_SERIES_ML_DATA_LAKE_ROOT` + `TIME_SERIES_ML_INSTRUMENTS_PATH`，无需硬编码路径
- 申万行业分类：自动 join instruments 表，支持行业筛选和截面分析
- 日期范围过滤：`--start-date` / `--end-date` 灵活切片
- 多标的加载：单标的或逗号分隔多标的，跳过缺失文件
- 涨跌停可交易性标记：自动识别主板/创业板/科创板/北交所的涨跌停限制，标记 `is_tradable`
- 流动性过滤：`--min-daily-amount` 排除日成交额过低的日期

## 特征工程

全部自实现，零外部依赖（无 pandas_ta / ta-lib）。共 30 个特征，分为 10 个特征族：

| 特征族 | 特征 | 说明 |
| --- | --- | --- |
| 动量 | SMA5/10/20/60 差分 | 多周期均线变化率 |
| 价格距离 | price_dist_5d/20d/60d | 收盘价距 SMA 的归一化距离 |
| 相对强弱 | RSI_14, MACD_hist, BB_position | Wilder RSI、MACD 柱、布林带位置 |
| 成交量 | Volume_SMA5/20_ratio, Volume_trend, vol | 量比与趋势 |
| 波动率 | ATR_14_pct, HistVol_20/60 | 归一化波幅与年化波动率 |
| 蜡烛形态 | body_ratio, upper/lower_shadow_ratio | 实体与影线比例 |
| 跳空 | gap_pct, open_gap | 昨日跳空幅度与今日开盘跳空 |
| 收益分布 | ret_skew_20d, ret_kurt_20d | 20 日收益偏度与峰度 |
| 换手 | turnover_rate_f, volume_ratio_raw | 换手率与量比原始值 |
| 估值 | log_pe, log_pb, log_mcap | PE/PB/市值对数化 |

## 模型

- XGBoost：主模型，低学习率 + 浅树 + 强正则化
- 多模型对比（`--compare-models`）：XGBoost / LR / RF / LightGBM / Ridge，自动选最优
- Ridge sanity baseline：线性模型基线，判断 XGBoost 的非线性增量是否真实
- 回归模式（`--regression`）：用 XGBoostRegressor 直接预测 `future_return` 数值
- 样本加权（`--sample-weight-halflife`）：指数衰减，近期数据权重更高
- 滚动训练窗口（`--train-window-days`）：只用最近 N 个交易日训练
- 概率校准（`--calibrate`）：CalibratedClassifierCV，isotonic 或 sigmoid
- 行业中性化（`--neutralize-industry`）：截面去行业均值，消除行业 beta
- Triple Barrier 标签：AFML 三向标签（profit_take / stop_loss / timeout）
- 元标签（`--meta-labeling`）：次级模型过滤主模型预测
- 滞后特征（`--use-lag-features`）：t-3 和 t-5 滞后特征，用于序列依赖测试
- YAML 配置驱动（`--config`）：所有参数可配置文件管理

## 验证体系

### 时序交叉验证

- PurgedTimeSeriesSplit：标准时序 CV + purge + embargo 三层保护
- Purge（`--purge-days`，默认 20）：丢弃训练集末尾，防特征窗口泄漏
- Embargo（`--embargo-days`，默认 1）：训练/测试边界间隔

### 样本外验证

- Final OOS 留出段（`--final-oos-size`）：保留最后一段样本完全不参与训练和调参
- 三层切分：train / validation / final OOS

### 评估指标

- 分类指标：Accuracy / Precision / Recall / F1 / ROC AUC
- Baseline 对比：Majority（多数类）和 Persistence（昨日方向）
- 过拟合诊断：`train_acc - test_acc`，自动 [OK] / [WARN] / [FAIL]
- 混淆矩阵 + 分类报告
- 回归指标（`--regression`）：RMSE / R² / IC

### IC 分析

- Rank IC：预测概率 vs 实际收益的 Spearman 相关系数
- ICIR：IC 均值 / IC 标准差，衡量信号稳定性
- 因子 IC：每个特征对 future_return 的 Rank IC
- 因子相关性：|r| > 0.8 自动预警
- 滚动 IC 窗口：6 个月和 12 个月滚动 IC 均值序列

### 防过拟合诊断

- 标签置换检验（`--permutation-test`）：100 次标签打乱，判断真实 accuracy 是否显著高于噪音分布
- 特征族消融（`--ablation`）：minus-one-family 实验，输出每族特征的边际贡献
- Ridge 基线检查：如果线性模型完全无信号但 XGBoost 有，可能是噪音过拟合

## 回测

- Walk-Forward（`--backtest`）：逐月重训，t 时刻模型只见过 t 之前数据
- 买卖分边成本：`--backtest-buy-cost-bps` / `--backtest-sell-cost-bps`，A 股印花税卖出单向收取
- 涨跌停过滤：自动识别并跳过涨跌停无法成交的信号
- 流动性过滤：`--min-daily-amount` 排除低流动性日期
- 信号过滤（`--prob-threshold`）：低置信度预测不交易
- 阈值优化（`--optimize-threshold`）：验证集 grid-search 最优分类阈值
- 市场状态分拆（`--regime`）：Bull / Range / Bear 三维回测统计

### 回测指标

Total Return / Annual Return / Annual Vol / Sharpe / Max Drawdown / Win Rate / Profit Factor / Turnover / Signal Rate / Skipped (limit)

## 工程化

- MLflow 实验追踪：自动记录参数和指标，本地 file store，零侵入
- 运行结果持久化（`--save-results`）：保存 summary.json + config.used.yml 到 artifacts/runs/
- 多标的汇总 CSV（`--symbols` 模式）：一行一标的指标对比表
- GitHub Actions CI：push/PR 触发 ruff + pyright + pytest
- 35 个测试：覆盖 labels / features / split / data / industry / crossval / backtest / triple_barrier
- 类型检查：pyright basic 模式 0 error
- Lint：ruff 0 error
- uv 包管理：lock 文件锁定版本，可复现
