# 未来优化方向

按优先级从高到低排列。已完成的标记 [x]。

## P0：验证加强

- [x] Purge 实现：`crossval.py` — `PurgedTimeSeriesSplit`
- [x] Embargo 实现：`crossval.py` — `embargo_days` 参数
- [x] Walk-forward 回测：`backtest.py` — `walk_forward()` 逐月重训 + TCA
- [x] TCA 实际成本：`backtest.py` — 往返成本 `cost_bps` 参数

## P1：模型改进

- [x] 阈值优化：`cli.py --optimize-threshold`
- [x] 特征扩展：波动率特征、蜡烛形态特征
- [x] 多模型对比：`cli.py --compare-models`
- [x] 多标的：`cli.py --symbols`，A 股数据湖 5774 只
- [x] 概率校准：`model.py — CalibratedClassifierCV`，`cli.py --calibrate`

## P2：执行与风控

- [x] 信号过滤：`cli.py --prob-threshold`，低置信度信号不交易
- [ ] 仓位管理：交由 quant-execution-engine 处理（不在研究 repo 重复造）
- [ ] 止损/止盈：交由 quant-execution-engine 处理

## P3：工程化

- [x] 配置文件：`config_yaml.py` — YAML 配置文件支持
- [ ] 实验追踪：MLflow 记录参数和指标
- [ ] CI/CD：GitHub Actions 自动化测试 + 模型验证

## P4：量化研究

- [x] 因子 IC 分析：`metrics.py — compute_factor_ic()`
- [x] 因子相关性矩阵：`metrics.py — compute_factor_correlation()`
- [x] IC 衰减曲线：`metrics.py — compute_ic_decay()`
- [ ] 行业中性：申万行业分类 + 截面中性化
- [ ] 市场状态分类：牛/熊/震荡市分别建模

## 测试覆盖缺口

当前测试覆盖 labels、features、split、cache。以下模块无测试：

- `model.py`（train_model、compare_models、calibrate_model）
- `metrics.py`（evaluate、compute_ic_icir、compute_factor_ic、compute_factor_correlation、compute_ic_decay）
- `crossval.py`（PurgedTimeSeriesSplit、purged_cross_val_score）
- `backtest.py`（walk_forward）

## 已知局限

1. 多标的横向比较：每个标的是独立跑流程，未做联合建模（面板模型）
2. 日频：无法捕捉日内模式
3. 无宏观变量：利率、资金面等未纳入
4. 幸存者偏差：当前 parquet 只包含至今仍在交易的股票
5. 未用复权价格：涨跌停判断需要不复权价格，但回测收益应用复权价
