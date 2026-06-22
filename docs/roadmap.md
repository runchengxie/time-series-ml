# 项目路线图（已完成）

按优先级从高到低排列。全部标记 [x]。

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
- [x] 仓位管理：已委托 quant-execution-engine（研究 repo 不负责任意映射）
- [x] 止损/止盈：已委托 quant-execution-engine

## P3：工程化

- [x] 配置文件：`config_yaml.py` — YAML 配置文件支持
- [x] 实验追踪：`tracking.py` — MLflow 本地追踪（local file store，零侵入）
- [x] CI/CD：GitHub Actions（lint + test，不做模型验证）

## P4：量化研究

- [x] 因子 IC 分析：`metrics.py — compute_factor_ic()`
- [x] 因子相关性矩阵：`metrics.py — compute_factor_correlation()`
- [x] IC 衰减曲线：`metrics.py — compute_ic_decay()`
- [x] 行业中性：申万行业分类 + 截面中性化
- [x] 市场状态分类：`regime.py` — MA20/MA60 + 偏离阈值，回测按状态分拆统计
