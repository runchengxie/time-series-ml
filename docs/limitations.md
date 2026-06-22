# 测试覆盖缺口

当前测试覆盖 `labels`、`features`、`split`、`cache`、`industry`。以下模块无单元测试，修改时需格外小心：

| 模块 | 未覆盖的函数 |
|------|-------------|
| `model.py` | `train_model`、`compare_models`、`calibrate_model` |
| `metrics.py` | `evaluate`、`compute_ic_icir`、`compute_factor_ic`、`compute_factor_correlation`、`compute_ic_decay` |
| `crossval.py` | `PurgedTimeSeriesSplit`、`purged_cross_val_score` |
| `backtest.py` | `walk_forward` |

这些模块通过 CLI 端到端运行间接验证，但缺乏独立的单元测试。添加测试的优先级：`crossval.py` > `metrics.py` > `model.py` > `backtest.py`。

# 已知局限

## 策略层面

1. **多标的横向比较**：每个标的是独立跑流程，未做联合建模（面板模型）。这意味着模型无法学习跨标的相对强弱关系。

2. **日频限制**：仅使用日线数据，无法捕捉日内模式（如开盘跳空、盘中反转）。

3. **无宏观变量**：利率、资金面、市场情绪等宏观因子未纳入特征体系。

4. **幸存者偏差**：当前 parquet 数据只包含至今仍在交易的股票，已退市标的被排除。回测结果可能高估策略表现。

5. **未用复权价格**：涨跌停判断需要不复权价格，但回测收益计算应用复权价。当前实现未做此区分。

## TCA 层面

6. **仅做多**：无做空逻辑，熊市只能空仓等待。

7. **简化成交假设**：默认当日收盘价成交，未模拟订单簿、滑点、涨跌停无法成交等情况。

8. **固定仓位**：等权 100 股 / 笔，无仓位管理和资金曲线动态调整。实际执行由 `quant-execution-engine` 负责。

## 数据层面

9. **数据湖路径硬编码**：默认 `data_lake_root` 指向 Richard 本机路径，其他用户需手动指定 `--data-lake-root`。

10. **无实时数据**：依赖 market-data-platform 定期更新 parquet，无 streaming 或实时行情接入。
