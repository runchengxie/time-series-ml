# 策略方法论

## 预测目标

time-series-ml 是一个时序分类模型，预测 A 股个股次日涨跌方向。

- 标签定义：`target[t] = 1` 当 `(close[t+1] - close[t]) / close[t] >= 0.2%`，否则为 0
- 标签阈值：可通过 `--threshold` 调整，默认 0.002（20 bps）
- 频率：日频，每天收盘后更新预测

### 为什么不用回归

回归预测收益率数值比分类预测方向更难，且金融时序中极端收益会主导损失函数。分类问题用 accuracy / recall / F1 评估更直接，也更容易和交易信号对接（买入 = 预测上涨概率高）。

## 特征体系

特征工程在 `features.py` 中实现，全部基于价格和成交量计算，无外部数据源：

| 类别 | 特征 | 代码 |
|------|------|------|
| 价格动量 | SMA5 到 SMA20 差分 | `SMA5_diff` — `SMA20_diff` |
| 相对强弱 | RSI(14) | `RSI_14` |
| 趋势 | MACD 柱 | `MACD_hist` |
| 波动率 | 滚动波动率 5/10/20 日 | `volatility_5`, `_10`, `_20` |
| 蜡烛形态 | 实体比例、上下影线比例 | `body_ratio`, `upper_shadow`, `lower_shadow` |
| 成交量 | 量比、换手率 | `volume_ratio`, `turnover` |

所有特征使用滚动窗口计算，无未来信息泄露。特征间相关性在评估时自动预警（|r| > 0.8 的特征对）。

## 模型架构

### 主模型：XGBoost

```python
XGBClassifier(
    n_estimators=200,
    learning_rate=0.01,
    max_depth=3,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=1.0,
    reg_lambda=1.0,
    objective="binary:logistic",
)
```

参数选择原则：低学习率 + 浅树 + 强正则化 → 对抗金融数据的高信噪比。

### 多模型对比

`--compare-models` 会同时跑 XGBoost、LogisticRegression、RandomForest、LightGBM（可选），用 purged CV accuracy 排序，自动选用最优模型。

### 概率校准

`--calibrate` 启用 `CalibratedClassifierCV`（isotonic 或 sigmoid），将 XGBoost 原始概率校准为真实概率。校准后 `--prob-threshold 0.55` 的含义是「模型认为至少有 55% 把握时交易」。

### 行业中性化

`--neutralize-industry` 在截面上去除行业均值：对每个特征，每交易日减去该行业均值再除以行业标准差。消除行业 beta 后，模型学到的是行业内相对强弱而非行业轮动。

### 市场状态分类

`--regime` 计算等权市场代理的 MA20/MA60 趋势，将每个交易日标记为 Bull / Range / Bear。回测时按状态分拆统计，用于识别模型在不同市场环境下的表现差异。

## 回测方法论

### Walk-Forward

`backtest.py` 中的 `walk_forward()` 实现逐月重训回测：

1. 按 `retrain_freq="ME"`（月末）分割时间线
2. 每个周期：用 cycle 开始之前所有数据训练 → 对下一周期做预测 → 高置信度预测触发买入
3. 逐周期推进，t 时刻的模型只见过 t 之前的数据

### 信号生成

- 预测概率 >= `prob_threshold` → 生成买入信号
- 信号率 = 有信号的交易日数 / 总测试天数
- 默认 `prob_threshold=0.50`；校准后建议 0.55+

### TCA 假设

| 假设 | 说明 |
|------|------|
| 交易方向 | 只做多（无做空） |
| 交易成本 | 往返 `cost_bps` 基点，默认 5 bps（0.05%） |
| 成本组成 | 包含佣金 + 印花税 + 滑点 |
| 可交易性 | 当日收盘价成交，不模拟订单簿 |
| 涨跌停 | 未模拟涨跌停无法成交的情况 |
| 资金管理 | 等权 100 股 / 笔，无仓位动态调整 |

这些假设适用于研究阶段的策略评估。实际执行时由 `quant-execution-engine` 做更细粒度的成交模拟和仓位管理。

## 回测指标

| 指标 | 含义 | 合理区间 |
|------|------|----------|
| Total Return | 策略累计收益 | > 0 |
| Annual Return | 年化收益 | > 无风险利率 |
| Annual Vol | 年化波动率 | 结合 Sharpe 看 |
| Sharpe | 风险调整后收益 | > 0.5 可接受，> 1.0 较好 |
| Max Drawdown | 最大回撤 | < 30% |
| Win Rate | 胜率 | > 50% |
| Profit Factor | 盈亏比 | > 1.0 |
| Turnover/yr | 年换手次数 | 合理范围 20-200 |
| Signal Rate | 信号占比 | 过高可能过拟合，过低可能太保守 |
