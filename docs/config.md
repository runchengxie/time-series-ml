# 配置参考

本页提供 YAML 配置键与默认行为的速查参考。
范围：只说明配置项和模板示例；策略语义见 `docs/methodology.md`。
适合读者：需要通过 YAML 文件管理实验参数的研究员。
相关页面：`docs/cli.md`、`docs/methodology.md`

## 最小 YAML 示例

```yaml
# my_run.yml
symbol: 000001.SZ
start_date: "20200101"
calibrate: true
prob_threshold: 0.55
backtest: true
regime: true
```

```bash
ts-ml --config my_run.yml
```

## 完整配置块速查

| 配置块 | 键 | 默认值 | 说明 |
| --- | --- | --- | --- |
| 标的 | `symbol` | `"000001.SZ"` | 单标的 |
| 标的 | `symbols` | — | 多标的列表 |
| 日期 | `start_date` | `"20150101"` | YYYYMMDD |
| 日期 | `end_date` | `""` | 留空=今天 |
| 数据 | `min_listed_days` | `252` | 最少上市天数 |
| 数据 | `min_daily_amount` | `0.0` | 最低日成交额（CNY），0=不启用 |
| 标签 | `threshold` | `0.002` | 上涨标签阈值 |
| 训练 | `test_size` | `0.2` | 测试集比例 |
| 训练 | `final_oos_size` | `0.0` | Final OOS 留出比例，0=不启用 |
| 训练 | `cv_splits` | `5` | CV 折数 |
| 训练 | `purge_days` | `20` | Purge 天数 |
| 训练 | `embargo_days` | `1` | Embargo 天数 |
| 加权 | `sample_weight_halflife` | `0` | 指数衰减半衰期，0=等权 |
| 窗口 | `train_window_days` | `0` | 滚动训练窗口，0=全历史 |
| 模式 | `regression` | `false` | 回归模式（预测收益率数值） |
| 校准 | `calibrate` | `false` | 概率校准（仅分类模式） |
| 校准 | `cv_method` | `"isotonic"` | 校准方法 |
| 信号 | `prob_threshold` | `0.50` | 信号过滤阈值 |
| 对比 | `compare_models` | `false` | 多模型对比 |
| 消融 | `ablation` | `false` | 特征族消融实验 |
| 置换检验 | `permutation_test` | `false` | 标签置换检验 |
| 回测 | `backtest` | `false` | 运行回测 |
| 回测 | `backtest_cost_bps` | `5.0` | 往返成本（遗留） |
| 回测 | `backtest_buy_cost_bps` | `0.0` | 买入成本 |
| 回测 | `backtest_sell_cost_bps` | `0.0` | 卖出成本（0=从 cost_bps/2 推导） |
| 回测 | `backtest_enforce_price_limit` | `true` | 涨跌停过滤 |
| 回测 | `regime` | `false` | 市场状态分拆 |

## 完整模板

```yaml
# research_config.yml — 完整实验配置模板
symbol: "000001.SZ"
start_date: "20150101"
end_date: ""

# 数据过滤
min_listed_days: 252
min_daily_amount: 0.0

# 标签
threshold: 0.002

# 训练
test_size: 0.2
final_oos_size: 0.0
cv_splits: 5
purge_days: 20
embargo_days: 1

# 样本加权（0=等权，252=一年半衰期）
sample_weight_halflife: 0

# 训练窗口（0=全历史，504=近2年）
train_window_days: 0

# 回归模式
regression: false

# 校准
calibrate: true
cv_method: "isotonic"

# 信号
prob_threshold: 0.55
optimize_threshold: false
compare_models: false

# 特征证据
ablation: false
permutation_test: false

# 回测
backtest: true
backtest_buy_cost_bps: 0.0
backtest_sell_cost_bps: 5.0
backtest_enforce_price_limit: true
regime: true

# XGBoost 超参数覆盖
xgb_params:
  n_estimators: 200
  max_depth: 3
```

## 成本参数逻辑

1. 如果 `backtest_sell_cost_bps == 0` 且 `backtest_cost_bps > 0`：sell = cost_bps/2, buy = cost_bps/2
2. 如果设置了 `backtest_sell_cost_bps` 或 `backtest_buy_cost_bps`：优先使用
3. A 股推荐：`backtest_buy_cost_bps: 0, backtest_sell_cost_bps: 5`

## 参数优先级

```
CLI 参数 > YAML 配置 > Settings dataclass 默认值
```
