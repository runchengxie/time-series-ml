# CLI 参考

本页提供 `ts-ml` 命令入口与全部参数的速查参考。工作流见 `docs/runbook.md`，配置语义见 `docs/config.md`。
目标读者：需要查阅命令和参数的开发者或研究员。
相关页面：`docs/runbook.md`、`docs/config.md`

## 快速决策

| 使用场景 | 对应命令 |
| --- | --- |
| 单标的快速验证 | `ts-ml --symbol 000001.SZ` |
| 单标的深度分析 | `ts-ml --symbol 000001.SZ --calibrate --optimize-threshold --compare-models --backtest --regime` |
| 多标的横向比较 | `ts-ml --symbols 000001.SZ,600519.SH --backtest` |
| 多标的 + 行业中性化 | `ts-ml --symbols 000001.SZ,600519.SH --neutralize-industry --backtest --regime` |
| 分边成本 + 涨跌停过滤 | `ts-ml --symbol 000001.SZ --backtest --backtest-buy-cost-bps 0 --backtest-sell-cost-bps 5` |
| 样本加权（近期数据更重要） | `ts-ml --symbol 000001.SZ --sample-weight-halflife 252` |
| 滚动训练窗口（只用近 2 年） | `ts-ml --symbol 000001.SZ --train-window-days 504` |
| 回归模式（预测收益率数值） | `ts-ml --symbol 000001.SZ --regression --backtest` |
| Final OOS 留出 | `ts-ml --symbol 000001.SZ --final-oos-size 0.10` |
| 标签置换检验 | `ts-ml --symbol 000001.SZ --permutation-test` |
| 特征族消融 | `ts-ml --symbol 000001.SZ --ablation` |
| 持久化结果 | `ts-ml --symbol 000001.SZ --backtest --save-results` |
| YAML 配置驱动 | `ts-ml --config my_run.yml` |

## 命令入口

```bash
ts-ml --help
```

## 全部参数

### 标的选择

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--symbol` | `000001.SZ` | 单标的 ts_code |
| `--symbols` | `""` | 多标的，逗号分隔（如 `000001.SZ,600000.SH`） |

### 日期范围

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--start-date` | `20150101` | 起始日期 YYYYMMDD |
| `--end-date` | `""`（今天） | 结束日期 YYYYMMDD |

### 数据

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--data-lake-root` | （Richard 本机路径） | parquet 数据湖根目录 |
| `--min-listed-days` | `252` | 最少上市天数，过滤新股 |
| `--min-daily-amount` | `0` | 最低日成交额（CNY），0=不启用。如 `1000000`=100 万 |

### 模型训练

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--threshold` | `0.002` | 上涨标签阈值（+0.2%） |
| `--test-size` | `0.2` | 测试集比例 |
| `--final-oos-size` | `0.0` | Final OOS 留出比例，0=不启用。如 `0.10`=最后 10% 完全隔离 |
| `--cv-splits` | `5` | Purged CV 折数 |
| `--purge-days` | `20` | Purge 天数（丢弃训练集末尾样本） |
| `--embargo-days` | `1` | Embargo 天数（训练测试边界间隔） |

### 样本加权与训练窗口

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--sample-weight-halflife` | `0` | 指数衰减半衰期（交易日），0=等权。如 `252`=一年内权重减半 |
| `--train-window-days` | `0` | 滚动训练窗口（交易日），0=全历史。如 `504`=仅用近 2 年 |

### 回归模式

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--regression` | `False` | 启用回归模式，预测 `future_return` 数值而非涨跌方向 |

### 概率校准

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--calibrate` | `False` | 启用 CalibratedClassifierCV（分类模式） |
| `--cv-method` | `isotonic` | 校准方法：`isotonic` 或 `sigmoid` |

### 信号过滤

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--prob-threshold` | `0.50` | 最低预测概率，低于此值不生成交易信号 |
| `--optimize-threshold` | `False` | 在验证集上 grid-search 最优分类阈值 |

### 模型对比

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--compare-models` | `False` | 同时跑 XGBoost/LR/RF/LightGBM/Ridge，自动选最优 |

### 特征消融

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--ablation` | `False` | 运行特征族消融（minus-one-family 实验） |

### 置换检验

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--permutation-test` | `False` | 标签置换检验（100 次打乱），判断模型是否在学噪音 |

### 回测

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--backtest` | `False` | 运行 walk-forward 回测 |
| `--backtest-cost-bps` | `5.0` | 往返交易成本（遗留参数） |
| `--backtest-buy-cost-bps` | `0.0` | 买入成本（bps），A 股印花税不在买入时收取 |
| `--backtest-sell-cost-bps` | `0.0` | 卖出成本（bps），0=从 `backtest-cost-bps/2` 推导 |
| `--no-price-limit-filter` | `False` | 禁用涨跌停可交易性过滤 |
| `--regime` | `False` | 按 Bull/Range/Bear 分拆回测统计 |

### 结果持久化

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--save-results` | `False` | 保存 summary.json 和 config.used.yml 到 artifacts/runs/ |
| `--artifacts-root` | `artifacts` | 产物根目录 |

## 参数优先级

CLI 参数 > YAML 配置 > Settings dataclass 默认值。

## 成本参数说明

三种成本参数方式（优先级从高到低）：

1. `--backtest-buy-cost-bps` + `--backtest-sell-cost-bps`：买入和卖出的明确成本
2. `--backtest-sell-cost-bps` 单独设置：买入=0, 卖出=指定值
3. 仅 `--backtest-cost-bps`：往返均分（buy=sell=cost_bps/2）

对于 A 股，推荐设置：`--backtest-sell-cost-bps 5`（印花税 5bps 仅在卖出时收取，买入成本可单独加佣金）。

## 回测输出

```text
-- Walk-Forward Backtest (buy=X.X sell=X.X bps, round-trip=XX.X bps) --
  Trades:          XXX
  ...
  Skipped (limit): XX     ← 涨跌停/流动性过滤跳过的信号数
```
