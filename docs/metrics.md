# 指标与结果解读

本页说明一次 `ts-ml` 运行结束后，常见评估指标分别代表什么、该先看哪些指标，以及哪些指标容易被误读。
本页不展开字段级技术契约；字段和参数见 `docs/cli.md`。
适合谁：需要判断模型结果、解释回测表现、排查异常 run 的研究员或开发者。
相关页面：`docs/methodology.md`、`docs/validation.md`、`docs/cli.md`

## CLI 输出的结构

一次 `ts-ml` 运行在终端输出以下区块（按顺序）：

1. `[features] Features built.` — 特征工程完成
2. `[split] Train: N rows, Test: M rows` — 时序切分
3. `[train] Training XGBoost ...` — 训练日志
4. `[eval] Evaluating ...` — 评估开始
5. `EVALUATION SUMMARY` — 核心指标总览
6. 自动诊断 `[OK] / [WARN] / [FAIL]` — 过拟合和 IC 诊断
7. `Baselines` — Majority 和 Persistence 对比
8. `Test Class Distribution` — 测试集涨跌分布
9. `Classification Report` — sklearn 分类报告
10. `Confusion Matrix` — 混淆矩阵
11. `Feature Importance` — XGBoost 特征重要度
12. `Factor IC Analysis` — 单因子 Rank IC
13. `High Factor Correlations` — 高相关特征预警（如有）
14. `Walk-Forward Backtest` — 回测结果（如启用）
15. `Per-Regime Performance` — 市场状态分拆（如启用）

## 先看哪些指标

优先按这个顺序看：

| 顺序 | 指标 | 位置 | 判断标准 |
| --- | --- | --- | --- |
| 1 | Test Accuracy vs Majority | Baselines 块 | test_acc > majority_acc |
| 2 | Overfitting Gap | EVALUATION SUMMARY | < 0.05 优秀，< 0.10 可接受 |
| 3 | Rank IC | EVALUATION SUMMARY | > 0.05 有信号，> 0 最低要求 |
| 4 | ICIR | EVALUATION SUMMARY | > 1.0 稳定，> 0.3 可接受 |
| 5 | CV Accuracy std | [train] 块 | < 0.05 |
| 6 | Sharpe（如有回测） | Walk-Forward Backtest | > 0.5 可接受，> 1.0 较好 |
| 7 | Signal Rate（如有回测） | Walk-Forward Backtest | 1-5% 正常区间 |

## 分类指标

### Accuracy、Precision、Recall、F1

| 指标 | 含义 | 什么时候更重要 |
| --- | --- | --- |
| Accuracy | 预测正确的比例 | 类别平衡时 |
| Precision | 预测为「涨」中真正涨的比例 | 减少假买入信号 |
| Recall | 真正涨中被预测出的比例 | 不想错过上涨机会 |
| F1 | Precision 和 Recall 的调和平均 | 综合衡量 |

### ROC AUC

预测概率排序能力的综合指标。> 0.5 表示比随机好，> 0.55 有一定区分能力。

常见误解：ROC AUC 只看排序不关心阈值，所以一个 AUC=0.55 但 accuracy=0.52 的模型，可能换个阈值就有更好的 accuracy。

### 混淆矩阵

```
           Pred Down   Pred Up
  True Down    TN          FP
  True Up      FN          TP
```

- TN（True Negative）：正确预测「不涨」
- FP（False Positive）：错误预测「涨」但实际不涨 → 亏钱交易
- FN（False Negative）：错误预测「不涨」但实际涨 → 错过机会
- TP（True Positive）：正确预测「涨」→ 赚钱交易

## Baseline 对比

| Baseline | 含义 | 如果模型打不过 |
| --- | --- | --- |
| Majority | 永远预测多数类 | 模型没有学到任何有用的东西 |
| Persistence | 预测明天方向 = 今天方向 | 模型不如「趋势延续」这个简单假设 |

模型至少要同时超过两个 baseline 才算有意义。

## 过拟合诊断

### Overfitting Gap

`train_accuracy - test_accuracy`

| Gap 范围 | 诊断 | 含义 |
| --- | --- | --- |
| < 0.05 | `[OK]` | 泛化良好 |
| 0.05-0.10 | `[WARN]` | 中度过拟合，考虑调整正则化 |
| > 0.10 | `[FAIL]` | 严重过拟合，模型可能在记忆训练数据 |

### CV 标准差

CV 折间 accuracy 的标准差。如果 > 0.05，说明模型在不同时间段表现不稳定。

## IC 指标

### Rank IC（Spearman）

预测概率和实际收益的排序相关性。衡量「高分预测是否对应高收益」。

| IC 范围 | 含义 |
| --- | --- |
| > 0.05 | 有一定预测能力 |
| > 0.10 | 较好的预测能力 |
| < 0 | 预测方向错误 |

### ICIR（Information Coefficient IR）

`mean(IC) / std(IC)`，衡量 IC 的稳定性。

ICIR 比 IC 本身更重要：一个 IC=0.05 但 ICIR=2.0 的策略，比 IC=0.10 但 ICIR=0.3 的策略更可靠。

| ICIR 范围 | 含义 |
| --- | --- |
| > 1.0 | 信号稳定 |
| > 0.5 | 较好 |
| > 0.3 | 可接受 |

### 因子 IC

每个特征对 `future_return` 的 Rank IC。用于判断哪些特征携带了预测信息。

### 因子相关性

特征间相关系数 |r| > 0.8 的 pair 会自动在 CLI 中输出预警。高相关特征会互相替代，特征重要度排名会失真。

## 回测指标

### 收益与风险

| 指标 | 含义 | 合理区间 |
| --- | --- | --- |
| Total Return | 策略累计净收益 | > 0 |
| Annual Return | 年化净收益 | > 无风险利率 |
| Annual Vol | 年化波动率 | 结合 Sharpe 看 |
| Sharpe | 风险调整后收益 | > 0.5 可接受，> 1.0 较好 |
| Max Drawdown | 最大回撤 | < 30% |

### 交易质量

| 指标 | 含义 | 合理区间 |
| --- | --- | --- |
| Win Rate | 胜率（盈利交易比例） | > 50% |
| Profit Factor | 盈亏比（总盈利 / 总亏损） | > 1.0 |
| Turnover/yr | 年换手次数 | 20-200 |
| Signal Rate | 信号占比（有信号日 / 总测试日） | 1-5% |
| N Trades | 总交易次数 | 越多越有统计意义 |

### Regime 分拆

`--regime` 输出 Bull / Range / Bear 三个市场状态下的独立统计：

```text
-- Per-Regime Performance --
  Regime   Trades  Win Rate      Ret  Sharpe
  Bull         XX     X.XXX    X.XXXX   XX.XX
  Range        XX     X.XXX    X.XXXX   XX.XX
  Bear         XX     X.XXX    X.XXXX   XX.XX
```

重点关注：

- 熊市 Sharpe 是否为正：如果正，警惕数据挖掘
- Bull / Range / Bear 的收益是否递减：合理预期牛市收益 > 震荡市 > 熊市
- 某一种状态交易数过少：该状态的统计意义有限

## 常见误读

### 「Accuracy 0.55 看起来很低」

金融时序预测的 baseline accuracy（多数类比例）通常在 0.48-0.52 之间，所以 0.55 可能已经比 random 高 3-5 个百分点。关键是看相对 baseline 的 delta，不是绝对值。

### 「Sharpe 1.5 太棒了」

回测 Sharpe > 1.5 通常是过拟合信号。检查：

- 换手率是否过高
- 是否只在某一段行情有效（看 Regime 分拆）
- 是否依赖几个大赢家交易（看 Profit Factor）

### 「特征重要度高说明这个特征有用」

不一定。高相关特征会互相稀释重要性。检查 `High Factor Correlations` 预警。如果两个特征 |r| > 0.8，它们的重要度都不能单独信。

### 「IC 为正就够了」

IC 需要稳定才可靠。IC=0.05 但 ICIR=0.2 的信号，比 IC=0.03 但 ICIR=1.5 的信号更不可靠。永远同时看 IC 和 ICIR。
