# 验证与防过拟合

## 核心原则

金融时序预测的最大风险是「看起来准但其实在作弊」。常见作弊方式：

1. **未来信息泄露**（lookahead bias）：训练时用了测试时才有的数据
2. **过拟合噪音**：模型记住了训练集噪音而非信号
3. **幸存者偏差**：用全样本统计量做特征标准化
4. **阈值偷看**：在测试集上调分类阈值，然后报告测试集结果

本项目采取的防护措施全部实现在代码中，无需人工检查。

## 时序交叉验证

### PurgedTimeSeriesSplit

`crossval.py` 中的 `PurgedTimeSeriesSplit` 在标准时序 CV 上增加 purge 和 embargo 两层保护。

```
Fold 1: [train ── purge] || embargo || [test]
Fold 2: [train ────── purge] || embargo || [test]
        时间 →
```

- 永远用过去预测未来
- 测试集从不参与特征计算
- 每折模型独立训练

### Purge（清洗）

问题：训练集末尾样本和测试集开头样本的标签可能共享同一笔未来数据。

本项目标签 `target[t] = sign(close[t+1] - close[t])`。特征工程使用最长 20 天滚动窗口（SMA20）。如果训练集末尾和测试集开头无间隔，训练集最后 20 天样本的特征窗口会延伸到测试集时间范围，导致信息泄漏。

实现：默认 `purge_days=20`，每折训练前丢弃末尾 20 天。参数 `--purge-days` 可调。参考 `crossval.py:41-68`。

### Embargo（禁运期）

问题：训练集和测试集间隔不够，测试集开头样本通过序列自相关受训练集末尾影响。

本项目标签使用 `shift(-1)`，天然有 1 天间隔。此外 `PurgedTimeSeriesSplit` 在边界额外设置 `embargo_days=1`。参数 `--embargo-days` 可调。

## 信息系数（IC）

IC 衡量预测概率与实际收益之间的排序关系，是量化策略中比 accuracy 更重要的指标。实现见 `metrics.py`。

### Rank IC（Spearman）

```python
from scipy.stats import spearmanr
ic, p_value = spearmanr(predictions, actual_returns)
```

| IC 范围 | 含义 |
|---------|------|
| > 0.05 | 有一定预测能力 |
| > 0.10 | 较好的预测能力 |
| < 0 | 预测方向错误 |

### ICIR（Information Coefficient IR）

```
ICIR = mean(IC) / std(IC)
```

衡量 IC 的稳定性。ICIR 比 IC 本身更重要：一个 IC=0.05 但 ICIR=2.0 的策略，比 IC=0.10 但 ICIR=0.3 的策略更可靠。

| ICIR 范围 | 含义 |
|-----------|------|
| > 0.3 | 可接受 |
| > 0.5 | 较好 |
| > 1.0 | 信号稳定 |

### 其他 IC 分析

CLI 运行时自动输出以下分析（见 `metrics.py`）：

| 函数 | 功能 | CLI 输出位置 |
|------|------|-------------|
| `compute_ic_icir()` | 预测概率 vs 实际收益的 Rank IC | `Rank IC / ICIR` 行 |
| `compute_factor_ic()` | 每个特征对 future_return 的 Rank IC | `-- Factor IC Analysis --` 块 |
| `compute_ic_decay()` | IC 衰减曲线（lag 1-20） | 需单独调用 |
| `compute_factor_correlation()` | 特征间相关性矩阵，|r| > 0.8 预警 | `-- High Factor Correlations --` 块 |

## 样本外检验清单

以下检查项 CLI 运行时会自动输出诊断信息。CI 中不强制检查（需要真实数据），但手动跑实验后应确认全部通过。

| 检查项 | 方法 | 通过标准 | CLI 对应输出 |
|--------|------|----------|-------------|
| 时序切分 | 确认 train 日期 < test 日期 | 严格不等 | 代码保证 |
| 无 NaN 特征 | `df.isna().sum() == 0` | 全部为 0 | 代码保证 |
| 标签无 NaN | `y.isna().sum() == 0` | 全部为 0 | 代码保证 |
| 最后一行已丢弃 | 标签行数 = 原始行数 - 1 | 严格相等 | 代码保证 |
| Majority baseline | test_acc > majority_acc | 测试 > baseline | `Baselines` 块 |
| Persistence baseline | test_acc > persistence_acc | 测试 > baseline | `Baselines` 块 |
| 过拟合 gap | train_acc - test_acc | < 0.10 | `Overfitting Gap` 行 + 诊断 |
| IC 为正 | Rank IC > 0 | > 0 | `Rank IC` 行 |
| ICIR | mean(IC) / std(IC) | > 0.3 | `ICIR (rolling)` 行 |
| CV 标准差 | cv_scores.std() | < 0.05 | `CV Accuracy` 行 |

CLI 会根据阈值自动输出诊断标签：

- `[OK]` — 通过
- `[WARN]` — 接近警戒线
- `[FAIL]` — 未通过

### 诊断示例

```
Train Accuracy:        0.624
Test Accuracy:         0.617
Overfitting Gap:       0.007

[OK] Low overfitting -- good generalisation.

Rank IC:               0.052
ICIR (rolling):        1.134

[OK] Rank IC > 0.05 -- meaningful predictive signal.
[OK] ICIR > 1.0 -- stable signal.
```

## 过拟合预警信号

以下信号出现时应警惕，不要仅看 accuracy：

1. **train_acc - test_acc > 0.10** — 模型在记忆而非学习
2. **CV 折间标准差 > 0.05** — 不同时间段表现不稳定
3. **特征重要性极度集中** — 1-2 个特征占 > 80% 总重要性
4. **IC 在近期明显衰减** — 跑 `compute_ic_decay()` 查看
5. **预测概率分布极端** — 大量 0 或 1，缺乏中间值（说明模型过于自信）

## 概率校准验证

使用 `--calibrate` 后，校准效果应通过以下方式检查：

- 校准后概率在 0.45-0.55 区间的样本比例应增加（模型更"谦逊"）
- `--prob-threshold 0.55` 过滤后的信号率不应降为 0
- 校准后 IC 不应明显下降（排序能力应保持）

## 回测稳健性检查

Walk-forward 回测的可靠性取决于：

1. **足够长的测试期**：至少跨越一个完整牛熊周期（A 股约 3-5 年）
2. **合理的换手率**：turnover < 200 / 年，否则交易成本假设不成立
3. **信号率适中**：1-5% 为正常区间；> 10% 可能过拟合；< 0.5% 样本不足
4. **按 regime 拆分**：用 `--regime` 检查熊市 Sharpe。熊市正 Sharpe → 可能是数据挖掘；熊市负 Sharpe 但总 Sharpe 正 → 合理
