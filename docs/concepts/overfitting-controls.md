# 防过拟合机制总览

本页说明：把项目里分散的时序切分、样本清理、验证、评估诊断和回测治理机制整理成一张防过拟合地图。
范围：解释每条防线解决什么问题、现在怎么使用、还有哪些边界条件；具体参数见 `docs/cli.md`，策略方法见 `docs/validation.md`。
适合对象：准备做正式研究、复核回测结果，或者想判断一条策略是否可靠的人。
读完后可以了解：哪些防线已经落地，每道防线防什么风险，以及解读结果时的保守规则。
相关页面：`docs/validation.md`、`docs/methodology.md`、`docs/metrics.md`

## 为什么时序预测特别容易过拟合

金融时间序列的最大风险不是模型复杂度太高，而是「看起来准但其实在作弊」。常见作弊方式：

| 风险 | 在本项目里的表现 | 需要的防线 |
| --- | --- | --- |
| 未来信息泄露（lookahead bias） | 训练时用了测试时才有的数据（特征滚动窗口跨越测试日期） | Purge |
| 序列自相关泄漏 | 训练集末尾的标签和测试集开头的标签共享同一笔未来数据 | Embargo |
| 过拟合噪音 | 模型记住了训练集噪音而非信号 | 时序 CV + 过拟合 gap 诊断 |
| 阈值偷看 | 在测试集上调分类阈值，然后报告测试集结果 | 验证集阈值优化（`--optimize-threshold` 训练内拆分） |
| 幸存者偏差 | 用全样本统计量做特征标准化，或数据只包含至今仍交易的股票 | 已知局限（见 `docs/limitations.md`） |
| 单条回测曲线过度解读 | 只看一条收益曲线就下结论 | Regime 分拆 + 多标的比较 |

简单说，金融样本有时间顺序和序列依赖。普通随机切分和单条回测曲线都容易让研究者过早相信结果。

## 已有机制

| 层级 | 机制 | 入口 | 防什么 |
| --- | --- | --- | --- |
| 切分 | PurgedTimeSeriesSplit | `crossval.py`、`--cv-splits` | 时序顺序，防止用未来数据训练 |
| 清洗 | Purge（`--purge-days`，默认 20） | `crossval.py`、`--purge-days` | 特征滚动窗口泄漏到测试期 |
| 禁运 | Embargo（`--embargo-days`，默认 1） | `crossval.py`、`--embargo-days` | 标签序列自相关泄漏 |
| 验证 | 时序 CV（永远用过去预测未来） | `crossval.py` | 随机切分带来的过拟合假象 |
| 评估 | 过拟合 gap 诊断 | `metrics.py`（`train_acc - test_acc`） | 模型记忆而非学习 |
| 评估 | CV 标准差检查 | `metrics.py`（CV Accuracy std） | 跨时间段不稳定 |
| 评估 | Baseline 对比（Majority + Persistence） | `metrics.py` | 模型是否比简单规则更好 |
| 评估 | Rank IC + ICIR | `metrics.py` | 信号强度和稳定性 |
| 评估 | 因子相关性审计（`|r| > 0.8` 预警） | `metrics.py` | 特征替代效应 |
| 回测 | Walk-forward 逐月重训 | `backtest.py`、`--backtest` | 用未来数据调参后回测 |
| 回测 | 信号过滤（`--prob-threshold`） | `backtest.py` | 低置信度预测污染回测 |
| 回测 | 阈值优化（验证集 grid-search） | `cli.py`、`--optimize-threshold` | 在测试集上偷看阈值 |
| 回测 | 市场状态分拆（`--regime`） | `backtest.py`、`regime.py` | 单一行情区间过度解读 |
| 模型 | 浅树 + 强正则化 | `xgb_params`（max_depth=3, reg_alpha/reg_lambda） | 模型复杂度导致的过拟合 |
| 校准 | 概率校准（`--calibrate`） | `model.py`、CalibratedClassifierCV | XGBoost 概率过于自信 |
| 追踪 | MLflow 实验记录 | `tracking.py` | 只记住赢家、忘记失败实验 |

## 防线层级地图

```
数据层：        parquet 日期过滤（无未来数据）
                   │
切分层：        PurgedTimeSeriesSplit（purge=20d + embargo=1d）
                   │
训练层：        浅树 + L1/L2 正则 + 概率校准
                   │
评估层：        train-test gap 诊断 + CV std + Baseline 对比
                   │
信号层：        Rank IC/ICIR + 因子 IC + 相关性审计
                   │
回测层：        Walk-forward 逐月重训 + 信号过滤 + 阈值优化 + Regime 分拆
                   │
记录层：        MLflow 追踪（所有实验，不只赢家）
```

## 和 cross-sectional-trees 的防过拟合差异

cross-sectional-trees 是截面排序策略，拥有更重的防过拟合工具箱（CPCV、PBO、DSR、特征消融、晋升门等），因为它在同一截面上比较多只股票，多重测试和选择偏差风险更高。

time-series-ml 是单标的时间序列方向预测，每只股票独立建模。当前不需要 CPCV 或 PBO 这类跨候选压力审计工具。如果未来扩展到面板模型（联合多标的建模），可以考虑加入这些防线。

## 读结果时的保守规则

- 不要只看 Test Accuracy。同时看 Rank IC、Overfitting Gap、CV 标准差和 baseline 对比。
- 单个标的的回测 Sharpe 只能作为线索。形成结论前，至少在多个标的上看到一致信号。
- `--regime` 分拆很重要：熊市正 Sharpe 可能是数据挖掘；熊市负 Sharpe 但总 Sharpe 正→合理。
- ICIR 比 IC 本身更重要。IC=0.05 但 ICIR=2.0 的策略，比 IC=0.10 但 ICIR=0.3 的策略更可靠。
- 特征重要度排名不等于因果解释。高相关特征（`|r| > 0.8`）会稀释彼此的重要度。

## 过拟合预警信号

出现以下信号时应警惕：

1. `train_acc - test_acc > 0.10` — 模型在记忆而非学习
2. CV 折间标准差 > 0.05 — 不同时间段表现不稳定
3. 特征重要性极度集中 — 1-2 个特征占 > 80% 总重要性
4. IC 在近期明显衰减 — 检查滚动 IC 窗口趋势
5. 预测概率分布极端 — 大量 0 或 1，缺乏中间值
6. 信号率过高（> 10%）— 可能过拟合；过低（< 0.5%）— 样本不足
7. 换手率过高（> 200/年）— 交易成本假设不成立
