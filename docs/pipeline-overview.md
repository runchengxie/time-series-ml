# 系统流程总览

本页说明 `ts-ml` 主流程如何从 parquet 数据走到回测结果。
本页不展开每个 CLI 参数或配置键。
适合谁：已经知道项目大概能做什么，但还没形成完整系统地图的人。
读完你会得到什么：主流程阶段、数据流、模块分工和下一步阅读路径。
相关页面：`docs/methodology.md`、`docs/cli.md`、`docs/config.md`

## 一句话地图

`ts-ml` 读入个股 parquet 日线 → 构建 30 个技术特征 → 生成次日涨跌标签（二元或 Triple Barrier）→ 训练 XGBoost（或多模型对比）→ 评估分类指标和 IC → 可选 walk-forward 回测。

```
parquet 日线
    │
    ▼
[data.py]         加载数据湖 parquet，join 申万行业分类，标记涨跌停可交易性
    │
    ▼
[features.py]     自实现技术指标：30 个特征，分属 10 个特征族
    │
    ▼
[labels.py]       生成 target：二元标签（次日涨跌幅 >= threshold）或 Triple Barrier 三向标签
    │
    ▼
[crossval.py]     PurgedTimeSeriesSplit：时序 CV + purge + embargo
    │
    ▼
[model.py]        训练 XGBoost（或多模型对比），可选概率校准
    │
    ▼
[metrics.py]      评估：Accuracy/Precision/Recall/F1/ROC AUC +
                  Baseline 对比 + Rank IC/ICIR + 因子 IC + 因子相关性
    │
    ▼
[backtest.py]     Walk-forward 逐月重训回测 + TCA + 信号过滤 + market regime
```

## 主流程阶段

| 阶段 | 模块 | 关键动作 | 输出 |
| --- | --- | --- | --- |
| 数据加载 | `data.py` | 读 parquet，join 申万行业，日期过滤，涨跌停标记 | DataFrame |
| 特征工程 | `features.py` | 30 个自实现技术指标 | 含特征列的 DataFrame |
| 标签构建 | `labels.py` | 二元或 Triple Barrier（AFML）标签 | `target` 列 + `future_return` 列 |
| 交叉验证 | `crossval.py` | PurgedTimeSeriesSplit（purge + embargo） | CV split indices |
| 模型训练 | `model.py` | XGBoost / 多模型对比 / 概率校准 | 训练好的模型 + CV stats |
| 评估 | `metrics.py` | 分类指标 + baselines + IC/ICIR + 因子分析 | 评估报告 dict |
| 回测 | `backtest.py` | Walk-forward 逐月重训 + TCA | 回测指标 dict |
| 市场状态 | `regime.py` | MA20/MA60 判定 Bull/Range/Bear | 状态标签 Series |
| 元标签 | `meta_labeling.py` | 次级模型过滤主模型预测 | 过滤后的信号 |
| 持久化 | `persistence.py` | summary.json + 多标的 CSV | artifacts/runs/ 目录 |

## 模块分工

| 模块 | 职责 | 依赖 |
| --- | --- | --- |
| `cli.py` | 命令行入口、参数解析、流程编排 | 全部模块 |
| `config.py` | Settings dataclass，不可变配置 | 无 |
| `config_yaml.py` | YAML 配置文件加载 | PyYAML（可选） |
| `data.py` | parquet 读取 + instruments join | numpy, pandas, pyarrow |
| `features.py` | 30 个技术指标（零外部依赖） | numpy, pandas |
| `labels.py` | 二元 / Triple Barrier 标签生成 | pandas |
| `crossval.py` | PurgedTimeSeriesSplit | numpy |
| `model.py` | 训练、多模型对比、概率校准 | sklearn, xgboost |
| `metrics.py` | 评估指标、baseline、IC/ICIR | sklearn, scipy |
| `backtest.py` | Walk-forward 回测 + TCA | numpy, pandas, sklearn |
| `industry.py` | 截面行业中性化 | numpy, pandas, sklearn |
| `regime.py` | 市场状态分类 | numpy, pandas |
| `meta_labeling.py` | AFML 元标签 | sklearn |
| `tracking.py` | MLflow 实验追踪（lazy import） | MLflow（可选） |
| `persistence.py` | 结果持久化（JSON + CSV） | numpy, pandas |

## 容易混淆的边界

### 1. 时序模型 vs 截面模型

time-series-ml 是时序方向预测：每只股票独立建模，预测「这只股票明天涨不涨」。

截面策略（如 `a-share-factor-core` / `cross-sectional-trees`）是横截面排序：同一交易日比较多只股票，预测「这些股票里谁更好」。

### 2. `--symbol` vs `--symbols`

- `--symbol`：单标的完整流程（数据 → 特征 → 训练 → 评估）
- `--symbols`：多标的分开独立跑（每个标的走自己的完整流程）
- `--symbols` + `--neutralize-industry`：多标的先池化做截面中性化，再拆分训练

三者是不同的执行路径，代码在 `cli.py` 中分支。

### 3. 概率校准 vs 阈值优化

- `--calibrate`：改变 `predict_proba` 的分布，让概率更接近真实概率
- `--optimize-threshold`：在验证集上搜索最优分类阈值（精确度和召回率的 trade-off）
- 两者独立，可同时使用

### 4. Purge vs Embargo

- Purge（`--purge-days`）：丢弃训练集末尾的样本。原因是这些样本的特征滚动窗口可能延伸到测试期，造成信息泄漏。
- Embargo（`--embargo-days`）：在训练集和测试集之间留间隔。原因是序列自相关可能让训练集末尾的标签和测试集开头的标签共享同一笔未来数据。

### 5. 回测收益 vs 实盘收益

Walk-forward 回测的收益是理论值，假设：

- 当日收盘价成交
- 无涨跌停限制
- 无流动性约束（等权 100 股/笔）

实盘执行由 `quant-execution-engine` 负责，做更细粒度的成交模拟和仓位管理。

## 下一步怎么读

- 想查命令：`docs/cli.md`
- 想查配置：`docs/config.md`
- 想理解策略算法：`docs/methodology.md`
- 想理解防过拟合：`docs/validation.md`
- 想理解指标解读：`docs/metrics.md`
- 想看能力清单：`docs/capabilities.md`
