# 快速上手

本页解决什么：用最短路径跑通一次完整流程。
范围：配置细节与模型选择见后续文档。
适合谁：第一次进入仓库或只想验证环境的人。
读完你会得到什么：一套可重复的最小跑通流程与产物检查清单。
相关页面：`docs/runbook.md`、`docs/cli.md`、`docs/config.md`、`docs/pipeline-overview.md`

## 前置条件

- Python 3.11 及以上版本
- `uv` 包管理器
- A 股日线 parquet 数据湖（由 `market-data-platform` 生产）

数据湖默认路径：

```text
~/data/market-data-platform/assets/tushare/a_share/daily/
  a_share_all_20150101_20260622_shadow_daily_clean/data/
```

如果你的数据在其他位置，通过环境变量指定：

```bash
export TIME_SERIES_ML_DATA_LAKE_ROOT=/your/path/to/parquet/data
export TIME_SERIES_ML_INSTRUMENTS_PATH=/your/path/to/instruments.parquet
```

或者通过 CLI 参数直接指定：

```bash
ts-ml --symbol 000001.SZ --data-lake-root /path/to/your/data/lake
```

如果数据不存在，需要先在 `market-data-platform` 仓库中运行数据管道。

## 最短跑通

```bash
cd ~/code/time-series-ml
uv sync --dev
source .venv/bin/activate

# 单标的，默认 000001.SZ（平安银行）

ts-ml --symbol 000001.SZ --start-date 20200101
```

## 运行后检查

CLI 会在终端输出完整的评估报告。你应该看到：

- `[features] Features built.` — 特征工程完成
- `[train] Training XGBoost ...` — 模型训练开始
- `[eval] Evaluating ...` — 评估进行中
- `EVALUATION SUMMARY` — 包含 Train Accuracy、Test Accuracy、ROC AUC、Rank IC 等
- `[OK]` / `[WARN]` / `[FAIL]` — 各项自动诊断

重点关注：

- `Test Accuracy` 是否超过 Majority baseline（表示模型比随机猜有用）
- `Overfitting Gap`（train_acc - test_acc）是否 < 0.10
- `Rank IC` 是否 > 0，`ICIR` 是否 > 0.3

## 下一步建议

- 想做深度分析（概率校准 + 多模型对比 + 回测 + 市场状态分拆）：`docs/runbook.md`
- 想理解策略算法和模型架构：`docs/methodology.md`
- 想理解防过拟合设计：`docs/validation.md`
- 想理解系统流程：`docs/pipeline-overview.md`
- 想查命令或配置定义：`docs/cli.md`、`docs/config.md`
