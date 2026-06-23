# 操作手册

本页提供典型工作流和常见问题解答。安装和快速跑通见 `docs/get-started.md`，CLI 参数见 `docs/cli.md`，YAML 配置见 `docs/config.md`。
目标读者：已经跑通首次运行、想做更复杂操作的人。
相关页面：`docs/get-started.md`、`docs/cli.md`、`docs/config.md`

## 典型工作流

### 1. 单标的快速验证

```bash
ts-ml --symbol 000001.SZ --start-date 20200101
```

只跑训练+评估，不回测。用于快速检查特征和模型在当前标的上是否有信号。

### 2. 单标的深度分析

```bash
ts-ml \
  --symbol 000001.SZ \
  --calibrate \
  --optimize-threshold \
  --compare-models \
  --backtest \
  --regime
```

这条命令会依次执行：概率校准 → 阈值优化 → 多模型对比选最优 → walk-forward 回测 → 按市场状态分拆统计。

### 3. 多标的横向比较

```bash
ts-ml \
  --symbols 000001.SZ,600000.SH,000858.SZ,600519.SH,601318.SH \
  --calibrate \
  --prob-threshold 0.55 \
  --backtest
```

每个标的独立训练和评估。适合比较不同标的的预测难度。

### 4. YAML 配置驱动

```yaml
# my_run.yml
symbol: 000001.SZ
start_date: "20200101"
calibrate: true
prob_threshold: 0.55
backtest: true
compare_models: true
regime: true
```

```bash
ts-ml --config my_run.yml
```

### 5. 多标的 + 行业中性化 + 市场状态分析

```bash
ts-ml \
  --symbols 000001.SZ,600000.SH,000858.SZ,600519.SH,601318.SH \
  --neutralize-industry \
  --backtest \
  --regime
```

行业中性化只在多标的模式下生效。它会池化全部标的的特征，在截面上减去行业均值再除以行业标准差，消除行业 beta 后逐标训练。

### 6. 调高信号门槛（校准后推荐）

```bash
ts-ml \
  --symbol 000001.SZ \
  --calibrate \
  --prob-threshold 0.55 \
  --backtest
```

概率校准后，`--prob-threshold 0.55` 的含义是「模型认为至少有 55% 把握时才交易」。未校准时 XGBoost 的概率偏极端，校准后阈值更有意义。

### 7. 多模型对比

```bash
ts-ml --symbol 000001.SZ --compare-models
```

同时跑 XGBoost / LogisticRegression / RandomForest / LightGBM（需安装），用 purged CV accuracy 排序自动选最优。输出：

```text
-- Model Comparison (CV accuracy) --
  Model                  CV Mean    CV Std
  XGBoost                 0.523     0.018
  LightGBM                0.520     0.020
  RandomForest            0.515     0.015
  LogisticRegression      0.508     0.012
```

## MLflow 实验追踪

本地 runs 存储在 `./mlruns/` 目录。启动 MLflow UI 查看所有实验记录：

```bash
mlflow ui --backend-store-uri file://./mlruns
```

浏览器打开 `http://localhost:5000`。

MLflow 通过 `tracking.py` 的 lazy-import 模式接入：如果 mlflow 未安装，所有调用静默跳过，不影响正常运行。

## 常见问题

### Q: `FileNotFoundError: .../000001.SZ.parquet`

数据湖目录不存在或路径错误。检查 `--data-lake-root` 是否指向正确路径，或确认 `market-data-platform` 的数据管道已经运行过。

### Q: 多标的模式很慢

5774 只股票全跑需要很长时间。建议先用 `--symbols` 选少量标的测试，或分批跑。

### Q: 概率校准和阈值优化冲突吗？

不冲突。`--calibrate` 校准 `predict_proba` 让概率更接近真实概率，`--optimize-threshold` 用验证集 grid-search 最优分类阈值。两者可以同时用，互不干扰。

### Q: 行业中性化为什么需要 --symbols？

行业中性化依赖截面（同一交易日多只股票）数据。`--symbol` 单标的模式没有截面，无法做中性化。需要在 `--symbols` 多标的模式下使用。

### Q: 回测的 return 是净收益还是毛收益？

walk-forward 回测输出的 Total Return / Sharpe 等指标已经是扣除了 `--backtest-cost-bps` 指定的往返交易成本后的净收益。

### Q: 如何在 CI 中运行？

```bash
uv run ruff check src/ tests/
npx pyright src/
uv run python -m pytest -q
```

详见 `docs/dev.md`。
