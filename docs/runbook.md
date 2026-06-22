# runbook

## 安装和依赖

```bash
cd ~/code/time-series-ml
uv sync --dev
source .venv/bin/activate
```

依赖项（`pyproject.toml`）：

- `numpy>=1.26, pandas>=2.0, pyarrow>=14.0` — 数据处理和 parquet 读取
- `scipy>=1.12, scikit-learn>=1.4` — 统计、校准
- `xgboost>=2.0` — 主模型
- `jupyterlab, matplotlib, seaborn` — 分析和可视化（dev）
- `pytest>=8.0, ruff>=0.6, pyright>=1.1` — 测试和代码质量（dev）
- `mlflow>=2.14` — 实验追踪（dev）

## 数据前置

必须存在数据湖目录：

```
~/data/market-data-platform/assets/tushare/a_share/daily/
  a_share_all_20150101_20260622_shadow_daily_clean/data/
```

如果数据不存在，需要先在 `market-data-platform` 仓库中运行数据管道。

## CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--symbol` | `000001.SZ` | 单标的 ts_code |
| `--symbols` | `""` | 多标的，逗号分隔 |
| `--start-date` | `20150101` | 起始日期 YYYYMMDD |
| `--end-date` | `""` | 结束日期（默认今天） |
| `--data-lake-root` | `...` | 数据湖路径 |
| `--threshold` | `0.002` | 上涨标签阈值（+0.2%） |
| `--test-size` | `0.2` | 测试集比例 |
| `--cv-splits` | `5` | CV 折数 |
| `--purge-days` | `20` | Purge 天数 |
| `--embargo-days` | `1` | Embargo 天数 |
| `--calibrate` | `False` | 启用概率校准 |
| `--cv-method` | `isotonic` | 校准方法（isotonic/sigmoid） |
| `--prob-threshold` | `0.50` | 信号过滤阈值 |
| `--optimize-threshold` | `False` | 阈值优化 |
| `--compare-models` | `False` | 多模型对比 |
| `--backtest` | `False` | Walk-forward 回测 |
| `--backtest-cost-bps` | `5.0` | 往返交易成本（基点） |
| `--regime` | `False` | 市场状态分拆统计 |
| `--config` | `""` | YAML 配置文件路径 |

## 典型工作流

### 1. 单标的快速验证

```bash
ts-ml --symbol 000001.SZ --start-date 20200101
```

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

### 3. 多标的横向比较

```bash
ts-ml \
  --symbols 000001.SZ,600000.SH,000858.SZ,600519.SH,601318.SH \
  --calibrate \
  --prob-threshold 0.55 \
  --backtest
```

### 4. YAML 配置驱动

```yaml
# my_run.yml
symbol: 000001.SZ
start_date: "20200101"
calibrate: true
prob_threshold: 0.55
backtest: true
compare_models: true
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

## 常见问题

### Q: `FileNotFoundError: .../000001.SZ.parquet`

数据湖目录不存在或路径错误。检查 `--data-lake-root` 是否指向正确路径。

### Q: 多标的模式很慢

5774 只股票全跑需要很长时间。建议先用 `--symbols` 选少量标的测试，或分批跑。

### Q: 概率校准和阈值优化冲突吗？

不冲突。`--calibrate` 校准 predict_proba，`--optimize-threshold` 用验证集搜最优分类阈值。两者可以同时用。

### Q: MLflow runs 在哪里看？

本地存储在 `./mlruns/` 目录。启动 MLflow UI：

```bash
mlflow ui --backend-store-uri file://./mlruns
```

浏览器打开 `http://localhost:5000` 即可查看所有实验记录。
