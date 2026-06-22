# xgboost-ts-ashare

使用 XGBoost 预测 A 股次日涨跌的**时序**二分类策略。从 `xgboost-aapl` 迁移而来，数据源切换为 market-data-platform A 股数据湖。

> 与截面策略区分：GitHub 上的 `a-share-factor-core` 是截面因子策略（横向比较同日多只标的），本项目是时序方向预测（单标的逐日预测次日涨跌）。

## 与 xgboost-aapl 的关键差异

| 方面 | xgboost-aapl | xgboost-ashare |
|------|-------------|----------------|
| 数据源 | TuShare API / Alpaca | 本地数据湖 parquet |
| 标的 | AAPL 单只美股 | 5774 只 A 股 |
| 标的池 | 单标的 | 多标的横向比较 |
| 概率校准 | 无 | CalibratedClassifierCV |
| 信号过滤 | 无 | prob_threshold 过滤 |
| Alpaca | 支持 | 删除 |
| 缓存 | 按 symbol+date 缓存 API 调用 | 直接读 parquet，无需缓存 |

## 项目结构

```text
xgboost-ashare/
├── src/xgboost_ashare/
│   ├── __init__.py        # 包信息
│   ├── config.py          # Settings dataclass
│   ├── config_yaml.py     # YAML 配置文件加载
│   ├── data.py            # 数据湖 parquet 读取
│   ├── features.py        # 自实现技术指标（12 个特征）
│   ├── labels.py          # 标签构建
│   ├── crossval.py        # PurgedTimeSeriesSplit
│   ├── model.py           # XGBoost + 校准 + 多模型对比
│   ├── backtest.py        # Walk-forward 回测 + 信号过滤
│   ├── metrics.py         # 评估：baseline、IC/ICIR、因子分析
│   └── cli.py             # 命令行入口
├── tests/
├── docs/
├── pyproject.toml
└── README.md
```

## 快速开始

### 安装

```bash
cd ~/code/xgboost-ashare
uv sync --dev
source .venv/bin/activate
```

### 数据前置

A 股日线数据已由 market-data-platform 产出到数据湖：

```
~/data/market-data-platform/assets/tushare/a_share/daily/
  a_share_all_20150101_20260622_shadow_daily_clean/data/
    000001.SZ.parquet
    000002.SZ.parquet
    ... 5774 files
```

### 运行

```bash
# 单标的（默认 000001.SZ 平安银行）
python -m xgboost_ashare.cli

# 单标的 + 概率校准 + 信号过滤
python -m xgboost_ashare.cli --calibrate --prob-threshold 0.55

# 多标的横向比较
python -m xgboost_ashare.cli --symbols 000001.SZ,600000.SH,000858.SZ,600519.SH

# 完整流程：校准 + 阈值优化 + 多模型对比 + 回测
python -m xgboost_ashare.cli --calibrate --optimize-threshold --compare-models --backtest

# 通过 YAML 配置
python -m xgboost_ashare.cli --config my_experiment.yml

# 自定义日期范围
python -m xgboost_ashare.cli --start-date 20200101 --end-date 20250622
```

## 新增功能

### 概率校准 (`--calibrate`)

XGBoost 的 `predict_proba` 不是真实概率。加 `CalibratedClassifierCV` 让概率回归校准，为信号过滤提供可信的基础。

```bash
python -m xgboost_ashare.cli --calibrate --cv-method isotonic
```

### 信号过滤 (`--prob-threshold`)

过滤低置信度预测，只对概率 >= 阈值的交易日产生信号。

```bash
python -m xgboost_ashare.cli --calibrate --prob-threshold 0.55 --backtest
```

### 多标的模式 (`--symbols`)

```bash
python -m xgboost_ashare.cli --symbols 000001.SZ,600000.SH,000858.SZ
```

## 测试

```bash
pytest                          # 全部测试
pytest tests/test_labels.py     # 标签测试
pytest tests/test_features.py   # 特征测试
```

## 代码质量

```bash
ruff check .                    # Lint
ruff format --check .           # 格式检查
pyright                         # 类型检查
```

## 下一步（Phase 2/3）

- 行业中性：申万行业分类 + 截面中性化
- MLflow 实验追踪
- 市场状态分类（牛/熊/震荡）
- 更多特征：资金流向、北向资金、融资融券

## 许可证

MIT
