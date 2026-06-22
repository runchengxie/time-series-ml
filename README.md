# time-series-ml

时序机器学习框架：用多种模型（XGBoost / LogisticRegression / RandomForest / LightGBM）预测次日涨跌方向。目前以 A 股为主要市场，数据源为 market-data-platform A 股数据湖。

> 与截面策略区分：GitHub 上的 `a-share-factor-core` 是截面因子策略（横向比较同日多只标的），本项目是时序方向预测（单标的逐日预测次日涨跌）。

## 项目结构

```text
time-series-ml/
├── src/ts_ml/
│   ├── __init__.py        # 包信息
│   ├── config.py          # Settings dataclass
│   ├── config_yaml.py     # YAML 配置文件加载
│   ├── data.py            # 数据湖 parquet 读取 + 行业 join
│   ├── features.py        # 自实现技术指标（12 个特征）
│   ├── labels.py          # 标签构建
│   ├── crossval.py        # PurgedTimeSeriesSplit
│   ├── model.py           # 多模型 + 概率校准
│   ├── industry.py        # 申万行业截面中性化
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
cd ~/code/time-series-ml
uv sync --dev
source .venv/bin/activate
```

### 运行

```bash
# 单标的（默认 000001.SZ 平安银行）
python -m ts_ml.cli

# 或 CLI 入口
ts-ml

# 多模型对比 + 概率校准 + 信号过滤 + 回测
ts-ml --calibrate --prob-threshold 0.55 --compare-models --backtest

# 多标的 + 行业中性化
ts-ml --symbols 000001.SZ,600000.SH,000858.SZ,600519.SH,601318.SH \
      --neutralize-industry --backtest
```

## CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--symbol` | `000001.SZ` | 单标的 |
| `--symbols` | `""` | 多标的，逗号分隔 |
| `--start-date` | `20150101` | 起始日期 |
| `--calibrate` | `False` | 概率校准 |
| `--prob-threshold` | `0.50` | 信号过滤阈值 |
| `--neutralize-industry` | `False` | 行业截面中性化 |
| `--compare-models` | `False` | 多模型对比 |
| `--backtest` | `False` | Walk-forward 回测 |

## 测试

```bash
pytest             # 23 个测试
ruff check .       # Lint
pyright            # 类型检查
```

## 文档

| 文档 | 内容 |
|------|------|
| [docs/methodology.md](docs/methodology.md) | 策略算法、特征体系、模型架构、回测方法论、TCA 假设 |
| [docs/validation.md](docs/validation.md) | 防过拟合措施、Purge/Embargo、IC/ICIR、检验清单、预警信号 |
| [docs/roadmap.md](docs/roadmap.md) | 已完成 / 待完成功能清单 |
| [docs/runbook.md](docs/runbook.md) | 安装、数据前置、典型工作流、常见问题 |

## 许可证

MIT
