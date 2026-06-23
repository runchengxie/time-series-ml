# time-series-ml

时序机器学习框架：用多种模型（XGBoost / LogisticRegression / RandomForest / LightGBM）预测次日涨跌方向。目前以 A 股为主要市场，数据源为 market-data-platform A 股数据湖。

> 与截面策略区分：GitHub 上的 `a-share-factor-core` 是截面因子策略（横向比较同日多只标的），本项目是时序方向预测（单标的逐日预测次日涨跌）。

## 项目结构

```text
time-series-ml/
├── src/ts_ml/
│   ├── __init__.py           # 包信息
│   ├── config.py             # Settings dataclass
│   ├── config_yaml.py        # YAML 配置文件加载
│   ├── data.py               # 数据湖 parquet 读取 + 行业 join
│   ├── features.py           # 自实现技术指标（30 个特征）
│   ├── labels.py             # 标签构建（二元 + Triple Barrier）
│   ├── crossval.py           # PurgedTimeSeriesSplit
│   ├── model.py              # 多模型 + 概率校准
│   ├── industry.py           # 申万行业截面中性化
│   ├── backtest.py           # Walk-forward 回测 + 信号过滤
│   ├── metrics.py            # 评估：baseline、IC/ICIR、因子分析
│   ├── regime.py             # 市场状态分类
│   ├── meta_labeling.py      # AFML 元标签（次级过滤模型）
│   ├── tracking.py           # MLflow 实验追踪
│   ├── persistence.py        # 结果持久化（summary.json + CSV）
│   └── cli.py                # 命令行入口
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_labels.py
│   ├── test_triple_barrier.py  # Triple Barrier 标签测试
│   ├── test_split.py
│   ├── test_industry.py
│   ├── test_crossval.py        # PurgedTimeSeriesSplit 测试
│   └── test_backtest.py        # Walk-forward 回测测试
├── docs/
│   ├── README.md             # 文档导航首页
│   ├── get-started.md        # 快速上手
│   ├── runbook.md            # 操作手册（工作流、FAQ）
│   ├── cli.md                # CLI 参考
│   ├── config.md             # YAML 配置参考
│   ├── pipeline-overview.md  # 系统流程总览
│   ├── methodology.md        # 策略方法论
│   ├── validation.md         # 验证与防过拟合
│   ├── metrics.md            # 指标与结果解读
│   ├── capabilities.md       # 功能清单
│   ├── limitations.md        # 已知局限
│   ├── dev.md                # 开发与测试
│   └── concepts/
│       ├── overfitting-controls.md  # 防过拟合机制总览
│       ├── execution-costs.md       # 成本与执行假设
│       └── model-landscape.md       # 模型版图与扩展路线
├── pyproject.toml
└── README.md
```

## 环境变量

数据湖路径可通过环境变量配置，方便不同机器复用：

```bash
export TIME_SERIES_ML_DATA_LAKE_ROOT=/path/to/parquet/data
export TIME_SERIES_ML_INSTRUMENTS_PATH=/path/to/instruments.parquet
```

不设置时使用内置默认值（指向 Richard 本机路径），CLI 的 `--data-lake-root` 参数也支持直接覆盖。

## 快速开始

```bash
cd ~/code/time-series-ml
uv sync --dev
source .venv/bin/activate

# 单标的快速验证
ts-ml --symbol 000001.SZ --start-date 20200101

# 深度分析
ts-ml --symbol 000001.SZ --calibrate --compare-models --backtest --regime
```

详细步骤见 `docs/get-started.md`。

## 测试

```bash
uv run python -m pytest -q     # 35 个测试（含 crossval、backtest、triple barrier）
uv run ruff check src/ tests/  # Lint
npx pyright src/               # 类型检查
```

详见 `docs/dev.md`。

## 文档

| 文档 | 内容 |
| --- | --- |
| [docs/README.md](docs/README.md) | 文档导航首页，按问题找页面 |
| [docs/get-started.md](docs/get-started.md) | 前置条件、最短跑通、运行后检查 |
| [docs/runbook.md](docs/runbook.md) | 典型工作流、常见问题 |
| [docs/cli.md](docs/cli.md) | CLI 命令与全部参数速查 |
| [docs/config.md](docs/config.md) | YAML 配置参考与模板 |
| [docs/pipeline-overview.md](docs/pipeline-overview.md) | 系统流程总览、模块分工 |
| [docs/methodology.md](docs/methodology.md) | 策略算法、特征体系、模型架构、回测方法论 |
| [docs/validation.md](docs/validation.md) | 防过拟合措施、Purge/Embargo、IC/ICIR、检验清单 |
| [docs/metrics.md](docs/metrics.md) | 指标含义、解读顺序、常见误读 |
| [docs/capabilities.md](docs/capabilities.md) | 功能清单：数据处理、特征、模型、验证、回测、工程化 |
| [docs/limitations.md](docs/limitations.md) | 已知局限 |
| [docs/dev.md](docs/dev.md) | 开发环境、测试命令、CI 架构、代码质量闸门 |
| [docs/concepts/overfitting-controls.md](docs/concepts/overfitting-controls.md) | 防过拟合机制全景地图 |
| [docs/concepts/execution-costs.md](docs/concepts/execution-costs.md) | 交易成本模型、执行假设、适用边界 |
| [docs/concepts/model-landscape.md](docs/concepts/model-landscape.md) | 模型版图与扩展路线 |

## 许可证

MIT
