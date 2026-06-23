# 开发与测试

本页说明本地开发环境、测试命令、CI 架构、代码质量闸门和贡献指南。
适合谁：需要改代码、跑回归测试、排查 CI 的开发者。
相关页面：`docs/get-started.md`、`docs/pipeline-overview.md`

## 环境准备

```bash
cd ~/code/time-series-ml
uv sync --dev
source .venv/bin/activate
```

## 日常检查命令

```bash
# Lint
uv run ruff check src/ tests/

# 类型检查
npx pyright src/

# 测试（23 个单元测试）
uv run python -m pytest -q

# 完整回归（Lint + 类型检查 + 测试）
uv run ruff check src/ tests/ && npx pyright src/ && uv run python -m pytest -q
```

注意：必须用 `uv run python -m pytest`，不能直接用 `uv run pytest`（pytest binary 不在 PATH 中）。

## 测试文件

| 测试文件 | 覆盖模块 |
| --- | --- |
| `tests/test_data.py` | `data.py`：数据加载、行业 join |
| `tests/test_features.py` | `features.py`：12 个技术指标 |
| `tests/test_labels.py` | `labels.py`：标签构建 |
| `tests/test_split.py` | `crossval.py`：PurgedTimeSeriesSplit |
| `tests/test_industry.py` | `industry.py`：截面行业中性化 |

### 测试覆盖缺口

以下模块无独立单元测试，通过 CLI 端到端运行间接验证：

| 模块 | 未覆盖的函数 |
| --- | --- |
| `model.py` | `train_model`、`compare_models`、`calibrate_model` |
| `metrics.py` | `evaluate`、`compute_ic_icir`、`compute_factor_ic`、`compute_factor_correlation`、`compute_ic_decay` |
| `crossval.py` | `PurgedTimeSeriesSplit`、`purged_cross_val_score` |
| `backtest.py` | `walk_forward` |

添加测试的优先级：`crossval.py` > `metrics.py` > `model.py` > `backtest.py`。

## 代码质量闸门

修改完成后应通过以下检查，否则不应认为工作完成：

1. `uv run ruff check src/ tests/` — 零问题
2. `npx pyright src/` — 零 error
3. `uv run python -m pytest -q` — 全部通过

### Ruff 配置

- 行长度：100 字符
- 引号风格：双引号
- 启用规则：E / F / I / B / UP / SIM / C4 / PTH / RUF
- 忽略 RUF001/RUF002/RUF003（中文标点是有意使用）

### Pyright 配置

- 模式：`basic`
- 范围：`src/`（tests 排除）
- `reportMissingTypeStubs: false`

## GitHub Actions CI

CI 触发条件：push / pull request。

运行步骤：

```yaml
# .github/workflows/ci.yml
steps:
  - checkout
  - uv sync --extra dev
  - ruff check src/ tests/
  - npx --yes pyright src/
  - uv run python -m pytest -q
```

CI 不运行模型验证或回测（需要真实数据），只做代码质量检查。

## 技术栈

| 类别 | 工具 | 版本 |
| --- | --- | --- |
| Python | 运行环境 | >= 3.11 |
| 包管理 | uv | 最新 |
| 数据处理 | numpy, pandas, pyarrow | >= 1.26 / >= 2.0 / >= 14.0 |
| 统计 | scipy | >= 1.12 |
| 机器学习 | scikit-learn, xgboost | >= 1.4 / >= 2.0 |
| 测试 | pytest | >= 8.0 |
| Lint | ruff | >= 0.6 |
| 类型检查 | pyright | >= 1.1 |
| YAML | pyyaml | >= 6.0（dev optional） |
| 实验追踪 | mlflow | >= 2.14（dev optional） |
| 可视化 | jupyterlab, matplotlib, seaborn | dev optional |

## 添加新依赖

1. 编辑 `pyproject.toml`
2. 运行 `uv sync`
3. `uv.lock` 会自动更新
4. 如果是必须的依赖（代码 import），放入 `[project.dependencies]`
5. 如果是开发工具，放入 `[project.optional-dependencies] dev`

## 项目约定

- **无 emoji**：CLI 输出使用 `[OK]`、`[WARN]`、`[FAIL]` 标签，不使用 emoji。
- **中文文档**：使用中文标点 （）「」：，。；不用英文引号、不用 `**粗体**`、不用破折号。写自然通顺的中文。
- **自实现特征**：特征工程零外部依赖，不使用 `pandas_ta` 或 `ta-lib`。
- **uv 命令**：测试用 `uv run python -m pytest`，不用 `uv run pytest`。
- **MLflow 可选**：`tracking.py` 使用 lazy-import 模式，mlflow 未安装时静默跳过。CI 中默认不启动 mlflow。
- **pyright basic 模式**：不对第三方无类型 stub 的库报错。

## 提交 PR 前检查清单

1. `uv run ruff check src/ tests/` 无新问题
2. `npx pyright src/` 零 error
3. `uv run python -m pytest -q` 全部通过
4. 如果修改了文档，检查 `docs/README.md` 和 `README.md` 是否需要同步更新
5. 如果新增了 CLI 参数，更新 `docs/cli.md`
6. 如果新增了配置键，更新 `docs/config.md`
7. 如果新增了依赖，更新 `pyproject.toml` 并运行 `uv sync`
