# 功能清单

## 数据处理

- **A 股数据湖接入**：从 market-data-platform parquet 数据湖读取个股日线，支持 5500+ 标的
- **申万行业分类**：自动 join instruments 表，支持行业筛选和截面分析
- **日期范围过滤**：`--start-date` / `--end-date` 灵活切片
- **多标的加载**：单标的或逗号分隔多标的，跳过缺失文件

## 特征工程

全部自实现，零外部依赖（无 pandas_ta / ta-lib）：

| 类别 | 特征 | 说明 |
|------|------|------|
| 动量 | SMA5/10/20 差分 | 多周期均线偏离 |
| 相对强弱 | RSI(14) | Wilder 平滑 |
| 趋势 | MACD 柱 | 标准 12/26/9 |
| 波动率 | 滚动波动率 5/10/20 日 | 年化波动率 |
| 成交量 | 量比、换手率 | 相对活跃度 |
| 形态 | 实体比例、上下影线比例 | 蜡烛形态量化 |
| 波动幅度 | ATR(14) 百分比 | 归一化波幅 |

## 模型

- **XGBoost**：主模型，低学习率 + 浅树 + 强正则化
- **多模型对比**（`--compare-models`）：XGBoost / LR / RF / LightGBM，自动选最优
- **概率校准**（`--calibrate`）：CalibratedClassifierCV，isotonic 或 sigmoid
- **行业中性化**（`--neutralize-industry`）：截面去行业均值，消除行业 beta
- **YAML 配置驱动**（`--config`）：所有参数可配置文件管理，支持 xgb_params 覆盖

## 验证体系

### 时序交叉验证

- **PurgedTimeSeriesSplit**：标准时序 CV + purge + embargo 三层保护
- **Purge**（`--purge-days`，默认 20）：丢弃训练集末尾，防特征窗口泄漏
- **Embargo**（`--embargo-days`，默认 1）：训练/测试边界间隔

### 评估指标

- 分类指标：Accuracy / Precision / Recall / F1 / ROC AUC
- Baseline 对比：Majority（多数类）和 Persistence（昨日方向）
- 过拟合诊断：`train_acc - test_acc`，自动 [OK] / [WARN] / [FAIL]
- 混淆矩阵 + 分类报告

### IC 分析

- **Rank IC**：预测概率 vs 实际收益的 Spearman 相关系数
- **ICIR**：IC 均值 / IC 标准差，衡量信号稳定性
- **因子 IC**（`compute_factor_ic`）：每个特征对 future_return 的 Rank IC
- **IC 衰减**（`compute_ic_decay`）：IC 随 forward horizon 的衰减曲线
- **因子相关性**（`compute_factor_correlation`）：|r| > 0.8 自动预警
- CLI 自动输出 [OK] / [WARN] / [FAIL] 诊断标签

## 回测

- **Walk-Forward**（`--backtest`）：逐月重训，t 时刻模型只见过 t 之前数据
- **信号过滤**（`--prob-threshold`）：低置信度预测不交易
- **阈值优化**（`--optimize-threshold`）：验证集 grid-search 最优分类阈值
- **TCA**（`--backtest-cost-bps`）：往返交易成本模拟
- **市场状态分拆**（`--regime`）：Bull / Range / Bear 三维回测统计

### 回测指标

Total Return / Annual Return / Annual Vol / Sharpe / Max Drawdown / Win Rate / Profit Factor / Turnover / Signal Rate

## 工程化

- **MLflow 实验追踪**（`tracking.py`）：自动记录参数和指标，本地 file store，零侵入
- **GitHub Actions CI**：push/PR 触发 ruff + pyright + pytest
- **23 个单元测试**：覆盖 labels / features / split / cache / industry
- **类型检查**：pyright basic 模式 0 error
- **Lint**：ruff 0 error
- **uv 包管理**：lock 文件锁定版本，可复现

## CLI

```bash
ts-ml --symbol 000001.SZ --backtest --regime
ts-ml --symbols 000001.SZ,600519.SH --neutralize-industry --calibrate --backtest
ts-ml --config my_run.yml
```

全部 CLI 参数见 `docs/runbook.md`。
