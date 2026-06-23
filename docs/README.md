# 文档首页

用途：提供唯一文档入口、问题索引和阅读路径。
范围：这里只做导航；具体命令、配置和概念细节放在对应页面。
适合读者：不知道先看哪一页的人。

## 先按问题找页面

| 我现在的问题 | 先看哪一页 |
| --- | --- |
| 我第一次进入仓库，想先跑起来 | `docs/get-started.md` |
| 我想先知道系统从数据到结果的流转过程 | `docs/pipeline-overview.md` |
| 我想查命令和参数 | `docs/cli.md` |
| 我想查 YAML 配置键和模板 | `docs/config.md` |
| 我想理解策略算法和模型架构 | `docs/methodology.md` |
| 我想理解防过拟合设计和检查清单 | `docs/validation.md` |
| 我想理解所有指标的含义和解读方法 | `docs/metrics.md` |
| 我想理解交易成本假设和执行边界 | `docs/concepts/execution-costs.md` |
| 我想看防过拟合机制的全景地图 | `docs/concepts/overfitting-controls.md` |
| 我想查典型工作流和常见问题 | `docs/runbook.md` |
| 我想看项目功能边界和能力清单 | `docs/capabilities.md` |
| 我想看已知局限和测试覆盖缺口 | `docs/limitations.md` |
| 我想改代码、跑测试、提交 PR | `docs/dev.md` |

## 四条阅读路径

1. 我想先跑起来：`docs/get-started.md`
2. 我想先建立系统心智模型：`docs/pipeline-overview.md` → `docs/methodology.md` → `docs/capabilities.md`
3. 我想做正式研究：`docs/get-started.md` → `docs/runbook.md` → `docs/validation.md` → `docs/concepts/overfitting-controls.md` → `docs/metrics.md`
4. 我想查某个细节：`docs/cli.md`、`docs/config.md`、`docs/methodology.md`、`docs/concepts/`

## 页面分工

入口：`README.md`、本页
任务路径：`docs/get-started.md`、`docs/runbook.md`
系统总览：`docs/pipeline-overview.md`、`docs/capabilities.md`
参考手册：`docs/cli.md`、`docs/config.md`、`docs/methodology.md`、`docs/metrics.md`
概念解释：`docs/concepts/execution-costs.md`、`docs/concepts/overfitting-controls.md`
验证与局限：`docs/validation.md`、`docs/limitations.md`
开发：`docs/dev.md`

## 常用术语

| 术语 | 本项目用法 |
| --- | --- |
| ts-ml | CLI 入口命令 |
| Purge | 丢弃训练集末尾样本，防止特征窗口泄漏到测试期 |
| Embargo | 训练/测试边界间隔，防止序列自相关泄漏 |
| IC / ICIR | 信息系数 / 信息系数信息比，衡量预测排序能力和稳定性 |
| Walk-Forward | 逐月重训回测，t 时刻模型只见过 t 之前数据 |
| TCA | 交易成本分析（Transaction Cost Analysis），模拟佣金+印花税+滑点 |
| Regime | 市场状态分类（Bull / Range / Bear），按 MA20/MA60 判定 |
| 行业中性化 | 截面上去除行业均值，消除行业 beta |
| 概率校准 | CalibratedClassifierCV，将 XGBoost 原始概率映射为真实概率 |
