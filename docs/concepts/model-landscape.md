# 模型版图与当前选择

> status: active
> owner: time-series-ml
> last_verified: 2026-06-23
> source_of_truth: yes

本页解决什么：从更广的算法宇宙理解 time-series-ml 当前为什么只维护 XGBoost 为主的几个模型，以及下一步什么时候值得扩展模型族。
范围：分析模型适配度，不展开具体参数。参数见 `docs/cli.md`、`docs/config.md`。
适合谁：想判断某类算法是否值得纳入本项目，或想解释当前模型选择原因的读者。
相关页面：`docs/methodology.md`、`docs/config.md`、`docs/validation.md`

先说结论：

- time-series-ml 是单标的时间序列方向预测任务，不是截面排序
- 当前代码支持的模型：XGBoost（分类/回归）、LR、RF、LightGBM、Ridge
- 最适合当前任务的模型族是 Boosting（XGBoost/LightGBM），其次是时间序列深度学习模型
- 当前优先级最高的改进是特征质量和标签设计，不是模型扩张
- 和 cross-sectional-trees 的关键差异：那里是截面排序 → XGBRanker 最贴题；这里是时序方向 → XGBClassifier 最贴题

## 1. 先把任务说对

time-series-ml 的核心任务是：

- 单只股票、独立建模、逐日预测
- 每一行是 (日期, 特征向量) → 预测未来某个标签
- 标签可以是二分类（涨/跌）、三分类（止盈/止损/到期）、或连续值（收益率）
- 评估关注 IC/ICIR（排序能力）、回测 Sharpe（实际收益）
- 最终动作：信号触发后进行单边买卖，不涉及跨股排序

这和 cross-sectional-trees 有本质不同。后者的最终动作是「同日 5000 只股票中选 Top-K」，天然适合 ranking 目标。time-series-ml 的最终动作是「这只股票今天该不该买」，天然适合分类或回归目标。

## 2. 我们可以考虑的主流机器学习算法

按任务结构分组：

| 类别 | 代表算法 | 在本项目里的典型角色 |
| --- | --- | --- |
| 线性分类 | Logistic Regression | 分类 baseline，判断非线性是否有增量 |
| 线性回归 | Ridge、LinearRegression | 回归 baseline（regression 模式） |
| 树模型 | 决策树、随机森林、Extra Trees | 非线性对照，特征交互探索 |
| Boosting | XGBoost、LightGBM、CatBoost | 当前主线——最贴表格数据与时间序列任务 |
| 时间序列深度学习 | LSTM、GRU、TCN | 显式建模序列依赖，理论上更优但工程成本高 |
| Transformer 系列 | Informer、PatchTST、TimesNet | 最新时间序列模型，长序列优势，需要大样本 |
| 概率模型 | Gaussian Processes | 自带不确定性估计，计算代价大 |
| 经典时序模型 | ARIMA、GARCH、VAR | 统计推断好但预测能力弱，适合做 regime 诊断 |
| 序列模型 | HMM、Markov Switching | regime 检测和状态切换，不是直接预测器 |
| 降维 / 表征学习 | PCA、Autoencoder、t-SNE | 特征工程或诊断工具，不是主预测模型 |
| 序列标注 | CRF、Structured Perceptron | 结构化输出（如标注涨跌停序列），过于复杂 |
| 概率增强 Boosting | NGBoost、Quantile Regression GBDT | 自带概率输出或分位数预测 |
| 集成方法 | Stacking、Blending | 合成多模型信号 |
| 生存分析 | Cox PH、Random Survival Forest | 天然贴合 Triple Barrier——建模 barrier 触及时间 |
| 可解释非线性 | MARS、GAM | 特征诊断和关系可视化 |
| 图神经网络 | GCN、GAT | 利用股票间关系（行业、相关性等） |

## 3. 各类算法在本项目里的适配度

### 3.1 线性分类家族

| 算法 | 擅长什么 | 为什么合适 / 不合适 |
| --- | --- | --- |
| Logistic Regression | 二分类 baseline，可解释性强 | 合适——当前二分类标签的最自然 baseline；如果 LR 无信号但 XGBoost 有，需警惕过拟合 |
| Softmax Regression | 多分类（3+ 类） | 合适——Triple Barrier 三分类标签的自然 baseline |

time-series-ml 天然是分类问题（预测方向），所以线性分类是必要的 sanity check。当前 `--compare-models` 中已包含 LR。

### 3.2 线性回归家族

| 算法 | 擅长什么 | 为什么合适 / 不合适 |
| --- | --- | --- |
| Ridge | 稳定的线性回归，抗共线性 | 合适——regression 模式的 sanity baseline（`--regression`） |
| Lasso | 稀疏化特征选择 | 价值有限，大部分被 elasticnet 和树模型的特征重要度覆盖 |
| ElasticNet | 收缩 + 稀疏化 | 可做线性对照，但 time-series-ml 当前特征量少（30 个），稀疏化意义不大 |

线性回归在 time-series-ml 的 regression 模式下有价值：如果 Ridge 的 IC=0，说明特征完全没有线性预测力，XGBoost 的 IC 可能只是噪音过拟合。

### 3.3 树模型与 Bagging

| 算法 | 擅长什么 | 为什么当前不优先 |
| --- | --- | --- |
| 单棵决策树 | 规则可解释 | 太不稳定，把噪声学成规则 |
| 随机森林 | 非线性、抗过拟合、特征交互 | 合适——当前已在 `--compare-models` 中，可做非 boosting 对照 |
| Extra Trees | 更随机的 RF | 和 RF 高度重叠，边际价值低 |

随机森林在 time-series-ml 中的价值：当 XGBoost 和 RF 的 CV accuracy 差距很大时，可以判断非线性增益来自 boosting 机制本身还是来自集成。

### 3.4 Boosting 家族——当前主线

| 算法 | 擅长什么 | 为什么适合 |
| --- | --- | --- |
| XGBoost Classifier | 表格数据、非线性、特征交互 | 主线——当前默认模型，经验最丰富，参数最成熟 |
| XGBoost Regressor | 表格数据回归 | 合适——regression 模式默认模型 |
| LightGBM | 更大规模数据、更快训练 | 合适——已集成在 `--compare-models` 中，作为 boosting challenger |
| CatBoost | 类别特征处理、少调参 | 有一定价值，但对当前纯数值特征的优势不明显 |

为什么 Boosting 最贴题：

- 当前数据是标准表格型量价 + 估值特征
- 非线性和条件交互在金融时序中很常见
- XGBoost 的低学习率 + 浅树 + 强正则化配置专门针对金融高信噪比设计
- 不需要像深度学习那样考虑序列长度、padding 等问题

当前结论：XGBoost + LightGBM 作为 main + challenger 已经足够。CatBoost 可以后续按需添加。

### 3.5 时间序列深度学习

| 算法 | 擅长什么 | 为什么当前不优先 |
| --- | --- | --- |
| LSTM | 长期依赖、序列记忆 | 理论上最合适——显式建模 sequence 结构，但需要重构数据为 (samples, timesteps, features) 格式 |
| GRU | 类似 LSTM，参数更少 | 和 LSTM 高度重叠，选一个即可 |
| TCN (Temporal CNN) | 并行训练、长感受野 | 比 LSTM 更工程友好，但在金融数据上的文献积累不如 LSTM |
| Transformer (Informer, PatchTST) | 长序列、注意力机制 | 近年 SOTA，但需要大样本量和 GPU |

什么时候它们会变得合理：

- 当前 30 个特征 + XGBoost 的 IC 天花板在 0.10 左右
- 如果确信量价序列中有 XGBoost 捕捉不到的模式（如形态组合、多尺度模式），LSTM/Transformer 可能突破
- 重构数据格式为滑动窗口 `(samples, window_size=60, features=30)`
- 需要 GPU 和更多实验时间

当前不优先的原因：窗口构建、训练稳定性、超参搜索的成本高，而特征质量和标签设计还有很大改进空间。先把特征和标签做好，再决定是否引入序列模型。

### 3.6 经典时间序列模型

| 算法 | 擅长什么 | 为什么不适合做主预测器 |
| --- | --- | --- |
| ARIMA / SARIMA | 单变量时序预测，统计推断完备 | 只能预测自身序列，不能纳入多维特征 |
| GARCH 家族 | 波动率建模 | 波动率是特征不是收益——可用来预测波动但不是 alpha |
| VAR | 多变量时序 | 高维时参数爆炸，不适合 30 个特征 |
| Markov Switching | regime 检测 | 当前已有 MA20/MA60 的 regime 分类，Markov 版本可作为更精细的替代 |

经典时序模型在 time-series-ml 中的合理定位：

- GARCH 作为特征：预测波动率 → 作为 HistVol 的替代特征
- Markov Switching 作为 regime：替代当前的 MA 规则 → 更精细的市场状态分类
- 不做主预测器：这些模型无法纳入 30 个特征，预测能力远不如 ML

### 3.7 概率模型

| 算法 | 擅长什么 | 为什么当前不优先 |
| --- | --- | --- |
| Gaussian Processes | 自带不确定性估计，小样本 | O(n³) 复杂度，2000+ 样本会非常慢 |
| Bayesian Neural Networks | 权重不确定性 | 工程复杂，边际收益不明 |
| 概率校准（已有） | 将分类概率映射到真实概率 | 已有——`--calibrate` 使用 CalibratedClassifierCV |

概率校准我们已经有了，更重的概率模型（GP、Bayesian NN）在当前规模下性价比低。

### 3.8 降维与表征学习

| 方法 | 擅长什么 | 定位 |
| --- | --- | --- |
| PCA | 线性降维、去共线性 | 当特征膨胀到 50+ 且有明显共线性时可考虑 |
| Autoencoder | 非线性表征 | 需要大量样本和 GPU，当前 30 特征不需要 |
| t-SNE / UMAP | 可视化、分布漂移检测 | 诊断工具——看特征是否随时间漂移 |

当前 30 个特征不需要降维。等基本面、资金流特征加入后（50+ 特征），PCA 可以作为线性支线的预处理。

### 3.9 概率增强 Boosting

| 算法 | 擅长什么 | 为什么值得考虑 |
| --- | --- | --- |
| NGBoost（Natural Gradient Boosting） | 自带概率输出，预测完整分布参数 | 不需要额外校准就能给出校准概率；在 Triple Barrier 三分类中天然输出每个 barrier 的概率 |
| Quantile Regression GBDT | 预测收益分布的各个分位数 | 对 Triple Barrier 的信号过滤很有价值——不只关心 profit 概率的均值，还关心分布的尾部 |

和 Triple Barrier 的关系：

- Triple Barrier 的三分类输出天然适合概率模型——NGBoost 可以直接输出 `P(stop_loss), P(timeout), P(profit_take)` 的完整分布
- Quantile Regression 可以输出 profit 概率的第 95 分位数——只在信号极端强时交易
- 当前 XGBoost + softmax 已经给了概率，NGBoost 的优势在于概率更校准

推进时机：当 XGBoost 的 IC 稳定 > 0.10 后，用 NGBoost 做对照，验证概率质量是否有提升。

### 3.10 集成 Stacking / Blending

| 方法 | 擅长什么 | 为什么值得考虑 |
| --- | --- | --- |
| Stacking | 多个异构模型输出作为 meta-model 输入 | XGBoost + LightGBM + RF → meta-LR，合成更稳健的信号 |
| Blending | 用 hold-out 训练 meta-model | 更简单，减少过拟合 |

和 Meta-Labeling 的区别：

- Meta-Labeling：primary model 预测 → secondary model 判断「是否信任 primary」
- Stacking：多个 primary model 各自预测 → meta model 综合所有预测
- 当前已有 Meta-Labeling（效果弱），Stacking 可以在 Meta-Labeling 稳定后作为补充

推进时机：当 XGBoost、LightGBM、RF 在同一个标的上都有正 IC 时，Stacking 有正期望。

### 3.11 生存分析（Survival Analysis）——最贴合 Triple Barrier

| 算法 | 擅长什么 | 为什么高度贴合 |
| --- | --- | --- |
| Cox Proportional Hazards | 建模「事件发生时间」和协变量的关系 | Triple Barrier 本质是「在多长时间内、以哪种 barrier 结束」——就是生存分析 |
| Random Survival Forest | 非线性的生存分析 | 比 Cox 更灵活 |

为什么这是 time-series-ml 最自然的模型扩展：

- Triple Barrier 标签已经给出了 `(time_to_barrier, event_type)` 的信息——只是我们把它简化成了单点三分类
- 生存分析可以直接预测「未来 20 天内触及 profit_take 的概率曲线」，而不是单一的「是/否」
- 当前 K=10 是全局参数——生存分析可以去掉这个限制，让模型自己学最优持有期
- `labels.py` 已经在重建路径——生存分析可以更精细地使用这些信息

推进时机：当 Triple Barrier 信号稳定（IC > 0.05）后，这应该是第一个模型扩展方向。它比 LSTM 更贴题，工程成本更低（Cox PH 就是线性模型）。

### 3.12 可解释非线性模型

| 算法 | 擅长什么 | 为什么当前不优先 |
| --- | --- | --- |
| MARS | 自动发现非线性 hinge 函数 | 可解释性强，但预测力不如 GBDT |
| GAM | 每个特征的独立非线性贡献可视化 | 特征诊断工具——看某个特征在哪个区间失效 |

定位：特征诊断工具箱，不是主预测器。例如用 GAM 看 `log_pe` 对 profit 概率的贡献是否在 PE>50 后逆转。

### 3.13 图神经网络（GNN）

| 算法 | 擅长什么 | 为什么当前不优先 |
| --- | --- | --- |
| GCN / GAT | 利用股票间的关系图 | 可以把「同行业、同板块股票的联动」编码进模型 |

和 single-stock 任务的关系：

- time-series-ml 是单股时序预测，天然不需要图。但可以扩展为：
  - 用同行业其他股票的走势作为这只股票的额外特征
  - 这就从时序模型变成了时空模型（temporal + graph）
- 需要先建设行业关联图或相关性图的数据层

推进时机：当单股特征遇到天花板后，作为多股联动信号的方向推进。目前不是优先级。

## 4. 为什么当前就保留这几个模型

| 模型 | 当前角色 | 它主要回答什么 |
| --- | --- | --- |
| XGBoost Classifier | 主线 | 给定特征，能否预测涨跌 / barrier 方向 |
| XGBoost Regressor | regression 模式主线 | 给定特征，能否预测收益率数值 |
| Logistic Regression | 分类 sanity check | 线性模型有没有信号？非线性增益可信吗 |
| Ridge | 回归 sanity check | 在回归模式下，线性基线是什么水平 |
| Random Forest | 非 boosting 对照 | 增益来自树本身还是 boosting 机制 |
| LightGBM | boosting challenger | XGBoost 的优势是否可复现到其他 boosting |

这套组合覆盖了：

- 线性 vs 非线性的对比（LR/Ridge vs XGBoost）
- Boosting vs Bagging 的对比（XGBoost vs RF）
- 不同 boosting 实现的对比（XGBoost vs LightGBM）
- 分类 vs 回归的对比（Classifier vs Regressor）

## 5. 和 cross-sectional-trees 模型选择的差异

| 维度 | cross-sectional-trees | time-series-ml |
| --- | --- | --- |
| 任务 | 截面排序：同日 5000 只股票排 Top-K | 时序方向：单只股票预测涨跌 |
| 最贴题的模型 | XGBRanker（学习排序） | XGBClassifier（学习分类） |
| 核心 baseline | Ridge（线性排序是否有效） | LR（线性分类是否有效） |
| Ranking 模型 | XGBRanker 是主线 | 不适合——单股没有排序目标 |
| 需要扩展的 | 排序目标优化、多截面一致性 | 序列建模（LSTM/Transformer）、特征质量 |

## 6. 哪些方向值得继续推进

| 方向 | 为什么值得做 | 什么时候推进 |
| --- | --- | --- |
| 特征质量（基本面、资金流） | 当前 30 个特征以量价为主，IC 天花板 0.10 | 现在——最高优先级 |
| Triple Barrier 持有期优化 | 当前 K=10 是最优，但不同股票的最优 K 可能不同 | 现在 |
| 生存分析（Cox PH） | Triple Barrier 天然是生存分析问题，可去掉固定 K | Triple Barrier IC > 0.05 后——第一个模型扩展 |
| NGBoost | 自带校准概率，Triple Barrier 三分类直接受益 | IC > 0.10 后做对照 |
| LSTM / Transformer 序列模型 | 显式建模时间依赖，可能突破 XGBoost 天花板 | 当特征 IC > 0.10 且稳定后 |
| Stacking / Blending | 合成多模型信号 | 当多个模型各有正 IC 时 |
| CatBoost 作为 challenger | 验证 boosting 家族的一致性 | 低优先级，LightGBM 已够用 |
| GARCH 波动率特征 | 替代当前 HistVol，更精准的波动率估计 | 中优先级，作为特征工程扩展 |
| MARS / GAM | 特征非线性关系诊断 | 按需使用——诊断工具 |
| Markov Switching regime | 替代 MA 规则的更精细市场状态 | 中优先级 |
| GNN | 利用行业/相关性图关系 | 单股特征遇天花板后 |
| 概率模型（GP） | 不确定性估计 | 低优先级，计算代价大 |
| 经典时序模型做 baseline | ARIMA 作为最朴素基线 | 低优先级，仅有教学意义 |

当前更合理的优先级：

1. 继续扩展特征维度（基本面 PIT 财务、资金流、行业截面特征）
2. 优化 Triple Barrier 参数（不同股票的最优 profit_take / stop_loss / holding_period）
3. Triple Barrier IC > 0.05 后，引入生存分析（Cox PH）——去掉固定 K，让模型自己学最优持有期
4. 等特征 IC 稳定 > 0.10 后，引入 LSTM/Transformer 做序列建模
5. NGBoost、Stacking、GNN 等放到更后面

## 7. 一句话收口

time-series-ml 当前选择 `XGBoost + LightGBM + RF + LR + Ridge` 的模型组合，已经覆盖了单股时间序列分类任务所需的所有 sanity check 和对比维度。在特征质量和标签设计还没到位之前，往模型动物园里加 LSTM 或 Transformer，通常不如先把特征从 30 个扩到 50+ 个更有效。
