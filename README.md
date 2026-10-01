# FactorBridge

从嘈杂基因表达数据中恢复可复核的表达因子，研究功能语义是否能改善数值恢复。

An experimental framework for evidence-guided expression factor recovery and programme analysis.

**截至 2026-10-01：两轮真实 Qwen QLoRA 训练已完成。第二轮完整评估发现，带功能语义的微调模型对验证集 56 个候选全部弃选；尚未证明优于数值基线。当前版本作为可复现的研究试验归档，不是已验证的生物学预测系统。**

## 从哪里开始

| 阅读目的 | 入口 |
| --- | --- |
| 和计算机方向合作者讨论问题、方法和结果 | [项目说明](docs/PROJECT_DESCRIPTION_CN.md) |
| 查看两轮训练、对照结果和失败证据 | [实验记录](experiments/README.md) |
| 区分已执行、仅实现和待研究部分 | [项目状态与讨论决策](docs/PROJECT_STATUS.md) |
| 理解文件、矩阵与分组要求 | [数据契约](docs/DATA_CONTRACT.md) |
| 复现第二轮实验 | [语义流程运行说明](docs/semantic_pipeline_CN.md) |
| 找到旧版下载、环境排错和训练命令 | [文档索引](docs/README.md) |

## 核心任务

输入为样本 × 基因的表达矩阵 X，以及物种、测量平台和 biological unit 元数据。输出为基因 × 因子的 loadings W、样本 × 因子的 scores Z、支持基因、证据、稳定性与不确定性。

```text
noisy X → 数值候选及分组 bootstrap → Candidate Evidence Card
        → Qwen 结构化选择 → 在同一个 noisy X 上数值重估 → W/Z 与独立评估
```

第二轮增加 NCBI/Reactome 功能注释、noisy-only 富集、受限子模块候选，以及 programme 分数、功能对应候选和时间分析。LLM 选择支持集合；实际 W/Z 由数值方法估计。稳定表达方向、功能解释和因果生物机制是不同层次的结论。

## 已观察到的结果

公共数据实际纳入 8 个研究、24 个数据面板、1,371 个样本；6 个研究用于训练、2 个用于验证。组织面板和噪声版本不算独立研究。内部 killifish 保持 frozen test，未使用。

| 实验 | 训练卡 / 验证卡 | 真实执行 | 结论 |
| --- | ---: | --- | --- |
| E01 数值弱监督 | 352 / 32 | 3 epochs，66 steps | 优化完成；所提供报告未包含该 adapter 的独立恢复评估 |
| E02 功能语义弱监督 | 542 / 56 | 3 epochs，102 steps，九项对照 | 完整语义微调模型 56/56 uncertain；对应 W 为零列 |

E02 的当前弱参考恢复率：loading refit 与稳定性均为 41.7%，未微调 Qwen 为 8.3%，完整语义微调 Qwen 为 0%。去掉语义后同一 adapter 恢复率为 41.7%，但其 52 个 retain 中有 50 个选了全部可见基因。该消融不能证明模型学会了语义推理。

新增的 190 个 programme 训练候选全部被旧 PCA 弱参考标成 uncertain；单个研究占 78.2% 的训练卡。下一步优先重审监督目标、研究均衡和 checkpoint 选择，而不是直接增加训练规模。详见[实验分析](experiments/README.md)。

## 代码导航

| 模块 | 职责 |
| --- | --- |
| `data.py`、`benchmark.py`、`factors.py` | 数据审计、分组、候选、弱参考和数值重估 |
| `contracts.py`、`llm.py`、`selectors.py` | 可见证据契约、QLoRA、非 LLM 对照 |
| `semantic.py`、`programmes.py` | 功能证据、programme 分数、时间与偏离分析 |
| `evaluate.py`、`recovery.py` | 对照评估、W/Z 输出、新公共数据恢复 |
| `cross_species.py`、`distill.py` | 配对/教师导入/辅助损失；两轮报告均未执行外部教师训练 |
| `scripts/`、`configs/`、`tests/` | 运行入口、配置与工程检查 |

原代码路径保留，避免破坏 AutoDL 命令和旧运行的溯源。外层 agent 只有函数接口；外部世界模型、强化学习、跨平台统一模型和统一 factor graph 均未运行。

## 复现与结果保护

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

GPU 依赖、实际数据和权重需要另外配置。默认配置文件不是已下载数据清单。第二轮入口为 `scripts/run_semantic_pipeline.sh <source-config>`，用于复现已归档方案；现有 adapter 不建议直接作为最终恢复器。

每次运行使用新目录。输入卡不能含 clean reference、目标标签或测试结果；原始参考与标签放在 private sidecar。study、biological unit、parent factor 及其所有噪声版本不跨 split。没有数据或依赖时明确失败，不使用假数据回退。

公开仓库只保存代码、文档和经过整理的实验摘要。原始表达矩阵、完整运行包、adapter 权重和内部数据保留在本地/AutoDL；摘要记录源包哈希，可回查证据。

## 归档

本次归档说明见 [2026-10-01 归档索引](docs/archive/2026-10-01.md)。仓库继续可编辑，旧实验结果作为历史证据保留，不因后续修改而重写。
