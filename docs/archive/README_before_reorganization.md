# 历史 README 快照

下文保留整理前状态；“尚未训练”等陈述已过时，当前结论见[项目概览](../../README.md)。

# FactorBridge

本轮公共数据扩展与直接训练入口见 [PUBLIC_EXPANDED_TRAINING.md](../PUBLIC_EXPANDED_TRAINING.md)：复用实际通过的 GPU smoke，公共研究按 train/validation 分组，内部 killifish 留作最终 frozen test。全目录下载与可训练数据纳入状态分开报告。

从 noisy expression 中恢复可复核的 biological factors，并研究跨物种 factor identity。

**当前状态：工程实现、受控模拟测试与真实公共数据的训练前准备；没有执行真实 Qwen GPU 训练，也没有已训练 adapter 或生物学增益结论。**

本仓库按 2026-09-30 的交接要求启动。用户随后明确增加了第二阶段和多教师对比/蒸馏；因此当前范围是 Stage 1 + Stage 2。跨平台、跨分辨率训练、统一 graph、RL、agent、自动教师路由均不实现。

## Stage 1

`数值候选 → noisy-only Candidate Evidence Card → Qwen 结构化选择 → noisy X 上 support-restricted PCA → W/Z → 独立评估`

- PCA rank 受有效 biological-unit 数量和矩阵秩限制；保留正负 loading。
- bootstrap 按完整 unit 抽样。study、parent dataset、parent factor 和 biological units 不跨 split。
- counts 才允许 binomial thinning；log expression 使用明确的加性噪声。missing features 用 mask 排除。
- 受控模拟真值或真实数据的稳定性弱参考提供监督。真实未确认基因不作为 non-LLM selector 的可靠负例。
- `retain / uncertain / reject_null`、当前可见 gene slots、混杂 axis flags、合法 evidence IDs；不预测 pathway 名称。
- completion-only loss；超长样本明确失败；不截断答案，不静默丢弃，不用 Qwen 自举答案造 gold。
- raw PCA 为独立原始诊断；loading-refit、stability、non-LLM、frozen Qwen、finetuned Qwen 共用相同数值重估器。
- 输出 full-gene W、discovery/held-out Z、support/missing masks、中心化参数、Factor Cards。
- sign/permutation-aware 计分、子空间覆盖、null/technical 假阳性、abstention、invalid rate、支持集、噪声曲线、按 study 重采样的区间。
- 报告候选可表示的支持覆盖上限，以及真值支持集交集的 oracle-refit 对照。后者是可达到的参考值，**不是对所有可能支持子集的数学上界**。

## Stage 2 与多教师

Stage 1 adapter 完成并通过 GPU smoke test 后，才继续跨物种 SFT。每次追加训练保留恢复任务 replay，并重测 Stage 1 的公开 validation W/Z 恢复。

- 默认先比较 ESM2 protein-token prior、UCE contextual factor vectors、无蒸馏学生、各单教师学生、固定权重多教师学生。
- ESM2 是 UCE 使用的蛋白/基因表示基础；两者不是独立生物证据。ESM2 pooling 不等于完整 UCE 推理。
- contextual teacher 必须先通过其官方实现对**同一 noisy 输入**运行。仓库提供有来源/哈希检查的导入和 factor-vector 导出；不内置或假装执行 UCE/SATURN/scGPT 的训练代码。
- 其他模型可按同一真实 embedding 契约加入独立 `teacher_id`；必须记录 checkpoint、revision、物种暴露情况和输入 assay。没有合适权重/输入就明确阻断。
- 多教师各自保留一个辅助回归头：`SFT biological-label loss + λ × fixed-weight teacher-similarity MSE`。回归读取 **最后一个 prompt token** 的隐状态，不读取答案位置；教师数值也从模型 prompt 中移除。
- 教师输出只监督辅助损失。`shared / partially_shared / unmatched / uncertain` 的主任务标签来自受控真值或带来源的 curated weak reference。
- 同时按 species、study、unit、parent factor、programme family 分组。Stage 1 replay 不得包含 Stage 2 的 held-out species。
- pair 比较先控制 assay/resolution；基于 sign-invariant 表示，不把 PCA 符号翻转叫 biological reversal。
- 只在公开 validation 上选择实验，且检查恢复退化与无效输出率。可按 species pair / relation 看各教师的强项，不建立自动 router。

## 快速使用

已完成单研究 pilot、准备下一步时，请用 [训练前准备](../BEFORE_TRAINING.md)：下载三个独立 study、固定 train/validation/reserved-test、生成弱监督 SFT、检查实际 Qwen tokenizer，并在模型加载与训练之前停止。该批只有小规模真实弱参考，不能作为完整混杂识别或跨物种训练集。

首次在 AutoDL/JupyterLab 获取代码，或先下载真实公共数据，请从 [公共数据起步](../PUBLIC_PILOT.md) 开始。仓库已公开，HTTPS clone/pull 无需账号或密码。已提供经过文件哈希校验的 GSE124109 下载/转换脚本，以及明确隔离的单研究数值试跑。

在仓库根目录运行，Python 3.10+。核心数值测试只需要 NumPy：

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m factorbridge environment --config configs/stage1.json
```

AutoDL 运行顺序、真实数据 manifest、单卡训练和 Stage 2 操作见 [AutoDL 说明](../AUTODL.md) 和 [数据契约](../DATA_CONTRACT.md)。

`configs/stage1.json` / `stage2.json` 是待填数据路径的配置，**不是可用数据清单**。默认 manifest 不存在时应当失败。没有假数据自动回退。

独立、明确标记的模拟工程验收：

```bash
python scripts/run_simulation_check.py --destination runs/engineering_check
```

这会实际产生模拟 W/Z、Factor Cards、基线报告、噪声/coverage-error 数据；不会加载或微调 Qwen，不证明真实生物学有效。

## 输出与冻结

每次实验使用新的 `run_dir`；prepared/evaluation/training 不覆盖旧结果。`commands.jsonl` 记录真实命令、时间、失败与错误。

`prepared/cards.jsonl` 只含候选可见证据；`prepared/private/` 单独存 reference、labels、lineage 和矩阵。SFT 的 prompt-completion 导出另存 `sft/`。文件中的 `example_id` 仅用于 join，不进入 prompt。

GPU `smoke` 必须实际完成加载、前后向、优化、保存、重载和 logits 比对后，`train` 才能运行。训练过程中和结束后保存真实环境、resolved dependency lock、模型 SHA、adapter、tokenizer、chat template、loss、显存峰值与训练参数清单。

`freeze` 固定规则、代码和模型/基线哈希；之后才能打开 public test。内部 killifish 从始至终保持 `internal_test`，只允许冻结后的局部无监督恢复，不提供训练入口。内部测试结论不能回调全局规则。

无 CUDA、数据、权重、独立标签或教师 embedding 时明确失败。不要把工程测试结果、模拟标签一致性、真实弱参考一致性叫作“恢复了真实因果因子”。

## 官方技术依据

- [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- [PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization)
- [UCE 官方实现与输入要求](https://github.com/snap-stanford/UCE)

训练采用 Transformers + PEFT 的显式 SFT 循环，以便直接验证 loss mask 与多教师辅助损失；没有引入 TRL/RL 的额外训练流程。依赖区间是安装约束；实际兼容版本只有 AutoDL smoke 完成后生成的 `requirements.resolved.txt` 才算已验证。

## 功能语义实验（2026-10-01）

新入口 `scripts/run_semantic_pipeline.sh <已完成公共数据训练的config.json>`：官方功能注释、noisy-only 功能候选与富集证据、Qwen QLoRA、同流程 W/Z 重估、九项对照、programme 分数、跨物种功能对应候选，以及按未来时间留出的进程/偏离与状态转移基线。复用已有公共 manifest；内部 killifish 不参与；外层 agent 仅提供函数接口。

完整启动命令、输入/输出、科学限制和世界模型参考见 [运行说明](../semantic_pipeline_CN.md)。此版本的发布不表示 AutoDL 新一轮训练已执行或模型已优于基线。
