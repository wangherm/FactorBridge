# AutoDL 运行说明

## 1. 获取代码与检测环境

还未成功 clone 或尚无公共数据时，先按 [JupyterLab 公共数据起步](PUBLIC_PILOT.md) 操作。该数值 pilot 无需 GPU；以下 GPU 训练流程只用于独立 study 数据划分齐备后的正式配置。

仓库为私有。先用你已授权的 GitHub 登录方式完成 Git 认证；不要把 token 写进代码、聊天或 Git URL。

```bash
git clone https://github.com/wangherm/FactorBridge.git
cd FactorBridge
python -m venv .venv
source .venv/bin/activate
# 先按 AutoDL 驱动/镜像选择支持实际 GPU 的 PyTorch CUDA wheel。
python -m pip install -e '.[train]'
python -m factorbridge environment
python -m unittest discover -s tests -v
```

用户指定单张 RTX PRO 6000；显存容量、驱动、CUDA、BF16 必须以 `environment` 及 `nvidia-smi` 实测为准。代码不把口述的 120 GB 当硬件事实。新架构 GPU 必须用兼容的 CUDA/PyTorch 构建，安装成功不代表训练成功。

`preprocessing_workers` 并行编码输入；`micro_batch_size` 控制同一 GPU 上每次训练的样本数，`gradient_accumulation_steps` 控制累积。单卡不启用 DDP，不在多个 Python 线程中各加载一份 Qwen。

QLoRA 默认 NF4/double quant；显存实测允许时可以把 `load_in_4bit` 改为 `false` 使用 LoRA。显式设置 BF16 时不支持就报错。调整 batch/精度后用新的 run 配置重新 smoke，不复用旧通过记录。

## 2. 数据和配置

把矩阵及 biological-unit metadata 上传到 AutoDL 数据盘，或从官方来源自行下载并保留来源。论文提到 GSE124109 等 accession 不表示本仓库附带数据。

```bash
cp configs/stage1.json configs/stage1.local.json
# 编辑 stage1.local.json 的 manifest、run_dir、batch 等。
python -m factorbridge pin-model --config configs/stage1.local.json
python -m factorbridge audit --config configs/stage1.local.json
python -m factorbridge prepare --config configs/stage1.local.json
python -m factorbridge baselines --config configs/stage1.local.json
python -m factorbridge smoke --config configs/stage1.local.json
python -m factorbridge train --config configs/stage1.local.json
```

先 pin 模型 revision，再 prepare，避免配置改变使 prepared hash 失效。若 tokenizer 报超长，明确缩小 `card_genes` 并新建 run；若 GPU OOM，调整 batch 并新建 run。不得修改错误日志或假造 passed 文件。

## 3. 公平评估

```bash
python -m factorbridge evaluate --config configs/stage1.local.json --split validation \
  --methods pca_raw loading_refit stability non_llm qwen_frozen qwen_finetuned
python -m factorbridge freeze --config configs/stage1.local.json
python -m factorbridge evaluate --config configs/stage1.local.json --split test \
  --methods pca_raw loading_refit stability non_llm qwen_frozen qwen_finetuned
```

选择 rank、阈值、support 大小等只能使用公开 validation；需要改变配置时用新 run 保存完整溯源。默认阈值是预声明起点，不是假装优化后的规则。

内部数据必须在 manifest 中事先保留为 `internal_test`；不要冻结后更改配置。冻结后才运行：

```bash
python -m factorbridge prepare-internal --config configs/stage1.local.json
python -m factorbridge evaluate --config configs/stage1.local.json --split internal_test \
  --methods pca_raw loading_refit stability non_llm qwen_frozen qwen_finetuned
```

`freeze --baseline-only` 只用于明确的数值工程对照；该冻结不能在看过测试后追加 Qwen 并伪装成同一预注册实验。

## 4. 跨物种教师数据

先完成 Stage 1。基于同一冻结数值规则分别生成不同物种的 Factor Cards 和实际 W/Z。对 train role 导出时可运行 `evaluate --split train`，其结果只能称训练集拟合诊断。

导入真实 ESM2 gene-symbol → tensor dictionary（例如 UCE 的 protein-token 资源），记录准确的来源版本：

```bash
python -m factorbridge import-protein-tokens --source /actual/path/species_ESM2.pt \
  --species SPECIES --revision ACTUAL_ESM2_RESOURCE_VERSION --destination data/species_esm2.npz
```

这里的全大写参数是需要替换的操作说明，正式 manifest 不可使用占位符。

对于 UCE 或其他 contextual 模型：

1. 固定其实际 checkpoint/revision，用官方推理代码对相应 **noisy** expression 运行。UCE 要求 scRNA counts，不能将 bulk/log matrix 改名后喂入。
2. 导出 sample-level `samples`、`embeddings` 到 NPZ。single-cell 结果必须先按真实 biological units 聚合，与 FactorBridge 输入一致。不要随机分组 cells 造 replicates。
3. 写明 provenance，包含 checkpoint/input/embedding 哈希和训练物种暴露状态。模型预训练已见某物种时，只能报告 student-held-out，不能声称 teacher 也未见。
4. 同一 noisy view 的 Factor Cards 放入单独 JSONL 后，导出其 factor-associated embeddings：

```bash
python -m factorbridge export-factor-teacher --source data/uce_samples.npz \
  --provenance data/uce_samples.provenance.json \
  --factor-cards data/one_noisy_view_factor_cards.jsonl \
  --destination data/uce_factors.npz
```

教师输出的数值相似性与独立主任务标签分开。没有对齐的 sample IDs、输入 hash、实际 vectors 就失败；不会运行另一个模型作 fallback。

## 5. 单教师/多教师训练与选优

```bash
cp configs/stage2.json configs/stage2.local.json
# 填 stage1_config、cross_species_manifest，并使用 Stage 1 相同的 model_revision。
# distillation_weights 中每个教师必须有实际可用数据；第三个教师可添加新的 ID。
python -m factorbridge ablation-configs --config configs/stage2.local.json --destination configs/teacher_experiments
```

依次对生成的 `supervised_only.json`、各单教师和 `multi_teacher.json` 执行以下命令。单卡串行跑各实验，避免多个 Qwen 进程争抢显存：

```bash
CFG=configs/teacher_experiments/multi_teacher.json
python -m factorbridge prepare --config "$CFG"
python -m factorbridge baselines --config "$CFG"
python -m factorbridge smoke --config "$CFG"
python -m factorbridge train --config "$CFG"
python -m factorbridge evaluate --config "$CFG" --split validation
python -m factorbridge regression --config "$CFG"
```

全部实验在同样的 validation pairs 和 recovery replay 上对比后：

```bash
python -m factorbridge select-experiment --config configs/stage2.local.json \
  --experiments configs/teacher_experiments/supervised_only.json \
    configs/teacher_experiments/esm2.json configs/teacher_experiments/uce.json \
    configs/teacher_experiments/multi_teacher.json \
  --destination runs/teacher_selection.json
```

选择标准是预配置的无效输出率上限、Stage 1 recovery 退化容限，以及通过门槛后的 validation relation agreement；不是查看 test 后挑最优。无实验满足要求时 `selected=null`。`by_species_pair` / `by_relation` 展示不同教师的适用差异。

只对选中的实验 `freeze`，再 `evaluate --split test`。保存 Stage 1 与 Stage 2 独立 adapters，不覆盖第一阶段模型。

## 实际执行边界

截至本地交付，未连接 AutoDL、未获得真实表达数据/教师 embeddings/跨物种参考标签、未下载 Qwen 权重，未执行任何 GPU 加载、训练、adapter 保存重载。GPU 路径需在 AutoDL smoke 中实测，不能把本地 NumPy 单元测试当作成功证据。
