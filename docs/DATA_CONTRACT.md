# 数据与跨物种证据契约

## Stage 1 manifest

JSON 顶层 `datasets` 数组。每个数据集必填以下字段；路径以 manifest 所在目录为基准，也可使用真实绝对路径。

| 字段 | 要求 |
|---|---|
| dataset_id | 唯一数据集 ID |
| study_id | 原始独立研究/队列 ID；整个 study 只属于一个全局 split |
| parent_dataset | 原始母数据集 ID；衍生视图必须一致 |
| matrix_path | 真实存在的 sample × gene CSV 或 NPZ |
| metadata_path | 真实存在的 CSV |
| species / assay / resolution | 来源可核实的实验信息，不能猜测 |
| data_scale | `raw_counts` 或 `log_expression` |
| biological_unit_col | metadata 中真实 donor/embryo/pool 等单位列名 |
| condition_col | metadata 中 condition 列；不自动进入模型输入 |
| role | `public_train`、`public_validation`、`public_test`、`internal_test` |
| source_kind | `public_real`、`internal_real`、`controlled_simulation` |
| truth_path | 只有 controlled_simulation 需要：包含 W/Z/axes 的 NPZ |

NPZ 含 `X`（二维有限数值）、`genes`（不重复字符串）、`samples`（不重复字符串），禁止 pickle。CSV 第一列 sample ID，第一行其余列为 gene IDs。

metadata 至少含 `sample_id`、指定 unit/condition 列。矩阵和 metadata 的 sample IDs 必须完全一致，程序不会悄悄丢行。

默认 unit 命名空间是 study；同一 donor/pool 在不同 accession 中复用时，额外提供统一的 `global_unit_id`。相同 parent dataset 跨 accession 也必须给同一个 parent ID。缺少真实独立单位信息时不能进入训练。

每个数据集至少 8 个真实独立单位，留出至少 2 个 unit 只用于投影。该最小数目是工程限制，不代表统计功效充足。regional H/B/T 同 embryo 全部一起抽样和划分。

一个 Stage 1 run 只处理同 species/assay/resolution/scale 的研究；不同物种分别做局部数值恢复。所有内部 killifish 数据必须为 `internal_real` + `internal_test`。

## 输出数组

`evaluation_*/factors/*.npz`：

- `W`: 原始 gene universe × recovered factors，未选中及未测量基因 loading 为 0，必须结合 mask 解释。
- `Z_discovery` / `Z_heldout`: 相应 sample 顺序 × recovered factors。
- `genes`、`discovery_samples`、`heldout_samples`: 显式坐标。
- `measured_mask` 与 `support_mask`: 未测量和未选中分开。
- `mean` / `measured_gene_order`: discovery 均值；数值流程中心化、不另做方差标准化。

当全部 abstain 时，W/Z 是 0 个 factor 的实际空列数组，不生成伪因子。零 loading 不表示未测量基因已恢复。

## Stage 2 manifest

顶层包含：

- `factor_card_files`: 路径数组，或 `{"path": "真实 factor_cards.jsonl", "prefix": "物种/研究唯一前缀:"}` 数组，避免不同 run 的 factor ID 冲突。
- `protein_embeddings`: `{"species": "真实物种", "path": "已导入的 ESM2 NPZ"}` 数组，各物种在同一 ESM2 representation space。
- `factor_embeddings`: `{"teacher_id": "uce", "path": "已导出的 factor vectors NPZ", "prefix": "与 Factor Cards 相同前缀:"}` 数组。可以加入额外经过核验的教师，不能只填模型名称。
- `orthology`: `path` 和固定数据库 `version`。
- `pairs_path`: 实际配对 JSONL，和标签文件分开。
- `labels_path`: 来源明确的标签 JSONL。

Orthology CSV 列：`species_a,gene_a,species_b,gene_b,relation`。首版只使用明确的 `one_to_one` 行；one-to-many 不强制折叠成一对一。匹配不到不代表无 biology。

Pair JSONL 每行含 `pair_id`、`factor_a`、`factor_b`、`split`、`programme_family`。factor IDs 是前缀 + Factor Card 的真实 ID。programme_family 是分组 sidecar，不进入输入。

Labels JSONL 每行含 `pair_id`、`label_source`（`controlled_truth` 或 `curated_weak_reference`）、`source_reference`（可追溯来源）、`target`。target 字段为：

```json
{
  "relation": "uncertain",
  "supported_links": [],
  "evidence_ids": [],
  "limitations": []
}
```

上面只说明 schema，不是可用训练样本。supported_links 只能引用按输入 orthology 顺序构建的 `l001` 等链接。只因 teacher cosine 很高而填写 `shared`，或随机配对后填写 `unmatched`，均不是合格监督。

species、study、parent factor、unit、programme family 都不能跨 Stage 2 splits。真实 Stage 2 需要物种整体留出；共享训练物种的新 pair 只能作为 in-distribution 诊断。Stage 1 已训练/回放物种不能宣称为 Stage 2 未见物种。

## Contextual teacher 原始 embedding provenance

sample embedding NPZ 含 `samples`、`embeddings`。provenance JSON 必须包含：

| 字段 | 含义 |
|---|---|
| model_name | 实际执行的教师模型名称 |
| model_revision | 实际代码/模型 revision |
| checkpoint_sha256 | 实际教师权重的 SHA256 |
| source_matrix_sha256 | 对应 Factor Card provenance 中 `noisy_matrix_sha256`；这是带坐标的数值内容哈希，不是 clean reference hash |
| embedding_sha256 | 导出的 sample embedding NPZ 文件哈希 |
| pretraining_species_exposure | 教师预训练是否见过相关物种；无法查清就记录 unknown，不冒称 zero-shot |
| input_assay | 实际输入 assay，必须匹配 Factor Card |

`export-factor-teacher` 使用 discovery samples 上的 score–embedding 协方差形成 factor vector；held-out projection samples 不参与构造。用同一模型/权重计算两个物种，才能在同一表示空间比较。ESM2 与 UCE 的维度不强行拼成一个空间：每个教师有单独辅助目标，权重在验证后冻结。

模型身份及哈希依赖提供的真实来源记录；程序能核对文件一致性，不能从一份任意 NPZ 自动证明它确由某个模型生成。所有教师原始命令与日志应随远程实验保存。
