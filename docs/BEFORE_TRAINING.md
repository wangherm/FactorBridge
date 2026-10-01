> 历史操作说明。当前实际训练与评估结论见[实验记录](../experiments/README.md)，文档导航见[索引](README.md)。

# 下一步：公共数据准备，到 Qwen 训练之前停止

已跑过 GSE124109 单研究 pilot 后，执行本页。新流程复用原有 candidate/card/refit/evaluate 实现，建立独立 study 的训练、验证和预留测试集。它只完成 **Stage 1 的小规模真实弱监督准备**；不加载 Qwen 权重、不做 GPU smoke、不微调、不运行跨物种教师，也不冻结或评估测试集。

## 在 AutoDL JupyterLab 的 Terminal 中整段执行

以下沿用之前的仓库位置。若你放在其他位置，只改 `cd` 那一行。使用子 shell，某步失败便停止。公开仓库 pull 不需要账号或密码。

```bash
(
  set -euo pipefail
  cd /root/autodl-tmp/FactorBridge
  git pull --ff-only
  test -f pyproject.toml
  test -x .venv/bin/python || python -m venv .venv
  source .venv/bin/activate
  python -m pip install -e '.[prepare,tokenizer]'
  python -m unittest discover -s tests -v
  export HF_HOME=/root/autodl-tmp/huggingface
  export OPENBLAS_NUM_THREADS=2
  export OMP_NUM_THREADS=2
  bash scripts/prepare_before_training.sh
)
```

Notebook 用户可将整段放在同一个 `%%bash` cell；不要分开使用 `!cd` / `!source`。

无需为这一步安装 PyTorch，也无需运行 `.[train]`。tokenizer 加载时提示没有 PyTorch/TF/Flax 是预期的；出现异常退出则不是成功。脚本会检测并记录实际环境，但不会声称已验证 CUDA 计算。

## 会执行什么

1. 从 NCBI 下载固定文件，核验 SHA256，精确对齐矩阵列和 GSM/BioSample 元数据，写出矩阵、metadata、provenance 和 manifest。缓存文件也需校验；不默许来源变化。
2. 审计 study、biological unit、parent lineage 和固定 split。按完整培养组划分每个开发研究的 discovery/held-out projection。
3. 仅为训练和验证研究生成噪声版本、PCA 候选和 noisy-only Evidence Cards。弱参考与标签保存在 `private/`，不进入 prompt。
4. 导出训练/验证 SFT 文本，审计支持基因、retain/uncertain 分布。只在训练集拟合非 LLM selector；运行四种数值方法的验证集诊断，输出实际 W/Z、Factor Cards、候选支持覆盖上限、oracle-refit 和噪声曲线。
5. 下载固定 Qwen revision 的 **tokenizer 文件**，使用实际训练编码函数检查 chat template、completion-only mask 和 2048-token 限制。超长直接失败，不截断、不丢弃样本。保存 tokenizer 哈希及真实依赖版本。
6. 写出 `pretraining_report.json` 和 `STOP_BEFORE_TRAINING.json`，停止。

数据目录：`data/public/pretraining_panel_v1/`。每次运行自动建立新的 `runs/pretraining_<UTC时间>/`，不覆盖旧 pilot 或实验；有效数据缓存可以复用。

## 固定公共数据划分

| Split | 官方研究 | 实际矩阵样本 | 分组与变换 |
|---|---|---:|---|
| Train | [GSE193258](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE193258) | 60 | 4 种 NSCLC 细胞系，按细胞系 × 重复编号合并为 12 组；同组处理与 washout 时间点一起留出。estimated counts → log1p(CPM)。 |
| Validation | [GSE63577](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63577) | 30 | 5 种成纤维细胞系，按细胞系 × 重复编号合并为 15 组；年轻/衰老培养一起留出。只用发布工作簿的 counts sheet → log1p(CPM)。 |
| Reserved public test | [GSE113957](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE113957) | 143 | 按作者发布的 cell ID 分组；log1p(FPKM)，RefSeq 转录本 ID 保留为 gene proxy。只下载、转换、审计，不生成候选、参考标签或评估结果。 |

GSE63577 系列的其余 18 个样本不在本次 counts 文件中，明确排除。该研究含重分析数据，相关 GSE64553 等 accession 不能当作额外独立 study。培养组不是独立人类 donor；不同研究保留各自基因注释和定量流程，本批不证明跨平台能力。以上角色在数值结果产生前固定，不按结果重新分配。

内部 killifish 不在本次 manifest 中，其 frozen-test 边界不变。本批是同物种 Stage 1 准备；用户授权的跨物种、多教师比较仍需另备实际兼容数据、权重与标签，不会用这批 bulk 表达假装跑过 UCE 等单细胞模型。

## 弱参考修正与限制

原 pilot 将稠密 PCA loading 与仅保留少数基因的参考直接比较，二者几何不一致，可能把可匹配候选全部判成 uncertain。新 run 显式启用 `dense_identity_sparse_refit_v2`：用 discovery 的稠密稳定方向匹配 identity；在参考 top-16 基因中保留 bootstrap recurrence ≥ 0.5 的支持，再用同一支持受限数值流程重估参考 W/Z。原有 factor stability ≥ 0.75、matching cosine ≥ 0.6、最少 3 个支持基因均未降低。旧 pilot 配置与历史结果不改写。

参考只服务于监督与评分；输入仍来自 noisy discovery，含缺失 mask，held-out 表达不用于选支持基因。真实弱参考中的未标记基因不是已知负例，稳定方向也不等于因果 biological factor。原始 PCA 对稀疏参考的分数存在表示差异，应结合共用数值重估器的方法比较和候选池覆盖来读。

本批仅每个 split 一个 study，rank 4 × 4 个噪声级别通常只产生 16 张卡；噪声版本不能被算作独立研究。没有可靠 null、技术混杂或细胞组成 gold。`pretraining_checks: passed` 只表示本页的数据与 tokenizer 检查通过，**不表示完整科学目标已满足，也不表示模型训练成功**。补充受控真值及更多独立研究后，才适合检验完整的拒绝/混杂目标和泛化结论。

## 跑完给我哪些文件

终端最后会打印准确的 `run_dir` 和报告路径。把该目录中的以下文件发回来：

- `pretraining_report.json`：准备状态、弱标签数量、未完成环节。
- `environment.json`：GPU/驱动/依赖实测。
- `evaluation_validation/report.json`：数值基线诊断；Qwen 两项会明确标记未执行。
- 若失败：`PREPARATION_FAILED.json` 和 `console.log` 最后约 80 行。

完整结果还包含 `audit.json`、`prepared/splits.jsonl`、`prepared/private/lineage.jsonl`、`sft_text/`、`sft/lengths.json`、`tokenizer_check.json`、`requirements.resolved.txt`、`pretraining_commands.jsonl` 及 `evaluation_validation/factors/` 下的 W/Z。

tokenizer 网络不可达时保留错误，可按 [离线 tokenizer 补查](OFFLINE_TOKENIZER.md) 接着已有 run 完成检查，无需重跑数值流程。若只想完成数据诊断，可显式改用 `bash scripts/prepare_before_training.sh --skip-tokenizer`；这种运行的最终状态一定为 `blocked`，不能当作 tokenizer 已通过。SHA256 不一致则先调查来源，不能靠修改校验值绕过。

本页结束后不要接着运行旧文档中的 `train`。下一步先审查这些报告，再单独安装匹配实际 GPU 的依赖，实际完成加载、前后向、保存重载 smoke，随后才进入训练。正式测试只在冻结后通过 `prepare-public-test` 打开；本脚本不会调用它。
