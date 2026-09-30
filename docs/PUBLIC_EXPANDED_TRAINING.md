# 公共数据扩展训练（v2）

用户本轮决定：扩大公共表达数据训练；内部 killifish 保留最终 frozen test；不再重复测试套件或 GPU smoke。模型仍为固定 revision 的 Qwen3-4B-Instruct-2507，沿用已经实测通过的 QLoRA 配置。

## 直接运行

在 AutoDL 上传本次 `FactorBridge_public_expanded_update.zip` 到 `/root/autodl-tmp/`。若上一轮 16 卡训练仍运行，等它结束再启动本轮；不要在同一张卡上同时启动两个训练。

```bash
cd /root/autodl-tmp/FactorBridge-upload
.venv/bin/python -m zipfile -e /root/autodl-tmp/FactorBridge_public_expanded_update.zip .
screen -dmS factorbridge_expanded bash scripts/run_public_expanded.sh \
  runs/pretraining_20260930T173002_680253Z/config.json
screen -r factorbridge_expanded
```

按 `Ctrl+A`，松开后按 `D` 离开 screen，后台继续运行。重新连接用 `screen -r factorbridge_expanded`。没有安装 screen 时先执行 `apt-get update && apt-get install -y screen`。

更新只包含代码、配置和说明，不覆盖 `.venv`、已有 runs、模型权重、tokenizer 或数据。无需重新安装/升级训练依赖；改变依赖可能使原 smoke 记录不再适用。

脚本执行：全量 GEO 补充文件下载（4 个并发研究）→ 与下载独立地准备已审计的公共矩阵 → 分组提取候选及弱标签 → 正式 SFT。日志直接打印每个 optimizer step 的真实 loss 和每轮 validation loss。不会调用 `unittest`、`smoke`、独立 benchmark、最终 killifish test、Qwen 生成标签或教师蒸馏。

## 已接入训练的数据

| 研究 | 模型 split | 表达样本数 | 分组与边界 |
|---|---|---:|---|
| GSE193258 | train | 60 | 同细胞系、同 replicate 跨处理分组 |
| GSE113957 | train | 143 | 按公开 cell ID；用户调整范围后由预留 public test 转为 train |
| GSE124109 | train | 30 | 复用已有真实 rat 数据转换；culture lineage 局限继续保留 |
| GSE132040 | train | 933 | 17 个组织面板是同一个研究；同年龄/性别保守 cohort 分组，不能将文库数称作动物数 |
| GSE189073 | train | 30 | 同终点 cohort 的重复一起分组；三个子实验取共同测得基因，不补零 |
| GSE255958 | train | 82 | 同细胞系 replicate 跨处理分组；PDX/organoid 按患者模型整体分组 |
| GSE63577 | validation | 30 | 复用原验证集；JenAge 相关重分析不能另作独立训练来源 |
| GSE167665 | validation | 63 | 16 只动物，所有组织跟随 deposited `individual`；作者已统一标准化，局部投影不能声称独立预处理验证 |

总计 8 个研究、1,371 个表达样本。该表来自本地真实矩阵转换；不代表 AutoDL 已完成下载、生成全部卡或训练。公共数据 train/validation 的作用分别是拟合与选择 checkpoint；你的 killifish 不在训练 manifest 中。

GSE132040 的 14 个 `tissue=NA, age=NA, sex=missing` 文库明确排除。其 source name 数字后缀不能确认是动物 ID，因此不按这个后缀虚构动物身份。GSE189073 的三个文件保留 53,086 个共有基因，排除项写入 `gene_intersection.json`。所有数据集在自己的表达矩阵上提取候选，合并的只是证据卡，不拼接不同物种的原始基因矩阵。

## 全量目录下载不等于全部可训练

`configs/public_catalogue_v2.json` 保留用户目录的 69 条记录和 54 个 GEO accession。`download_public_catalogue.py` 下载每个 GEO 系列的 SOFT 与所有已公开 supplementary files，包括混合模态；不从 accession 推断已到位，不把 FASTQ、ATAC、Hi-C 或 proteomics 文件作为 gene expression 输入。

全量 GEO 下载没有默认单文件大小上限，可能需要较大磁盘空间。下载进度与错误写入 `data/public/catalogue_archive_v2/acquisition_report.json`；单文件有校验和 receipt，可重用完整文件。失败文件会重试 3 次，不能下载的文件保留错误，不生成成功标记。当前不实现 HTTP 断点续传，未完成的 `.part` 会从头重试。

**目前不是全目录都已接入训练。** 缺少独立样本、只有条件均值、细胞缺少 donor/pool 映射、非表达模态、受控访问，以及尚未完成可靠转换的文件，都必须保留在 acquisition/admission 状态中。不能因为用户希望扩大训练，就把这些文件静默当作有效独立训练数据。非 GEO 来源和仅文献指针也会列为待仓库专用处理；下载脚本退出码 2 表示明确未完成，不是训练成功或失败的替代状态。

重新尝试下载：

```bash
python -u scripts/download_public_catalogue.py \
  --destination data/public/catalogue_archive_v2 --workers 4
```

主训练只接受 `configs/public_expression_recipes_v2.json` 中已审计、固定 SHA256 的转换配方。配方变更必须建新 data 目录和新 run，不能覆盖已有训练 lineage。缺失/校验失败直接报错，不用模拟数据回退。

## 复用真实 GPU smoke 的条件

新训练配置保存 `reuse_smoke_config`，读取原 run 的实际 `smoke/status.json`、原配置、prepared 文件哈希、环境、adapter 配置和保存重载诊断。只有原 smoke 签名可验证，GPU/依赖与模型训练配方一致，且重载诊断确实通过，才允许沿用。v2 显式兼容 ad51e79 的已实测模型实现代码哈希；没有生成新 smoke pass 或把新数据写进旧 smoke 目录。

训练证据记录在新 run 的 `training/gpu_smoke_evidence.json`。如果核对不一致，会明确停止；不要手改 `status.json`、哈希或删掉保护来继续。新数据数量、研究和噪声视图不改变模型加载实现；token 长度和有限 loss/梯度仍在训练过程中检查。

## 输出与实际状态

- `runs/expanded_launcher_*/console.log`：准备与实际训练控制台日志。
- `runs/expanded_launcher_*/catalogue_download.log`：目录下载日志。
- `runs/expanded_launcher_*/model_run/config.json`：这轮完整配置。
- `model_run/prepared/`：证据卡、仅 sidecar 可见的弱标签、splits、noisy 数值矩阵与 lineage。
- `model_run/training/steps.jsonl`：实际 optimizer steps 和 validation loss。
- `model_run/training/adapter/`：按 validation loss 选择的真实 adapter。
- `model_run/training/status.json`：实际训练状态；只有 `completed` 表示本轮训练完成。

训练成功仅指训练程序完成。弱标签仍不是因果 biological factor gold；未加入受控真值的 null/confounding 标签。此轮多物种证据卡训练也不等于已证明跨物种 identity 或完成多教师蒸馏。W/Z、基线比较、噪声恢复曲线与最终内部 test 仍应在训练后按既有数值重估与冻结流程实际执行，不能提前报告结果。
