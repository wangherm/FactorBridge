> 历史操作说明。当前实际训练与评估结论见[实验记录](../experiments/README.md)，文档导航见[索引](README.md)。

# Hugging Face 不可达时完成训练前检查

网络异常发生在 `tokenizer_check`，应先保留现有 run。无需重新下载表达数据、重新提取候选或再次评估基线；不能把跳过 tokenizer 当作检查通过。

离线包使用已实际验证的 `Qwen/Qwen3-4B-Instruct-2507` tokenizer，revision 为 `cdbee75f17c01a7cc42f958dc650907174af0554`，由 Transformers 4.57.6 的 `save_pretrained` 导出，带官方 LICENSE、来源与每个文件的 SHA256。不含模型权重、训练结果或 AutoDL 的状态文件。文件放在 `assets/qwen3_4b_instruct_2507_tokenizer/`；这些大文件不会提交到 Git。

从聊天附件下载 `FactorBridge_offline_tokenizer_fix.zip` 并上传到 AutoDL `/root/autodl-tmp`。包中有 tokenizer 和对应的小范围代码更新，解压到此前的源码目录，然后执行：

```bash
(
  set -euo pipefail
  cd /root/autodl-tmp
  python -m zipfile -e FactorBridge_offline_tokenizer_fix.zip FactorBridge-upload
  cd FactorBridge-upload
  source .venv/bin/activate
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  python scripts/finish_pretraining_offline.py --tokenizer-dir assets/qwen3_4b_instruct_2507_tokenizer
)
```

该脚本只自动选择唯一一个 tokenizer 失败或显式跳过的 run；有多个时列出候选并停止，需加 `--run-dir runs/实际目录名`。它校验原配置与 prepared 文件哈希、训练/验证 lineage、四项数值基线报告和本地 tokenizer 的 model/revision/文件哈希。已完成、已 smoke/训练或已冻结的 run 不可续写。

本地加载强制 `local_files_only=True`，不尝试访问 Hugging Face；仍调用正式训练的编码逻辑进行长度、模板与 mask 检查。补查成功后更新 `pretraining_report.json` 和 `STOP_BEFORE_TRAINING.json`。`tokenizer_resume_<时间>/` 保存本次真实状态、环境、依赖和旧报告备份；原 `PREPARATION_FAILED.json` 作为历史失败保留，新报告注明已补查。原数值代码哈希与此次代码哈希分别记录，不改写之前的数值报告。表达矩阵、split、标签、配置和 W/Z 不改变。

最终应看到 `pretraining_checks: passed`、`qwen_training_executed: false`。如有其他 blocker，照实保留 blocked。此包仅解决 tokenizer 的离线准备；不代表模型权重已下载，也不代表 GPU smoke 或微调已通过。

今后新建 run 可在原准备命令后加 `--tokenizer-dir assets/qwen3_4b_instruct_2507_tokenizer`，显式复用离线 tokenizer。官方离线机制见 [Transformers 文档](https://huggingface.co/docs/transformers/installation#offline-mode)。
