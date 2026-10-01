# 文档索引

## 当前阅读顺序

1. [项目说明](PROJECT_DESCRIPTION_CN.md)：面向计算机方向合作者，解释任务、方法、实验与讨论问题。
2. [实验记录](../experiments/README.md)：E01/E02 实际结果和源包指纹。
3. [项目状态与讨论决策](PROJECT_STATUS.md)：区分完成、失败、未验证和待研究部分。
4. [数据契约](DATA_CONTRACT.md)：矩阵、元数据、split 和证据边界。
5. [语义实验运行说明](semantic_pipeline_CN.md)：复现 E02；其模型失效结果已归档。

## 历史操作说明

以下文件保留原路径，便于旧链接、AutoDL 命令与排错记录继续使用。它们记录当时流程，当前进展以项目概览和实验记录为准。

| 阶段 | 文件 |
| --- | --- |
| 公共数据起步 | [PUBLIC_PILOT](PUBLIC_PILOT.md) |
| 训练前准备 | [BEFORE_TRAINING](BEFORE_TRAINING.md) |
| 离线 tokenizer | [OFFLINE_TOKENIZER](OFFLINE_TOKENIZER.md) |
| GPU smoke | [QWEN_SMOKE](QWEN_SMOKE.md) |
| 保存重载精度修复 | [SMOKE_RELOAD_FIX](SMOKE_RELOAD_FIX.md) |
| ZIP 签名兼容修复 | [SMOKE_SIGNATURE_ZIP_FIX](SMOKE_SIGNATURE_ZIP_FIX.md) |
| 公共数据扩展训练 | [PUBLIC_EXPANDED_TRAINING](PUBLIC_EXPANDED_TRAINING.md) |
| AutoDL 与实验性 Stage 2 操作 | [AUTODL](AUTODL.md) |

[归档说明](archive/2026-10-01.md)记录代码快照、源包、已知限制和后续维护方式。实验摘要公开，原始数据与模型权重不提交 Git。
