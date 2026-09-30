# 已通过 GPU runtime 检查后：固定权重与 Qwen smoke

用户实际日志已确认 RTX PRO 6000 Blackwell Server Edition，compute capability 12.0；torch 2.10.0+cu128、Transformers 4.57.6、PEFT 0.18.1、Accelerate 1.12.0、bitsandbytes 0.49.2、huggingface-hub 0.36.2 的基础 BF16/NF4 前后向检查通过。这不是 Qwen 模型 smoke 或正式训练通过记录。

## 权重来源与固定版本

使用 Qwen 官方 ModelScope 仓库下载，固定 ModelScope commit `2de2439ea21be1dc5cb21f22f88af07e43393cbb`。11 个文件的 size/SHA256 已与 Hugging Face commit `cdbee75f17c01a7cc42f958dc650907174af0554` 核对：大文件使用 HF LFS 发布的 SHA256，小文件使用实际下载的固定 HF 版本内容计算 SHA256。源清单在 `configs/qwen3_4b_snapshot.json`。

三个 safetensors 权重分片约 8.045 GB，加上配置、索引和 tokenizer 约 8.06 GB。下载程序采用两条线程并行传输，HTTP 断点续传，逐文件校验哈希；连接失败最多重试三次，部分文件保留。若服务器忽略 Range，会明确输出重下提示。哈希不符直接失败，不接受替代模型或静默替换损坏缓存。

校验成功的原始字节存入 `$HF_HOME/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/<HF revision>/`。这是本地缓存导入，不改变上游代码或模型内容；原配置中的 model ID/revision 保持不变。后续用 HF 离线模式加载，避免再次访问不可达的 Hugging Face。无需安装 ModelScope SDK，也无需登录账号。

官方来源：[HF 固定版本](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/tree/cdbee75f17c01a7cc42f958dc650907174af0554)、[ModelScope Qwen 仓库](https://modelscope.cn/models/Qwen/Qwen3-4B-Instruct-2507)、[Transformers 离线加载](https://huggingface.co/docs/transformers/installation#offline-mode)。

## AutoDL 执行

无法 pull GitHub 时，将聊天提供的 `FactorBridge_qwen_smoke_update.zip` 上传到 `/root/autodl-tmp`。该包只含脚本、说明和校验清单，不含权重或训练产物。在 Terminal 解压：

```bash
cd /root/autodl-tmp
python -m zipfile -e FactorBridge_qwen_smoke_update.zip FactorBridge-upload
```

大文件下载应在 `screen` 会话内运行；新建会话用 `screen -S factorbridge-smoke`。然后运行：

```bash
cd /root/autodl-tmp/FactorBridge-upload
bash scripts/smoke_qwen_offline.sh runs/pretraining_20260930T173002_680253Z/config.json
```

其他 run 使用其实际 config.json 路径。按 Ctrl+A 后按 D 离开 screen，重连使用 `screen -r factorbridge-smoke`。同一时刻只运行一份下载/smoke，不并发写相同缓存。

先验证原 prepared 文件、配置、准备报告和当前 CUDA，再下载模型。下载成功后复用现有 `python -m factorbridge smoke`：真实加载 Qwen + QLoRA，验证仅 adapter 参数可训练，使用训练集执行两次优化、检查有限且非零梯度，保存 smoke adapter，释放模型、重载，并比较 logits。smoke 不执行 validation/test 评估，也不选择模型。

这是带优化步骤的工程 smoke，产生的 adapter 位于 `smoke/adapter`，不是正式训练的 `training/adapter`，不可据此报告性能提升。脚本结束后明确停止，不调用 `train`；当前仅 16 条训练卡的真实弱监督仍不足以验证完整科学目标。

## 查看结果或恢复失败

出现 `Save/reload logits differ` 时，按 [QLoRA 重载精度修复](SMOKE_RELOAD_FIX.md) 保留失败现场并重跑；不降低通过阈值。

- 启动器日志：`runs/qwen_smoke_launcher_<时间>_<PID>/console.log`。
- 权重下载/校验记录：`$HF_HOME/factorbridge_downloads/<时间>.json`。
- 模型 smoke 状态：原 run 的 `smoke/status.json`；成功必须为 `passed`，包含 reload delta、实际显存峰值和优化步数。
- 模型步骤日志、参数、环境与版本：原 run 的 `smoke/steps.jsonl`、`smoke/trainable_parameters.json`、`smoke/environment.json`、`smoke/requirements.resolved.txt`。

下载期间失败且尚未创建 `smoke/` 时，可重跑同一命令，校验缓存并续传。若已有 `smoke/`，脚本拒绝覆盖，应保留现场并查看失败日志；不要手动把 status 改为 passed。旧版 smoke 实现中异常可能留下 `status: running`，必须结合启动器 traceback 和 `commands.jsonl` 判断，只有实际 `passed` 才能进入后续阶段。

单独检查下载文件（不加载模型）：

```bash
source .venv/bin/activate
python scripts/download_qwen_weights.py --hf-home /root/autodl-tmp/huggingface --verify-only
```

`--metadata-only` 仅供开发诊断，会显式标记 `metadata_only_not_ready`，绝不能当作权重齐备。本地 Windows 只核验过官方清单、小文件真实下载、HF 缓存中的离线 tokenizer/配置解析以及下载器测试；8 GB 权重传输和完整 Qwen GPU smoke 须以 AutoDL 的真实结果为准。
