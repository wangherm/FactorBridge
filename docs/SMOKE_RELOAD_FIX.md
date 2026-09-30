# QLoRA 保存／重载 logits 不一致

AutoDL 的真实 smoke 已完成模型加载、两步优化、adapter 保存和第二次加载，但最终比较失败：`Save/reload logits differ: 1.479004979133606`。这不是 smoke 通过，不能进入正式训练。

代码审计发现训练与重载精度策略不一致：训练分支调用 `prepare_model_for_kbit_training`，PEFT 0.18.1 会将非 `Params4bit` 的 FP16/BF16 参数转成 FP32；重载评估分支没有调用该函数，保留 BF16。因此嵌入、归一化和输出层等可能使用不同精度。这是明确的实现缺陷，也是当前差异的主要怀疑原因；是否足以解释全部差异，要以修复后的实际 GPU 重跑为准。[PEFT 0.18.1 源码](https://github.com/huggingface/peft/blob/v0.18.1/src/peft/utils/other.py)

修复内容：

- 所有 4-bit 路径，包括训练、重载 adapter 和未微调 Qwen 对照，使用相同 FP32 非量化参数准备；只对训练开启 gradient checkpointing。
- 显式设置 `use_reentrant=False`，使用当前 Transformers 的 `dtype` 参数。依赖下限同步为当前已实测安装的 Transformers 4.57.6。
- 保存时明确不保存未修改、未训练的 embedding，避免离线环境下 PEFT 自动检查词表产生网络警告。本实现不调用词表 resize。
- 比较前后都关闭 KV cache。原 `rtol=0.01`、`atol=0.05` 完全不变。
- 增加参数结构／dtype 清单、量化计算 dtype 清单、LoRA tensor 的逐项 SHA256，以及重载前后 logits。只有参数精度一致、adapter 字节一致且 logits 原阈值通过，才能写 `passed`。

## AutoDL 重试

把 `FactorBridge_reload_precision_fix.zip` 上传到 `/root/autodl-tmp`。在已有 screen 会话里执行：

```bash
(
  set -euo pipefail
  cd /root/autodl-tmp
  python -m zipfile -e FactorBridge_reload_precision_fix.zip FactorBridge-upload
  cd FactorBridge-upload
  bash scripts/retry_qwen_smoke.sh runs/pretraining_20260930T173002_680253Z/config.json
)
```

重试入口只归档 `status=failed` 的旧 `smoke/`，完整保留在原 run 下的 `smoke_attempts/failed_<UTC>/`，包括旧 adapter、loss、状态与归档收据。不移动运行中或已通过的 smoke；不更改原配置、数据、split、弱标签或数值评估。随后从同一原始 Qwen 权重重新跑两步工程 smoke，不复用失败 attempt 的 adapter。模型缓存会重新校验，哈希正确的文件不会重新下载。

已安装的 Transformers 4.57.6、PEFT 0.18.1 等依赖无需重装。本地只完成加载分支回归测试和归档保护测试，未执行实际 Qwen GPU 重载。正式训练仍不会启动。

成功后请提供原 run 下的 `smoke/status.json` 和 `smoke/reload_diagnostics.json`。若仍失败，也保留这两个文件以及启动器日志；新的 `before_reload_fingerprint.json`、`after_reload_fingerprint.json`、`reload_logits.npz` 用于进一步定位，不能放宽阈值伪装通过。
