# 扩展训练的 smoke 签名兼容修复

扩展 run 已完成 PREPARE 并输出 TRAIN_CONFIG 后，在旧 smoke 签名检查处停止。失败发生在加载训练模型、创建 training 目录和 optimizer step 之前。不要重新运行全量启动脚本，不需要重新生成训练卡。

发现的兼容遗漏：初始 `FactorBridge_source.zip` 保留部分 CRLF 文件，后续离线 tokenizer 与 precision fix 的增量 ZIP 使用 LF。现有签名按文件原始字节计算；上次兼容列表只包含全 LF 的代码哈希 `34dbe91c...`，没有包括实际增量 ZIP 组合的 `8672b3c2...`。

本次从原始三个交付 ZIP 重新构建了该组合，并比对 Python AST：模型加载、步进、保存、smoke 和签名算法与原全 LF 实现一致。修复仅加入这个明确核实的旧分发版本，保留原始数据、配置、环境、保存重载诊断及 adapter 配置校验。不会修改旧 smoke/status.json，不会伪造新 smoke 成功，不会跳过签名检查。

将 `FactorBridge_smoke_signature_fix.zip` 上传到 `/root/autodl-tmp/`，执行：

```bash
cd /root/autodl-tmp/FactorBridge-upload
.venv/bin/python -m zipfile -e /root/autodl-tmp/FactorBridge_smoke_signature_fix.zip .
screen -dmS factorbridge_resume bash scripts/resume_prepared_training.sh \
  runs/expanded_launcher_20260930T194955_6242/model_run/config.json
screen -r factorbridge_resume
```

如果仍出现签名错误，新 run 的 `smoke_compatibility_failure.json` 会列出实际签名与核对的候选签名；保留它继续定位，不能通过改状态文件继续。修复尚未在用户 AutoDL 上运行，训练是否启动以真实日志为准。

## 下载进程单独检查

上传的 catalogue_download.log 最后为 SERIES 37/54，没有最终完成报告，不能据此判断现在的进程状态。执行：

```bash
pgrep -af '[d]ownload_public_catalogue.py'
```

若仍有对应下载进程，不要重复启动。若没有输出，在独立 screen 中继续下载：

```bash
screen -dmS factorbridge_download bash -lc 'cd /root/autodl-tmp/FactorBridge-upload && .venv/bin/python -u scripts/download_public_catalogue.py --destination data/public/catalogue_archive_v2 --workers 4 >> runs/catalogue_download_resume.log 2>&1'
tail -n 20 -f runs/catalogue_download_resume.log
```

完整文件按 receipt 校验后复用；未完成大文件当前从头重试。训练使用已经准备好的数据，因此这个下载进程与继续训练独立。GEO 目录不存在等错误会明确记入 acquisition_report.json，不能把下载退出视为全部数据成功纳入训练。
