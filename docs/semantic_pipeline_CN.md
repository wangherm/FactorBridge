> 结果更新 2026-10-01：此流程已在 AutoDL 完成。完整语义微调模型对 56 个验证候选全部弃选，尚不建议作为最终恢复器。下面保留复现说明；实际结果见[实验记录](../experiments/README.md)。

# FactorBridge 功能语义实验运行说明

本轮复用已有公共数据转换、study/biological unit/parent factor 分组、数值候选、completion-only Qwen QLoRA、保存重载检查、W/Z 重估与独立研究评估。旧目录不覆盖。内部 killifish 不进入此入口。

## AutoDL 启动

在仓库根目录更新代码。若通过 ZIP 更新，解压覆盖代码即可，保留现有 `.venv`、`data`、`runs`；不要用旧代码启动新配置。

```bash
cd /root/autodl-tmp/FactorBridge-upload
screen -S factorbridge-semantic
bash scripts/run_semantic_pipeline.sh \
  runs/expanded_launcher_20260930T194955_6242/model_run/config.json
```

`Ctrl+A` 然后 `D` 脱离；`screen -r factorbridge-semantic` 返回。控制台开始时打印 `LOG_DIRECTORY`，可用 `tail -f <该目录>/console.log` 查看。Shell 使用 `set -euo pipefail`，任何阶段失败即停止，不会继续宣称训练成功。异常退出码也记录到 console.log。

官方功能注释需要访问 NCBI/Reactome。可以预先上传本地生成的 `data/annotations/ncbi_reactome_v1/bundle.json` 和 `bundle.lock.json` 到同一路径，避免服务器再次联网。现有 Qwen 权重从原来的 HF_HOME 缓存离线加载。

新证据卡增加了注释与候选基因，序列上限为 16384。真实 tokenizer 检查曾发现 9245-token 卡，因此明确提高预算；训练和推理预留空间均检查，不静默截断。启动脚本会自动执行一次这个配置的两步前后向/保存重载检查，随后正式训练；不复用旧 2048-token 配置的签名，也不要求反复手工跑测试。若内存不足、token 超长或重载不一致会明确停止。

## 已实现的数据流

1. 官方 NCBI Gene 信息与 Reactome gene–pathway 映射及功能描述下载，保存来源、时间和 SHA256。首次 SHA 为实际下载字节的指纹，不冒充发布方签名；缓存复用必须匹配。
2. 按物种精确映射符号及 Ensembl ID，支持现有矩阵里的 Ensembl 版本后缀、`Ensembl|symbol` 和重复 `symbol_symbol` 格式。歧义映射排除，未映射基因保留数值证据。RefSeq transcript 缺少可靠映射时报告缺失。
3. 每个 noisy view 单独计算富集；背景为全部测量基因，canonical gene 去重，BH 在所有合格条目上校正。候选卡保留最强 24 个基因，按功能与数值复现证据最多扩展到 40 个；最多展示六个功能条目。
4. Qwen 读取数值、基因描述、功能成员和富集证据，输出受支持的测量基因、决定、混杂标记、证据引用及 uncertain。训练目标仍来自已有数值弱参考，注释不是 gold，模型输出不反过来当监督。
5. 除原始 PCA 候选外，最多加入四个由 noisy 富集提出的子模块候选，每个子模块独立进行数值提取及按 biological unit bootstrap。所有选择方法共享扩展候选池并做相同 rank-1 支持集重估，输出真实 W/Z；原始 PCA 只保留原始候选作未重估诊断。候选预算固定，未实现无限制搜索。
6. Programme 层输出各样本的中心化基因秩均值、测量覆盖及缺失 mask；这是转录表达分数，不代表蛋白活性或通量。固定 Reactome vocabulary 用来源中 5–300 个基因的条目定义，不按验证集结果挑条目。
7. 跨物种按各自测量基因及保留 factor 的功能证据输出对应候选。相同功能名/重叠只构成可检查的候选关系，未作为跨物种身份真值，也未执行新的外部教师蒸馏。
8. 时间模块读取已审核的 GSE124109 饥饿天数、GSE132040 年龄月数。前者不叫休眠深度。其他数据明确报告无已审核时间 recipe。最后一部分时间点作未来留出，整个 biological unit 保持同侧。
9. 时间比较包含参考均值、最后观测状态、线性时间、二次参考曲线，以及“历史 programme 状态→低维状态漂移→递归预测未来”的 ridge 基线。进程估计器只从历史样本拟合；参考曲线使用实际已知时间。坐标推断与偏离检查剔除 anchor programme 所覆盖的基因重叠，但共享秩背景，不能声称完全独立的 biological clock 验证。
10. 条件偏离使用过去数据按生物学单位留一得到的预测残差尺度，输出描述性标准化偏离；残差 programme 因子仅在过去数据拟合，仍需独立复现。时间未来预测与跨研究 factor 恢复是不同评估，分开报告。

## 比较与产物

`evaluation_validation/report.json` 汇总九种方法：原始 PCA、loading refit、稳定性、同证据非 LLM（包含固定哈希的功能文字特征）、显式功能先验、未微调 Qwen、微调 Qwen，以及移除语义/打乱文字的同 adapter 推理干预。后两项不是重新训练的消融，不能据此声称已完成完整训练消融。

当前真实弱参考只有正例/未标记基因，非 LLM 基因分类器可能退化为常数；检查 `non_llm.json`，不要以胜过该项作为充分科学证据。所有方法都有候选池支持覆盖及参考支持重估 oracle。真实数据指标为弱参考一致性，受控真值才支持真正的恢复准确率主张。

- `annotation_audit.json`：实际映射覆盖；`audit.json`、`prepared/splits.jsonl`：文件及分组审计。
- `prepared/cards.jsonl`、`sft_text/`、`sft/`：输入与真实训练文本；reference/labels 只在 private sidecar。
- `smoke/`、`training/steps.jsonl`、`training/status.json`、`training/adapter/`：实际 GPU 日志与 adapter。
- `evaluation_validation/factors/*.npz`：W、Z_discovery、Z_heldout、基因与样本顺序、测量 mask。
- `evaluation_validation/factor_cards.jsonl`、`noise_curves.csv`：因子证据与噪声曲线。
- `programmes_train/`、`programmes_validation/`：A、coverage、时间分割、预测、参考尺度、偏离、残差 programme loadings/scores、跨物种功能对应候选及报告。

各方法 W 列仍分别 rank-1 拟合，可能相关；此版本不把它们称为联合最优分解。Programme 原始表达分数与恢复后的 factor 样本分数是不同产物。缺失时间、功能映射、可用 programme 或独立单位会明确标为 unavailable；没有伪造补齐。

## 世界模型参考与 agent 接口

这轮借鉴“状态→状态变化→未来状态”的设计，实际运行的是可审计的小型数值基线，尚未下载或运行外部世界模型。

- [Arc State](https://arcinstitute.org/news/virtual-cell-model-state)：单细胞状态表示与扰动后状态转移。扰动预测不等同于真实时间轨迹，bulk 和跨物种应用需另行验证。
- [VCWorld](https://arxiv.org/abs/2512.00306)：结合结构化知识及 LLM 迭代推理的细胞扰动模拟，可参考证据组织方法；当前没有启用它的自主推理循环。
- [Reactome 数据来源](https://reactome.org/download-data/)；[NCBI Gene 信息](https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/)。

`factorbridge.programmes.agent_tool(operation='programme_analysis', config=c, split='validation')` 是可调用的版本化工具入口。它调用当前确定性分析函数、返回产物目录，拒绝 test/internal_test，没有自主 agent、RL 或路由器。未来外层 agent 可经这个入口运行分析；不能读取内部最终测试来调规则。

另一个操作 `recover_expression` 接受 `source` 单数据 manifest、`destination` 新输出目录和 `method`。它从新表达矩阵直接提取/选择/重估因子，不构建参考标签。也可从命令行调用：

```bash
python -m factorbridge recover --config runs/<本轮目录>/config.json \
  --source configs/new_public_dataset.json --destination runs/new_public_recovery \
  --methods qwen_finetuned
```

单数据 manifest 遵循已有 datasets 格式，必须提供实际 matrix/metadata、物种及 biological unit；探索入口只接受 public_real、public_train/public_validation，最终 killifish 应走冻结评估。`factors.npz` 包含 W、按原样本顺序排列的 Z、中心值和 discovery/heldout 分组；`factor_cards.jsonl` 保留数值/功能证据与选择结果。至少八个独立生物学单位，不能用单个样本建立新因子。

## 本轮科学边界

现有弱监督仍不能证明模型已学会生物机制、休眠特异性、真实负例或因果混杂。下一步依据这轮真实输出决定是候选覆盖、监督还是模型选择需要改进。训练 loss 下降不等于 biological factor 恢复成功。外部世界模型输出将来只能是带来源的预测或辅助监督，不能充当 gold。
