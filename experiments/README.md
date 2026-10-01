# 实验记录

归档日期：2026-10-01。统计来自用户返回的真实 AutoDL 运行包，不来自模拟展示或 LLM 自报。两轮都使用 Qwen/Qwen3-4B-Instruct-2507，固定 revision `cdbee75f17c01a7cc42f958dc650907174af0554`，NF4 QLoRA、BF16 compute、LoRA rank 16、learning rate 1e-4、3 epochs，completion-only SFT。

## 数据与分组

实际纳入 8 个研究、24 个面板、1,371 个样本。训练研究为 GSE113957、GSE124109、GSE132040、GSE189073、GSE193258、GSE255958；验证研究为 GSE167665、GSE63577。分别为 1,278 和 93 个样本。输入均为明确转换的 log expression；噪声为 0、0.5、1、2 倍 gene-wise SD 的加性噪声，非零噪声另屏蔽 10% 基因。

卡片是候选及其噪声版本，不是独立样本。所有噪声版本按原研究/单位/parent factor 留在同一 split。上述两次实际训练全部使用真实数据弱参考；代码中的受控模拟真值没有加入这两轮 SFT。内部 killifish 未使用。目录列出的 accession 或已下载原始文件不自动等于训练数据。

## E01 数值弱监督

[机器可读记录](2026-10-01/E01_numerical_weak_sft.json)。352 个训练卡、32 个验证卡，3 epochs、66 optimizer steps，训练循环约 470 秒。最佳 validation token loss 为 0.11653。该运行包未提供此 adapter 的独立 W/Z 恢复比较，因此不能把 loss 下降解释为因子恢复成功。

## E02 功能语义弱监督

[机器可读记录](2026-10-01/E02_semantic_weak_sft.json)。542 个训练卡、56 个验证卡，3 epochs、102 optimizer steps，训练循环约 5,877 秒，峰值 PyTorch allocated memory 50.44 GiB。真实加载、前后向、保存重载检查通过，reload logits 最大差为 0。最长样本 9,433 tokens；未截断、未丢弃。

新增功能注释、富集证据、24–40 个可见基因和至多四个 annotation-guided 子模块候选。训练监督仍由原来的 PCA 弱参考产生。最佳 token loss 在 epoch 2 为 0.07629；epoch 3 为 0.08324，按现有代码保存最佳 epoch 的 adapter。权重本体未包含于报告包；包内清单记录了文件大小和 SHA256，不能据此声称已独立重载上传权重。

### 九项方法结果

恢复率按研究汇总，再对两个研究平均；参考为数值弱参考，loading cosine 阈值为 0.7。候选保留数为原始 56 个候选中的 retain 数，未经过重估去重。两列分母不同。

| 方法 | 弱参考恢复率 | 保留候选 |
| --- | ---: | ---: |
| pca_raw | 0.0% | 32 / 56 |
| loading_refit | 41.7% | 56 / 56 |
| stability | 41.7% | 49 / 56 |
| non_llm | 29.2% | 13 / 56 |
| semantic_prior | 20.8% | 42 / 56 |
| qwen_frozen | 8.3% | 24 / 56 |
| qwen_finetuned | 0.0% | 0 / 56 |
| qwen_no_semantics | 41.7% | 52 / 56 |
| qwen_shuffled_text | 0.0% | 0 / 56 |

完整语义的微调 Qwen 和文字打乱版本都输出 56 个 identical uncertain / empty-support JSON；八个 view 的 W 均有零列。去掉语义后，52 个 retain 中 50 个选择了全部可见基因；这一结果不支持精细语义选择已成功。

原始 PCA 为未重估诊断，参考是稀疏重估因子，0% 不表示 PCA 完全没有表达信号。其余选择方法共享 noisy 输入、扩展候选池与支持集 rank-1 重估。非 LLM 的 gene classifier 因正例/未标记监督退化为 constant-one；它不是充分强的 gene selection 基线。

### 失败证据与解释边界

- 190 个新增 programme 训练候选全部 uncertain；24 个新增验证候选也全部 uncertain。新增候选没有获得任何 retain 监督。
- 训练集无 gene-function 注释的 16 个例子全部 retain；有注释的 526 个例子为 143 retain、383 uncertain。
- GSE132040 占 424/542 个训练卡，约 78.2%，但仍只是一个研究。
- 低噪声时，参考支持交集经过同一数值重估可达到 100% 恢复；完整语义 Qwen 仍全部弃选。最高噪声下该 oracle 对照也为 0，候选/数值恢复本身存在边界。

这些发现支持重审监督与选择环节；“模型学到注释存在性、结构或长度的捷径”仍是待验证解释。移除语义同时改变了多项输入因素，不能单凭它判断生物语义的因果贡献；两种消融均为同一 adapter 的推理干预，没有单独训练。

### Programme 与时间分析

训练侧输出 88 个 view，验证侧输出 8 个 view 的 programme 分数/覆盖。GSE113957 RefSeq 映射缺失，其 programme 为不可用而非生物学零表达。验证输出 21 条跨物种功能对应候选，均不能作为已验证 identity；完整语义微调模型没有贡献对应因子。

时间分析只在训练研究执行：GSE124109 的 4 档噪声中，latent state transition 仅 1 档优于参考均值；GSE132040 的 68 个组织/噪声组合中仅 11 个更好。这些组合不能当作独立研究。两个验证研究没有已审核时间 recipe，因此没有独立验证研究的时间预测结果。外部世界模型未运行。

## 证据与后续

源包 SHA256、配置对应签名、真实训练状态和指标保存在上述 JSON。公开记录只含汇总，完整日志和 adapter 保留于原运行目录。E02 的 100 个包内 prepared 文件已逐一与索引核对哈希，lineage 分组检查通过；这不替代对生物学标签的验证。

建议的下一轮设计见[项目状态与讨论决策](../docs/PROJECT_STATUS.md)。本次归档不启动新的训练、不改变任何既有实验结果。
