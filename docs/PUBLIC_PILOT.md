# 公共数据起步与 JupyterLab 操作

本地先准备代码和公开数据，AutoDL 上由用户自行拉取和下载。无需从 Windows 传输 GitHub 登录凭据。

## 克隆失败的原因

`FactorBridge` 是私有仓库。GitHub 的 HTTPS Git 操作已不接受账号密码；出现 `Password` 提示时要使用有仓库读取权限的 personal access token，或使用已经配置好的 SSH/GitHub CLI 登录。Windows 上的登录不会自动出现在 AutoDL 上。不要把 token 写入命令、URL 或 Notebook。

你贴出的日志表明 clone 没成功；因此 `cd FactorBridge` 失败，pip 随后在 `/root` 中找不到项目。不要继续使用那次误建的 `/root/.venv`；下面显式调用仓库内的 Python。无需删除旧环境。

[GitHub 官方认证说明](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/about-authentication-to-github#https)

## 在 JupyterLab 的 Terminal 中执行

复制代码框内容，不要复制 Markdown 链接语法，不要在每行末尾加反斜杠。第一次下载用 `clone`，仓库已存在后才用 `pull`。

先完成第一步，确认退出码成功后执行安装：

```bash
mkdir -p /root/autodl-tmp
git clone https://github.com/wangherm/FactorBridge.git /root/autodl-tmp/FactorBridge
```

用户名填 `wangherm`；Password 提示里输入 GitHub token，输入过程不回显。也可先用已安装的 `gh auth login` 和 `gh auth setup-git` 登录，再 clone。

以下整段在子 shell 内运行，任何一步失败都会停止，不会接着运行下一步：

```bash
(
  set -euo pipefail
  cd /root/autodl-tmp/FactorBridge
  test -f pyproject.toml
  python -m venv .venv
  .venv/bin/python -m pip install -e .
  .venv/bin/python -m unittest discover -s tests -v
  export PATH="$PWD/.venv/bin:$PATH"
  bash scripts/run_public_pilot.sh
)
```

这一步只需要 NumPy，不会安装/启动 Qwen。此时 GPU 可以暂不开机，仅需 CPU 环境。数据下载失败、校验不符、已有实验结果都会明确停止。下载成功后可重复运行下载脚本核对缓存；数值实验不覆盖旧结果，重跑应指定新的 config/run_dir 并逐步执行命令。

已有仓库更新时：

```bash
cd /root/autodl-tmp/FactorBridge && git pull --ff-only
```

若使用 Notebook cell，`!cd` 和 `!source` 不会为后续 cell 保留状态。建议使用上面的 Terminal；已安装后的无交互运行也可在同一个 `%%bash` cell 中执行完整子 shell。不要在 Notebook 中保存 token。

## 当前实际选用数据

[NCBI GSE124109](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE124109)：大鼠 REF 成纤维细胞，血清饥饿 0、2、3、4、6、8、10、12、14、16 天，每点 3 个 biological replicates。下载两个官方文件：

- `GSE124109_family.soft.gz`：完整样本元数据。
- `GSE124109_genes.processed.fpkm_table.txt.gz`：17,353 genes × 30 samples，FPKM。

脚本按矩阵列名精确匹配 GEO 的 `Sample_title`，映射到 GSM 和 BioSample；不会按文件行顺序猜测对应关系。保留全部基因，使用自然对数 `log1p(FPKM)`，不冒充 raw counts，不执行计数 thinning，不重新做 CPM 归一化。官方文件 SHA256 固定在下载器中；来源发生变化时停止并要求重新审计，不自动接受新文件。

生物学单位依据是作者声明的逐时间点生物重复及 30 个独立 BioSample。未发布的 donor、passage、跨时间培养批次关联仍未知，已记录在 provenance。局部样本留出只能解释为 culture-sample projection diagnostic。

## 分组边界和输出

全部样本和噪声版本都属于 `public_train`。只有一个 study，**没有 public validation/test，没有独立泛化结论**。新增的显式 `pilot_only` 模式复用原有 candidate/card/refit/evaluate，禁止 Qwen 训练、validation/test 评估和冻结协议。正式模式仍要求独立 study 的 train/validation/test，不能把同一研究的时间点拆成不同 study。

`data/public/GSE124109/` 包含原始文件、`matrix.npz`、`metadata.csv`、`manifest.json`、`provenance.json` 和真实下载状态。

`runs/public_gse124109/` 包含 audit、split、noisy Candidate Cards、隔离的弱参考、数值基线、实际 W/Z、Factor Cards、noise curves 和命令日志。参考是稳定性弱参考，不是独立 biological gold。

2026-09-30 的实际首轮：16 张卡全部为 uncertain、已确认支持基因标签数为 0；非 LLM 训练报 `No supervised observations`。密集候选与仅保留 16 个基因的弱参考相似度不足，当前标签构造不适合直接进入训练。脚本显式输出监督审计，先运行 PCA、loading-refit、stability 三个无需拟合监督标签的方法；不生成非 LLM 或 Qwen 的假结果。后续应在独立公开开发研究上检验候选池和弱参考定义，再冻结规则；不要通过复制样本、强制赋标签或缩小评估范围来掩盖问题。

GSE117444 的公开设计仅有 6 个库，不能直接满足当前最少 8 个独立单位的约束。GSE3169 的 94 个阵列包含同一 worm pool 的多个时间点，且跨多个 array platform，不能把 94 个阵列当成 94 个独立单位加入当前 RNA-seq pilot。

下一步补充同物种/assay/resolution 的独立研究作为 validation/test，再按 [AutoDL 训练说明](AUTODL.md) 安装训练依赖、实测 GPU、pin 模型、prepare、smoke，最后正式训练。跨物种与多教师阶段在真实数据、教师表示和独立监督齐备后执行。
