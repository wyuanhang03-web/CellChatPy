# CellChatPy 2026-10-20 研究开发工作区

选题：**保持 CellChat 计算定义的大规模、低内存通讯推断**。

此目录为本次研究开发工作区的 GitHub 发布快照。源码、报告、主要验证结果
与 PPT 保留；原始数据和大型中间对象留在本地。安装、复现前请先阅读
[发布范围与复现说明](PUBLICATION.md)。本次未部署 Read the Docs。

## 源码来源

已通过 GitHub API 核对线上 main 为
`986bf7ef909f1e15059aad8a9a9ee31ac675ffc3`。
本地 `CellChatPy_GitHub` 的 HEAD 是同一提交，但其 working tree 中的
`CellChatPy/modeling.py` 已有未提交改动。根目录 `CellChatPy` 有 4 个模块
不同，`work2/CellChatPy` 有 9 个模块不同。

本项目使用 `git archive` 从该提交提取准确基线，核对 commit/tree 哈希；
没有把改动过的本地工作树当作 GitHub 发布版。详细记录见
`provenance/source_comparison.json`、`provenance/github_commit.json`。

## 内容

| 路径 | 用途 |
|---|---|
| `upstream/` | 固定 GitHub 源码，保留不改 |
| `development/` | 新引擎、原功能、测试和文档的完整开发包 |
| `METHOD.md` | 计算契约、算法、复杂度与研究边界 |
| `RESEARCH_POSITIONING.md` | 论文贡献、等价性论证与投稿前待补实验 |
| `RESULTS.md` | 最终实验汇总，由原始证据生成 |
| `validation/` | 独立回归测试、JUnit、教程源核验和数值比较 |
| `scripts/` | 教程重跑、大数据基准和汇总脚本 |
| `results/tutorials/` | 每本完整 executed.ipynb、图像和数值快照 |
| `results/benchmarks/` | 每个独立进程的时间、内存、结果和对照 |
| `data/` | 新下载真实 PBMC 数据及固定预处理输入 |
| `provenance/` | 来源、原始 archive、环境、版本记录 |
| `dist/` | 已构建的本地 wheel；源码/数据库完整性核验见 `validation/wheel_integrity.json` |

最终交付核验记录为 `validation/delivery_audit.json`：重新检查保存的数值
数组，并核对源码、回归记录、教程、性能输出和安装包之间的对应关系。

## 使用开发版

```powershell
python -m pip install ".\development[accelerate]"
```

新引擎是 `compute_communication_probability` 的默认执行方式，也可显式
传入 `engine="streaming"`。`engine="legacy"` 保留原计算实现用于诊断。
`gene_batch_size` 和 `lr_batch_size` 控制回退密集计算与 LR 临时张量大小。

算法和限制见 `development/docs/streaming_engine.md`。本地版本为
`1.1.0+streaming.20261020`，本分支提供研究源码快照，尚未发布至 PyPI。

## 复现

测试环境记录在 `provenance/environment.json` 和
`provenance/requirements-tested.txt`。本机 `.venv` 只读复用了现有 pytorch
环境的依赖，没有修复或替换系统默认 Anaconda 环境。
迁移至其他机器时应重新创建 Python 3.11 环境并安装所记录依赖。

```powershell
# 基线完整性 + 原发行测试 + 新算法回归
.\run_checks.ps1

# 再执行七个教程的基线/开发版对照（完整执行可能较久）
.\run_checks.ps1 -Tutorials

# 真实大数据推断核基准，默认单线程、三个重复
.\run_checks.ps1 -LargeBenchmarks

# 检查当前已保存的完整交付证据（不重新运行推断）
.\.venv\Scripts\python.exe -B scripts/validate_delivery.py
```

教程 runner 的 `--resume` 仅复用**本次工作区中已经完整重跑并通过**的
执行记录；不会读取原项目的旧推断 pickle 作为新引擎结果。若希望重新
生成所有教程证据，请使用新的 label 分别调用 `scripts/run_tutorials.py`。
成对 runner 会在全部 notebook 结束后自动对四本教程新保存的对象作补充
比较，涵盖可用的通路与 spot 网络。全部测试的源码指纹和 JUnit 校验值
记录在 `validation/regression_provenance.json`。

大型实验与教程按资源窗口顺序执行；若 `results/tutorials/PAUSE` 存在，
教程 runner 会在下一本开始前等待。这个文件只用于本次资源协调。

## 结果解释

真实数据与合成规模测试分开报告；推断核速度与整个 Python 进程/完整教程
速度分开报告。概率及 P 值的一致性不等价于真实生物学通讯验证。
本次没有把加一校正、自适应停止、细胞下采样或改变 triMean 定义作为加速。
保存/加载顺序与教程手算断言的修复单独记录，不冒充算法贡献。
