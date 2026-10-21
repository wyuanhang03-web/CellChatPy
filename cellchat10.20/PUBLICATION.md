# GitHub 研究成果快照

本分支在原仓库根目录下新增 `cellchat10.20/`，不替换主分支安装包。
开发包位于 `cellchat10.20/development/`；算法、测试源码、固定上游基线、
实验报告、教程执行 notebook、图像、主要数值证据、安装 wheel 和汇报 PPT 均保留。
这次公开的是研究代码快照，不代表 PyPI 发布或 Read the Docs 部署。

## 上传范围与限制

为避免将数 GB 可再生数据写入 Git，以下文件留在原始本地工作区：

- `.venv/`、缓存、构建临时文件、PPT `.build/`、旧的失败/中间教程执行目录。
- 原始教程输入、PBMC 下载压缩包和预处理数据、教程 pickle/HDF5 对象。
- `saved_object_evidence/` 大量展开数组及巨大清单；对应比较报告保留于 `validation/`。
- 调研时下载的第三方仓库源码副本；调研索引和获取脚本保留。

`PUBLICATION_MANIFEST.json` 记录复制文件的 SHA-256、大小及筛选规则。
其中 README 的哈希对应原本地版本；本发布副本另加了上传范围说明。
原实验记录中保留的本机路径只是来源记录，并非新机器的安装路径。

`validation/delivery_audit.json` 是原完整本地交付的历史核验结果，
**不是删减后 GitHub 快照已经通过完整交付审计的声明**。
直接对本快照运行 `scripts/validate_delivery.py` 会因大型原始证据缺失而失败；
须恢复对应完整证据，或重新完成实验及证据生成后再审计。

## 新机器安装与检查（Windows / Python 3.11）

从仓库根目录运行：

```powershell
cd cellchat10.20
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r provenance/requirements-tested.txt
.\.venv\Scripts\python.exe -m pip install "./development[accelerate]"
.\run_checks.ps1
```

教程输入的 8 个文件名见 `development/data/README.md`。需要自行从原数据
提供方或项目维护者取得，并放到**仓库根目录的 `data/`**（即
`cellchat10.20` 的同级目录），因为教程 runner 使用该位置。
如需全量重跑，在运行 `run_checks.ps1 -Tutorials` 前，将此快照已有的
`results/tutorials/reference` 与 `results/tutorials/development` 移到备份位置，
避免把公开的部分输出当作完整的可恢复执行目录。

真实 PBMC 数据由脚本下载和预处理：

```powershell
.\run_checks.ps1 -LargeBenchmarks
```

原报告的速度比来自已记录的特定硬件和参数；本次上传未重新执行大型实验。
详细条件和科学结论边界见 `RESULTS.md` 与 `RESEARCH_POSITIONING.md`。

## 发布副本检查

本次上传前在独立发布副本中重新核对了固定基线的全部 393 个文件，
并运行 `development/tests`、`validation/test_streaming_engine.py` 和
`validation/test_sparse_quantiles.py`：303 项通过，无失败。
检查未发现超过 GitHub 100 MiB 单文件限制的待上传文件，亦未发现所扫描的
GitHub token、AWS access-key ID 或私钥头部模式。这不是全面安全审计。
