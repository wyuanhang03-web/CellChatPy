# CellChatPy 上传 GitHub 与生成教程网页操作指南

本目录 `E:\Jin\cellchat.python_new\CellChatPy_GitHub` 已整理为可发布仓库。
它参考 Spatial-Query 的组织方式，包含 Python 源码、七个教程 Notebook、
数据说明、教程输出、Sphinx 文档和 Read the Docs 配置；原始大数据不放入
GitHub 仓库。

> **教程数量说明：** 源目录 `E:\Jin\cellchat.python_new\work2\tutorial`
> 实际包含 7 个 Notebook，而非 6 个。第 7 个
> `SpatialCellChat_analysis_of_spatial_transcriptomics_data.ipynb` 包含热点、
> 空间共现和 communication motif 分析；本项目的
> `Spatial Visualization of Motif` 教程网页依赖它，因此已一并保留。

## 1. 已整理的内容

```text
CellChatPy_GitHub/
|-- CellChatPy/             Python 包和 CellChatDB
|-- data/                   仅保留数据说明，原始大数据本地保存
|-- tutorial/               七个可运行 Notebook、图片和 CSV 输出
|-- docs/                   Sphinx/nbsphinx 教程网页源文件
|   |-- tutorials/          七个 Notebook 的发布副本、motif 页面和比较图集
|   `-- _static/            网页样式与图集图片
|-- .readthedocs.yaml       Read the Docs 构建配置
|-- pyproject.toml          安装与依赖配置
|-- README.md
|-- LICENSE
`-- GITHUB_UPLOAD_GUIDE_CN.md
```

七个教程输入数据约 1.04 GiB，已从发布目录移除，不会上传到 GitHub。
运行 Notebook 前，请按照 `data/README.md` 的说明把数据放回本地
`data/`；`.gitignore` 会阻止 `.h5` 和 `.h5ad` 被误提交。各套
`figures_*` 图片目录和交互结果 CSV 会正常上传。

## 2. 关于 `.pkl` 输出

教程网页不依赖 `.pkl`，当前没有把它们复制进发布目录（高级空间教程的
结果表格 CSV 已保留在 `tutorial/spatial_tutorial_output/`）。原工作区中有多个
60 MB 至 5.2 GB 的序列化对象，其中最大的文件通常超过 GitHub Free/Pro
可接受的单文件大小，也不适合作为普通 GitHub 文件长期保存。

推荐把 `.pkl` 作为可选复现产物上传到 Zenodo、Figshare、OSF 或 GitHub
Release，并在 Release 说明中给出文件名、版本、SHA256 和数据许可证。
图片、表格和 Notebook 已足够生成完整教程网页，不会因缺少 `.pkl` 受影响。

## 3. 准备教程数据（不上传 GitHub）

进入已整理目录：

```powershell
cd "E:\Jin\cellchat.python_new\CellChatPy_GitHub"
```

如果需要运行 Notebook，把原工作区中的教程数据复制到发布目录：

```powershell
Copy-Item "E:\Jin\cellchat.python_new\data\*.h5" -Destination data -Force
Copy-Item "E:\Jin\cellchat.python_new\data\*.h5ad" -Destination data -Force
Get-ChildItem data
git status --ignored data
```

这些数据仅用于本地运行，不要使用 `git add -f` 强制添加。

## 4. 创建 GitHub 仓库

如果继续使用现有仓库（文件中的示例用户名是 `wyuanhang03-web`）：

```text
https://github.com/wyuanhang03-web/CellChatPy
```

可以直接进入下一节。若新建仓库，在 GitHub 点击 `New repository`，仓库名
填写 `CellChatPy`，不要勾选自动创建 README、`.gitignore` 或 License，
以免首次推送产生无关冲突。

如果你的 GitHub 用户名不是 `wyuanhang03-web`，把下面命令中的用户名和仓库
地址换成你自己的。例如用户名是 `xiaoming` 时，地址就是
`https://github.com/xiaoming/CellChatPy.git`。只改地址，不要改本地文件夹名。

## 5. 建立干净的 main 分支并推送

在 `CellChatPy_GitHub` 中执行：

```powershell
git init
git branch -M main
git add .
git status
git commit -m "Initial public release of CellChatPy"
git remote add origin https://github.com/wyuanhang03-web/CellChatPy.git
git push -u origin main
```

`git status` 应包含：

- `CellChatPy/` 源码和数据库；
- `tutorial/` 下七个 Notebook、空间 motif 输出、图片目录和 CSV；
- `data/README.md`；
- `docs/` 和 `.readthedocs.yaml`；
- 根目录的 `r_permutations.csv`。

不应包含 `data/*.h5`、`data/*.h5ad`、虚拟环境、`__pycache__`、
`docs/_build` 或 `.pkl`。

如果提示 `origin already exists`：

```powershell
git remote set-url origin https://github.com/wyuanhang03-web/CellChatPy.git
```

现有远程如果仍以 `master` 为默认分支，上述操作会新增 `main`，不会覆盖
`master`。推送成功后在 `Settings > Branches` 中把默认分支切换为 `main`；
确认内容完整前先保留旧分支。

GitHub HTTPS 不接受账户密码。首次推送时使用 Git Credential Manager 的
浏览器登录、`gh auth login`，或 Personal Access Token。

## 6. 发布到 PyPI

发布前先确认 GitHub 中的源码是准备公开的最终版本，并登录
[PyPI](https://pypi.org/)。进入 `Account settings > API tokens`，创建一个
API Token。首次发布时可创建账户级 Token；项目建立后，后续可以改用只允许
访问 `CellChatPy` 项目的 Token。Token 只会完整显示一次，不要提交到 GitHub，
也不要写入 README、脚本或本指南。

进入发布目录，安装官方打包工具并构建：

```powershell
cd "E:\Jin\cellchat.python_new\CellChatPy_GitHub"
python -m pip install --upgrade build twine
python -m build
python -m twine check dist\*
```

检查通过后，上传到正式 PyPI：

```powershell
python -m twine upload dist\*
```

Twine 提示输入凭据时：

- `username` 填写固定值 `__token__`；
- `password` 粘贴以 `pypi-` 开头的完整 API Token；
- PowerShell 中粘贴密码时屏幕不显示字符是正常现象。

上传成功后打开 `https://pypi.org/project/CellChatPy/`，再在一个新的虚拟环境
中从 PyPI 验证安装：

```powershell
python -m venv "$env:TEMP\cellchatpy-install-test"
& "$env:TEMP\cellchatpy-install-test\Scripts\python.exe" -m pip install --upgrade pip
& "$env:TEMP\cellchatpy-install-test\Scripts\python.exe" -m pip install CellChatPy
& "$env:TEMP\cellchatpy-install-test\Scripts\python.exe" -c "import CellChatPy as cc; print(cc.__version__); print(cc.__file__)"
```

当前发行版本为 `1.1.0`。PyPI 不允许删除后重新上传相同版本的发行文件；以后
每次发布都要同时修改 `pyproject.toml` 和 `CellChatPy/__init__.py` 中的版本号，
删除旧的本地 `dist/` 后重新构建。例如修复版本可从 `1.1.0` 升为 `1.1.1`。
包名大小写不影响安装，`pip install CellChatPy` 和 `pip install cellchatpy`
会定位到同一个 PyPI 项目。

PyPI 发行包只包含 Python 源码和运行所需的 CellChatDB，不包含约 1 GiB 教程
数据、Notebook 或输出图片。Notebook、图片和表格由 GitHub 仓库提供；原始
教程数据需要单独保存和分发。

## 7. 本地预览教程网页

网站机制与 Spatial-Query 相同：Sphinx 负责站点，MyST 负责 Markdown，
`nbsphinx` 把 Notebook 的标题、Markdown、代码、表格和已保存图片输出转换
为 HTML。配置中的 `nbsphinx_execute = "never"` 表示构建时不重新执行计算，
因此 Read the Docs 不需要读取约 1 GiB 教程数据。

`nbsphinx` 还需要系统中存在 Pandoc。先检查：

```powershell
pandoc --version
```

如果命令不存在，可以在 Windows PowerShell 中执行
`winget install --id JohnMacFarlane.Pandoc -e`，安装后重新打开终端。
Read the Docs 会按照 `.readthedocs.yaml` 自动安装 Pandoc，无需另行配置。

然后在虚拟环境中执行：

```powershell
cd "E:\Jin\cellchat.python_new\CellChatPy_GitHub"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r docs/requirements.txt
sphinx-build -b html docs docs\_build\html
```

然后打开：

```text
docs\_build\html\index.html
```

每个 Notebook 中的 `#`、`##`、`###` 标题会生成章节和锚点。为确保得到
与 Spatial-Query 完全同型的定位链接，发布包还提供了独立页面
`docs/tutorials/spatial_motif_enrichment.md`，其标题是
`# Spatial Visualization of Motif`。网站地址为：

```text
https://你的项目名.readthedocs.io/en/latest/tutorials/spatial_motif_enrichment.html#spatial-visualization-of-motif
```

页面中给出了从 `identify_cell_topics` 到 `plot_spatial_topics` 的完整代码，
并展示已保存的 `15_incoming_topics_spatial.png`。这就是本项目生成
“Spatial Visualization of Motif”的固定入口。

NL/LS 比较 Notebook 当前没有内嵌输出，所以额外提供了
`docs/tutorials/comparison_gallery.md`，展示已保存的代表性结果；完整 115 张
图片仍在 `tutorial/figures_comparison/` 中并随 GitHub 仓库公开。

## 8. 连接 Read the Docs

1. 登录 [Read the Docs](https://readthedocs.org/) 并授权 GitHub。
2. 点击 `Add project` 或 `Import a Project`。
3. 选择 `wyuanhang03-web/CellChatPy`。
4. 项目名称建议填写 `cellchatpy`，默认分支选择 `main`。
5. Advanced Settings 中确认配置文件为 `.readthedocs.yaml`。
6. 点击 `Build version`，等待 `latest` 构建完成。
7. 打开生成的网址，通常为 `https://cellchatpy.readthedocs.io/en/latest/`。

首次构建重点查看 `tutorials` 导航、七个 Notebook 页面、motif 页面、图片输出、代码块、
比较图集和标题锚点。当前配置只安装 `docs/requirements.txt` 中的文档依赖
以及系统 Pandoc，不导入完整科学计算运行时，也不会执行 Notebook。

文档构建只使用 Notebook 内嵌输出，不需要真实 H5/H5AD 内容，因此 Read the
Docs 构建不会因为仓库没有教程原始数据而失败。

## 9. 修改 Notebook 后同步文档副本

日常在根目录的 `tutorial/` 中运行和保存 Notebook。发布前执行（七个 notebook
都要同步）：

```powershell
Copy-Item tutorial\*.ipynb docs\tutorials\ -Force
sphinx-build -b html docs docs\_build\html
git add tutorial docs
git commit -m "Update tutorials and documentation"
git push
```

Read the Docs 会在 GitHub 更新后自动重建。不要只更新 `tutorial/` 而忘记
`docs/tutorials/`，否则 GitHub 上的 Notebook 与网站内容会不一致。

## 10. 验证 GitHub 安装和教程数据准备

推送后在一个新目录测试：

```powershell
git clone https://github.com/wyuanhang03-web/CellChatPy.git CellChatPy-test
cd CellChatPy-test
python -m pip install -e ".[tutorial]"
python -c "import CellChatPy as cc; print(cc.__version__); print(cc.__file__)"
python -c "import CellChatPy as cc; db=cc.load_database('human'); print(db['interaction'].shape)"
```

正常情况下版本为 `1.1.0`，human interaction 数据表形状为 `(3233, 28)`。
如果要运行 Notebook，还需要按照 `data/README.md` 将教程输入数据单独放入
测试目录的 `data/`；这些文件不会从 GitHub clone 下来。

## 11. 建议完善的发布信息

- GitHub About：`Python toolkit for cell-cell communication analysis`；
- Topics：`single-cell`、`spatial-transcriptomics`、
  `cell-cell-communication`、`bioinformatics`、`python`；
- 在 README 中补充正式作者、单位、联系方式、论文和数据 DOI；
- 为七个教程数据补充来源、许可证和 SHA256，并发布到 Zenodo、Figshare 或 OSF；
- 创建 `v1.1.0` Release，并在 Release 说明中附上教程数据的 DOI；
- 将不适合放入 GitHub 的可选 `.pkl` 放入专业数据仓库或合适的 Release 资产中。
