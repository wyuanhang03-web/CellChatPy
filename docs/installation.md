# Installation

CellChatPy requires Python 3.10 or newer.

## Install from PyPI

```bash
pip install CellChatPy
```

## Install the development version from GitHub

```bash
pip install "git+https://github.com/wyuanhang03-web/CellChatPy.git"
```

## Development and tutorials

Clone the repository. Large tutorial input datasets are intentionally kept
outside GitHub, so download or copy them separately before running notebooks:

```bash
git clone https://github.com/wyuanhang03-web/CellChatPy.git
cd CellChatPy
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[tutorial]"
# Put the required .h5/.h5ad files in data/; see data/README.md.
jupyter lab
```

On macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[tutorial]"
jupyter lab
```

Run Jupyter from the repository root. The notebooks discover `CellChatPy/`,
`data/`, and their output directories relative to that root.

## Build these docs locally

```bash
python -m pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` after the build completes.
