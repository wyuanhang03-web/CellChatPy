from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "CellChatPy"
copyright = "2026, CellChatPy contributors"
author = "CellChatPy contributors"
release = "1.1.0"

extensions = [
    "sphinx_copybutton",
    "myst_parser",
    "sphinx_design",
    "nbsphinx",
]

myst_enable_extensions = ["colon_fence"]
nbsphinx_execute = "never"
nbsphinx_allow_errors = True

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "titles_only": True,
    "includehidden": False,
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "CellChatPy documentation"

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

exclude_patterns = ["_build", "**/.ipynb_checkpoints"]
