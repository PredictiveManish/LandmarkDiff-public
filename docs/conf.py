# Sphinx configuration for LandmarkDiff documentation

project = "LandmarkDiff"
author = "dreamlessx"
copyright = "2024, dreamlessx"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# MyST settings for Markdown support
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# HTML theme
html_theme = "furo"
html_title = "LandmarkDiff"

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# Exclude patterns
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
