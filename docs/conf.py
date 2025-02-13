from typing import Any, Dict

# -- General configuration ------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.coverage",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "numpydoc",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "myst_nb",
    "sphinx_design",
    "notfound.extension",
    "sphinx_copybutton",
    "sphinx_codeautolink",
    "jupyter_sphinx",
]

# Project information
project = "pyloo"
copyright = "2025, Jordan Deklerk"
author = "Jordan Deklerk"
version = "0.0.1"
release = version

# codeautolink configuration
codeautolink_autodoc_inject = False
codeautolink_search_css_classes = ["highlight-default"]
codeautolink_concat_default = True

# ipython directive configuration
ipython_warning_is_error = False

# Generate API documentation when building
autosummary_generate = True
numpydoc_show_class_members = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# MyST related params
nb_execution_mode = "auto"
nb_execution_excludepatterns = ["*.ipynb"]
nb_kernel_rgx_aliases = {".*": "python3"}
myst_heading_anchors = 0
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "amsmath"]

# copybutton config: strip console characters
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# numpydoc configuration
autodoc_typehints = "none"
numpydoc_xref_param_type = True
numpydoc_xref_ignore = {
    "optional",
    "default",
    "of",
    "or",
}

# The base toctree document
master_doc = "index"
default_role = "code"

# List of patterns to ignore when looking for source files
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# Configure notfound extension
notfound_urls_prefix = "/en/latest/"

# The name of the Pygments (syntax highlighting) style to use
pygments_style = "sphinx"

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages
html_theme = "pydata_sphinx_theme"

# Theme options
html_theme_options = {
    "logo": {
        "image_light": "pyloo_logo.png",
        "image_dark": "pyloo_logo.png",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/jordandeklerk/pyloo",
            "icon": "fa-brands fa-github",
        },
    ],
    "navbar_start": ["navbar-logo", "navbar-version"],
    "navbar_align": "content",
    "header_links_before_dropdown": 5,
    "secondary_sidebar_items": ["page-toc", "edit-this-page", "sourcelink"],
    "use_edit_page_button": True,
}

html_context = {
    "github_user": "jordandeklerk",
    "github_repo": "pyloo",
    "github_version": "main",
    "doc_path": "docs/",
    "default_mode": "light",
}

html_sidebars: Dict[str, Any] = {"index": []}

# Add any paths that contain custom static files
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Example configuration for intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "arviz": ("https://python.arviz.org/en/latest/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
}

# -- Options for LaTeX output ---------------------------------------------
latex_elements: Dict[str, str] = {}

# Grouping the document tree into LaTeX files
latex_documents = [(master_doc, "pyloo.tex", "pyloo Documentation", author, "manual")]

# -- Options for manual page output ---------------------------------------
man_pages = [(master_doc, "pyloo", "pyloo Documentation", [author], 1)]

# -- Options for Texinfo output -------------------------------------------
texinfo_documents = [
    (
        master_doc,
        "pyloo",
        "pyloo Documentation",
        author,
        "pyloo",
        "Python implementation of the R package loo for LOO-CV and PSIS.",
        "Miscellaneous",
    )
]
