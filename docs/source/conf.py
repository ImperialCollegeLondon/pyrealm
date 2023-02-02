"""Configure the Sphinx documentation builder.

-- Path setup --------------------------------------------------------------

If extensions (or modules to document with autodoc) are in another directory,
add these directories to sys.path here. If the directory is relative to the
documentation root, use os.path.abspath to make it absolute, like shown here.
"""

import os
import sys

from pyrealm import __version__ as pyrealm_version

sys.path.insert(0, os.path.abspath("../"))


# -- Project information -----------------------------------------------------

project = "pyrealm: Ecosystem Models in Python"
copyright = "2020, David Orme"
author = "David Orme"

# The full version, including alpha/beta/rc tags
version = pyrealm_version
release = pyrealm_version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "autodocsumm",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinxcontrib.bibtex",
    "myst_nb",
    # "sphinx_astrorefs",  # Gives author year references
    "sphinx_rtd_theme",
]

autodoc_default_flags = ["members"]
autosummary_generate = True

myst_enable_extensions = ["dollarmath", "deflist"]

# Enable mhchem for chemical formulae
mathjax3_config = {
    "tex": {
        "extensions": ["mhchem.js"],
        # 'inlineMath': [['$', '$']]
    }
}

# Turn off ugly rendering of class attributes
napoleon_use_ivar = True
napoleon_custom_sections = [("PModel Parameters", "params_style")]

# Autodoc configuration:
# - Suppress signature expansion of arguments
autodoc_preserve_defaults = True
# - Have funcname not pyrealm.pmodel.funcname in autodoc
add_module_names = False
# - Group members by type not alphabetically
autodoc_member_order = "groupwise"

bibtex_bibfiles = ["refs.bib"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["maxime*", "**.ipynb_checkpoints"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

# html_theme = 'sphinx_material'
html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "top",
    "style_external_links": False,
    "style_nav_header_background": "grey",
    # Toc options
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


def setup(app):  # type: ignore
    """Use setup to remove .ipynb from sources.

    Note that this assumes that all ipynb files are paired with Myst Markdown
    files via Jupytext
    """
    # Ignore .ipynb files
    app.registry.source_suffix.pop(".ipynb", None)
