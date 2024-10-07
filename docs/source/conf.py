"""Configure the Sphinx documentation builder.

-- Path setup --------------------------------------------------------------

If extensions (or modules to document with autodoc) are in another directory,
add these directories to sys.path here. If the directory is relative to the
documentation root, use os.path.abspath to make it absolute, like shown here.
"""

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime

import sphinxcontrib.bibtex.plugin
from sphinx.application import Sphinx
from sphinxcontrib.bibtex.style.referencing import BracketStyle
from sphinxcontrib.bibtex.style.referencing.author_year import AuthorYearReferenceStyle

from pyrealm import __version__ as pyrealm_version

# Add the project's root directory to the sys.path
sys.path.insert(0, os.path.abspath("../"))

# -- Project information -----------------------------------------------------
project = "pyrealm: Ecosystem Models in Python"
current_year = datetime.today().strftime("%Y")
copyright = "2020-" + current_year + ", Pyrealm Developers"
author = "Pyrealm Developers"
version = pyrealm_version
release = pyrealm_version

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "autodocsumm",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinxcontrib.bibtex",
    "myst_nb",
    "sphinx_rtd_theme",
    "sphinx_external_toc",
    # "sphinx_autodoc_typehints",  # For type hints formatting
]

# External table of contents
external_toc_path = "_toc.yml"
external_toc_exclude_missing = False


# Citation styling
def bracket_style() -> BracketStyle:
    """Custom citation parenthesis style."""
    return BracketStyle(left="(", right=")")


@dataclass
class MyReferenceStyle(AuthorYearReferenceStyle):
    """Custom referencing style."""

    bracket_parenthetical: BracketStyle = field(default_factory=bracket_style)
    bracket_textual: BracketStyle = field(default_factory=bracket_style)
    bracket_author: BracketStyle = field(default_factory=bracket_style)
    bracket_label: BracketStyle = field(default_factory=bracket_style)
    bracket_year: BracketStyle = field(default_factory=bracket_style)


sphinxcontrib.bibtex.plugin.register_plugin(
    "sphinxcontrib.bibtex.style.referencing", "author_year_round", MyReferenceStyle
)

# Cross-reference checking
nitpicky = True
nitpick_ignore = [
    ("py:class", "numpy._typing._array_like._ScalarType_co"),
    ("py:class", "numpy._typing._generic_alias.ScalarType"),
    ("py:class", "numpy.float32"),
    ("py:class", "numpy.int64"),
    ("py:class", "numpy.timedelta64"),
    ("py:class", "numpy.bool_"),
    ("py:class", "numpy.ndarray"),
    ("py:class", "numpy.dtype"),
    ("py:class", "numpy.dtype[+ScalarType]"),
    ("py:class", "numpy.typing.NDArray"),
    ("py:class", "dataclasses.InitVar"),
]

intersphinx_mapping = {
    "pytest": ("https://docs.pytest.org/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "pandas": ("http://pandas.pydata.org/pandas-docs/dev/", None),
    "shapely": ("https://shapely.readthedocs.io/en/stable/", None),
}

autodoc_default_flags = ["members"]
autosummary_generate = True

myst_enable_extensions = ["dollarmath", "deflist", "colon_fence"]
myst_heading_anchors = 4

# Enable mhchem for chemical formulae
mathjax3_config = {
    "tex": {
        "extensions": ["mhchem.js"],
    }
}

# Napoleon settings
napoleon_use_ivar = True
napoleon_custom_sections = [("PModel Parameters", "params_style")]

# Autodoc configuration:
autodoc_preserve_defaults = True
add_module_names = False
# napoleon_use_ivar = True
napoleon_preprocess_types = True
napoleon_use_param = True
napoleon_use_rtype = False

autodoc_member_order = "groupwise"
autodoc_typehints_format = "short"
autodoc_typehints = "description"

bibtex_bibfiles = ["refs.bib"]

templates_path = ["_templates"]
exclude_patterns = ["maxime*", "**.ipynb_checkpoints"]
html_static_path = ["_static"]

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "top",
    "style_external_links": False,
    "style_nav_header_background": "grey",
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}


def setup(app: Sphinx):  # type: ignore
    """Setup function to connect the event handler to autodoc-process-docstring."""

    # Ignore .ipynb files
    app.registry.source_suffix.pop(".ipynb", None)
