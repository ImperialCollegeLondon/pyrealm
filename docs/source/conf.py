"""Configure the Sphinx documentation builder.

-- Path setup --------------------------------------------------------------

If extensions (or modules to document with autodoc) are in another directory,
add these directories to sys.path here. If the directory is relative to the
documentation root, use os.path.abspath to make it absolute, like shown here.
"""

import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Import Matplotlib to avoid this message in notebooks:
# "Matplotlib is building the font cache; this may take a moment."
import matplotlib.pyplot  # noqa: F401
import sphinxcontrib.bibtex.plugin
from sphinxcontrib.bibtex.style.referencing import BracketStyle
from sphinxcontrib.bibtex.style.referencing.author_year import AuthorYearReferenceStyle

from pyrealm import __version__ as pyrealm_version
from pyrealm.core.experimental import ExperimentalFeatureWarning

sys.path.insert(0, str(Path("../").resolve()))
sys.path.append(str(Path("./sphinx_extensions").resolve()))

# Suppress the 2.0.0 user warnings in the documentation and also suppress the
# experimental feature warnings.
warnings.filterwarnings(
    "ignore", category=UserWarning, message="Pyrealm 2.0.0 uses a new default value"
)
warnings.filterwarnings("ignore", category=ExperimentalFeatureWarning)


# -- Project information -----------------------------------------------------

project = "pyrealm: Ecosystem Models in Python"
html_logo = "_static/images/pyrealm_logo_white_background.png"
html_favicon = "_static/images/pyrealm_favicon.png"

current_year = datetime.today().strftime("%Y")
copyright = "2020-" + current_year + ", Pyrealm Developers"
author = "Pyrealm Developers"

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
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinxcontrib.bibtex",
    "myst_nb",
    "sphinx_rtd_theme",
    "sphinx_design",
    "sphinx_external_toc",
    "experimental",
]

# External table of contents
external_toc_path = "_toc.yml"  # optional, default: _toc.yml
external_toc_exclude_missing = False  # optional, default: False

# Include TODOs
todo_include_todos = True


# Citation styling
def bracket_style() -> BracketStyle:
    """Custom citation parenthesis style."""
    return BracketStyle(
        left="(",
        right=")",
    )


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

bibtex_reference_style = "author_year_round"
bibtex_default_style = "plain"

# Cross-reference checking
# TODO - find some better solution than this to all of these bizarre cross reference
#        problems.

nitpicky = True
nitpick_ignore = [
    ("py:class", "numpy._typing._array_like._ScalarType_co"),
    ("py:class", "numpy._typing._generic_alias.ScalarType"),
    ("py:class", "numpy.float32"),
    ("py:class", "numpy.float64"),
    ("py:class", "numpy.int64"),
    ("py:class", "numpy.timedelta64"),
    ("py:class", "numpy.bool_"),
    ("py:class", "numpy.ndarray"),
    ("py:class", "numpy.dtype"),
    ("py:class", "numpy.dtype[+ScalarType]"),
    ("py:class", "numpy.typing.NDArray"),
    ("py:class", "numpy.NDArray"),
    ("py:class", "NDArray"),
    ("py:class", "dataclasses.InitVar"),
    (
        "py:class",
        "dataclasses.InitVar[numpy.ndarray[typing.Any, numpy.dtype[+ScalarType]]]",
    ),
    (
        "py:class",
        "dataclasses.InitVar[numpy.ndarray[typing.Any, numpy.dtype[+_ScalarType_co]]]",
    ),
    (
        "py:class",
        (
            "tuple[numpy.ndarray[typing.Any, numpy.dtype[+ScalarType]], "
            "numpy.ndarray[typing.Any, numpy.dtype[+ScalarType]]]"
        ),
    ),
    (
        "py:obj",
        (
            "typing.Union[~numpy.ndarray[~typing.Any, "
            "~numpy.dtype[~numpy._typing._generic_alias.ScalarType]], "
            "tuple[numpy.ndarray[typing.Any, numpy.dtype[+ScalarType]], "
            "numpy.ndarray[typing.Any, numpy.dtype[+ScalarType]], "
            "numpy.ndarray[typing.Any, numpy.dtype[+ScalarType]]]]"
        ),
    ),
    (
        "py:class",
        (
            "tuple[numpy.ndarray[typing.Any, numpy.dtype[+ScalarType]], "
            "numpy.ndarray[typing.Any, numpy.dtype[+ScalarType]], "
            "numpy.ndarray[typing.Any, numpy.dtype[+ScalarType]]]"
        ),
    ),
    (
        "py:class",
        (
            "tuple[numpy.ndarray[typing.Any, numpy.dtype[+ScalarType]], "
            "numpy.ndarray[typing.Any, numpy.dtype[+ScalarType]], "
            "numpy.ndarray[typing.Any, numpy.dtype[+ScalarType]]]"
        ),
    ),
    ("py:class", "pandas.core.frame.DataFrame"),
]

intersphinx_mapping = {
    "pytest": ("https://docs.pytest.org/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "python": ("https://docs.python.org/3/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "pandas": ("http://pandas.pydata.org/pandas-docs/dev/", None),
    "shapely": ("https://shapely.readthedocs.io/en/stable/", None),
    "marshmallow": ("https://marshmallow.readthedocs.io/en/stable/", None),
}


autodoc_default_flags = ["members"]
autosummary_generate = True

myst_enable_extensions = ["dollarmath", "deflist", "colon_fence"]
myst_heading_anchors = 4

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

# html_theme = "sphinx_material"
html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "logo_only": True,
    "version_selector": True,
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
