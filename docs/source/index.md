---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
language_info:
  codemirror_mode:
    name: ipython
    version: 3
  file_extension: .py
  mimetype: text/x-python
  name: python
  nbconvert_exporter: python
  pygments_lexer: ipython3
  version: 3.11.9
---

<!-- markdownlint-disable-next-line MD041-->
```{image} /_static/images/pyrealm_logo.png
:width: 50%
:align: center
:alt: The pyrealm logo: a green leaf over the shining sun.
```

The `pyrealm` package provides Python implementations of models of plant productivity
and demography.

# The `pyrealm` package

The package is in active development and currently provides the following modules:

The `pmodel` module
: Fitting the [P Model](users/pmodel/module_overview), which is an ecophysiological
  model of optimal carbon dioxide uptake by plants {cite:p}`Prentice:2014bc,
  Wang:2017go,Stocker:2020dh`, along with various extensions.

The `tmodel` module
: Estimating plant allocation of gross primary productivity to growth and respiration,
  using the [T Model](users/tmodel/tmodel) {cite:p}`Li:2014bc`.

The `core` module
: Contains fundamental utilities and physics functionality shared across the
  package, including the [hygro](users/hygro) and the utilities submodules.

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
