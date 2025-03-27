---
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
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

<!-- markdownlint-disable-next-line MD041-->
```{image} /_static/images/pyrealm_logo.png
:width: 50%
:align: center
:alt: The pyrealm logo: a green leaf over the shining sun.
```

The `pyrealm` package provides Python implementations of models of plant productivity
and demography. All of the functionality within the package is built to accept arrays of
data and uses the {mod}`numpy` package to efficiently calculate values across datasets
with multiple dimensions, up to analyses of global spatial datasets of long-running time
series.

# The `pyrealm` package

:::{admonition} Version 2.0.0
The `pyrealm` package has just been updated to version 2.0.0. There are a quite a few
breaking changes to the previous version, documented in the [migration
guide](users/versions.md) to help update existing code. We strongly recommend upgrading
to the new version.
:::

The package currently provides the following modules:

The `core` module
: Contains fundamental utilities and physics functionality shared across the
  package, including the [hygro](users/hygro) and the utilities submodules.

The `pmodel` module
: Fitting the [P Model](users/pmodel/module_overview), which is an ecophysiological
  model of optimal carbon dioxide uptake by plants {cite:p}`Prentice:2014bc,
  Wang:2017go,Stocker:2020dh`, along with various extensions.

The `splash` module
: Fits the [SPLASH v1 model](users/splash.md), which can be used to
  estimate soil moisture, actual evapotranspiration and soil runoff from daily
  temperature, precipitation and sunshine data {cite:p}`davis:2017a`.

The `demography` module
: Provides functionality for [modelling plant allocation and growth and
  demography](users/demography/module_overview.md), including classes to represent plant
  functional types, cohorts and communities. This module includes an implementation of
  the T Model for estimating plant allocation of gross primary productivity to growth
  and respiration {cite:p}`Li:2014bc`. This module is still in active development but a
  lot of initial functionality is present.

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
