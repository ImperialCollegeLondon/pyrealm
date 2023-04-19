---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: pyrealm_python3
---

# The `pyrealm` package

The `pyrealm` package provides Python implementations of models of plant productivity
and demography. The package is in active development and currently provides the
following modules:

## Module overview

The `pmodel` module
: Fitting the [P Model](pmodel/module_overview), which is an ecophysiological model of
  optimal carbon dioxide uptake by plants {cite:p}`Prentice:2014bc,
  Wang:2017go,Stocker:2020dh`, along with various extensions.

The `tmodel` module
: Estimating plant allocation of gross primary productivity to growth and respiration,
  using the [T Model](tmodel/tmodel) {cite:p}`Li:2014bc`.

The `hygro` module
: Provides functions for [converting common hygrometric variables](./hygro) to vapour
  pressure deficit for use in the P Model.

```{eval-rst}
.. toctree::
  :maxdepth: 4
  :caption: For Users
  :hidden:

  module_overview.md
  pmodel/pmodel.md
  tmodel/tmodel.md
  hygro.md
  constants.md
  z_bibliography.rst
```

```{eval-rst}
.. toctree::
  :maxdepth: 4
  :caption: API
  :hidden:

  api/tmodel_api.md
  api/pmodel_api.md
  api/hygro_api.md
  api/utilities_api.md
  api/constants_api.md
```

```{eval-rst}
.. toctree::
  :maxdepth: 4
  :caption: Developers
  :hidden:

  development/developers.md

```

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
