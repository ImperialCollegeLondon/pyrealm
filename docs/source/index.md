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

## Module overview

The package implements several modelling approaches to estimating the optimal
photosynthetic behaviour of plants and then using the resulting estimates of
productivity to model plant growth and the demography of plant communities.

The package is in active development and currently provides the following functionality:

* Fitting the [P Model](pmodel/pmodel), which is a ecophysiological model of optimal
  carbon dioxide uptake by plants {cite:p}`Prentice:2014bc, Wang:2017go,Stocker:2020dh`.

  Extensions to this model include:

  * Estimation of [isotopic discrimination of
    carbon](pmodel/isotopic_discrimination.md) resulting from photosynthesis.
  * A model of [C3 / C4 plant competition](pmodel/c3c4model), giving an estimate of
    the expected fraction of C4 plants in a community.
  * Applying the P Model to datasets with [subdaily observations](pmodel/subdaily), when
    slow responses of photosynthetic mechanisms to changing environmental conditions
    need to be included to give realistic predictions.

* Estimating plant allocation of gross primary productivity to growth and respiration,
  using [T Model](tmodel/tmodel) {cite:p}`Li:2014bc`.
* Functions for [converting common hygrometric variables](./hygro) to vapour pressure
  deficit for use in the P Model.

```{eval-rst}
.. toctree::
  :maxdepth: 4
  :caption: For Users
  :hidden:

  pmodel/pmodel.md
  pmodel/pmodel_details/pmodel_details.md
  pmodel/subdaily.md
  pmodel/subdaily_grid.md
  pmodel/memory_effect.md
  pmodel/isotopic_discrimination.md
  pmodel/c3c4model.md
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
