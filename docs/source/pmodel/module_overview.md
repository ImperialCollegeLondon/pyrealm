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

# The `pmodel` module

The module implements the P Model, which is a ecophysiological model of optimal carbon
dioxide uptake by plants, along with a number of extensions to the model.

The main components are:

* The standard implementation of the [P Model](pmodel_details/pmodel_overview.md)
  {cite:p}`Prentice:2014bc,Wang:2017go,Stocker:2020dh`, along with more recent updates
  to estimate soil moisture effects.

* An extension to the standard model that incorporates fast and slow responses within
  the photosynthetic pathway to changing environmental conditions. These different
  responses need to be included to give realistic predictions from the P Model at fine
  temporal resolutions, particularly when the P Model is applied to datasets with
  [subdaily observations](subdaily_details/subdaily_overview.md).

* The estimation of [isotopic discrimination of
  carbon](isotopic_discrimination.md) resulting from photosynthesis.

* A model of [C3 / C4 plant competition](c3c4model), giving an estimate of
  the expected fraction of C4 plants in a community.

```{eval-rst}
.. toctree::
  :maxdepth: 4
  :hidden:

  pmodel_details/pmodel_overview.md
  subdaily_details/subdaily_overview.md
  isotopic_discrimination.md
  c3c4model.md
```
