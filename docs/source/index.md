---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# The `pyrealm` package

## Module overview

The package implements several modelling approaches to estimating the optimal
photosynthetic behaviour of plants and then using the resulting estimates of
productivity to model plant growth and the demography of plant communities.

The package is in active development and currently provides the following functionality:

* Fitting [P Model](pmodel/pmodel), which is a ecophysiological model of optimal carbon
  dioxide uptake by plants (:{cite}`Prentice:2014bc`, :{cite}`Wang:2017go`,
  :{cite}`Stocker:2020dh`). Extensions to this model include:

  * Estimation of [isotopic discrimination of
    carbon](pmodel/isotopic_discrimination.md) resulting from photosynthesis.
  * A model of [C3 / C4 plant competition](pmodel/c3c4model), giving an estimate of
    the expected fraction of C4 plants in a community.

* Estimating plant allocation of gross primary productivity to growth and respiration,
  using [T Model](tmodel/tmodel) (:{cite}`Li:2014bc`).
* Functions for [converting common hygrometric variables](./hygro) to vapour pressure
  deficit for use in the P Model.

## Package documentation

The documentation for `pyrealm` is maintained using `sphinx`. The module code is
documented using Google style docstrings in RST format. Much of the rest of the
documentation uses Jupyter notebooks written using `myst-nb` format in order to
dynamically include Python code examples showing the behaviour of functions. In
general, the code will be concealed but the underlying code can be seen by
clicking on the buttons like the one below.

```{code-cell} python
:tags: [hide-input]
# This is just an example code cell to demonstrate how code is included in 
# the pyrealm documentation.
```

## Development notes

```{eval-rst}
.. toctree::
  :maxdepth: 4
  :caption: The P Model
  :hidden:

  pmodel/pmodel.md
  pmodel/photosynthetic_environment.md
  pmodel/optimal_chi.md
  pmodel/lue_limitation.md
  pmodel/envt_variation_outputs.md
  pmodel/soil_moisture.md
  pmodel/extreme_values.md
  pmodel/global_example.md
  pmodel/isotopic_discrimination.md
  pmodel/c3c4model.md
  pmodel/rpmodel.md
```

```{eval-rst}
.. toctree::
  :maxdepth: 4
  :caption: The T Model
  :hidden:
  
  tmodel/tmodel.md
```

```{eval-rst}
.. toctree::
  :maxdepth: 4
  :caption: API
  :hidden:

  ``tmodel`` module <api/tmodel_api.md>
  ``pmodel`` module <api/pmodel_api.md>
  ``hygro`` module <api/hygro_api.md>
  ``utilities`` module <api/utilities_api.md>
  ``constants`` module <api/constants_api.md>
```

```{eval-rst}
.. toctree::
  :maxdepth: 4
  :caption: Additional detail
  :hidden:
  
  hygro.md
  constants.md
  z_bibliography.rst
```

```{eval-rst}
.. toctree::
  :maxdepth: 4
  :caption: Developers
  :hidden:


```

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
