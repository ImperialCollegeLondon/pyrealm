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

The package contains a number of different modules implementing different models:

1. The [P Model](pmodel/pmodel): a ecophysiological model of carbon dioxide
   uptake by plants (:{cite}`Prentice:2014bc`, :{cite}`Wang:2017go`, :{cite}`Stocker:2020dh`)
2. The [T Model](tmodel/tmodel)

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
  pmodel/pmodel_reference.md
  pmodel/rpmodel.md
```

```{eval-rst}
.. toctree::
  :maxdepth: 4
  :caption: The T Model
  :hidden:
  
  tmodel/tmodel.md
  tmodel/tmodel_reference.md
```

```{eval-rst}
.. toctree::
  :maxdepth: 4
  :caption: Additional detail
  :hidden:
  
  hygro.md
  utilities.md
  params.md
  z_bibliography.rst
```

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
