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

The `pyrealm` package provides Python implementations of models of plant productivity
and demography. The package is in active development and currently provides the
following modules:

## Module overview

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
