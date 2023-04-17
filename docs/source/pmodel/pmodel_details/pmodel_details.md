---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3
  language: python
  name: pyrealm_python3
---

# The P Model: details

This section provides details of the calculations used in the P Model and provides
examples of the outputs of those calculations for a range of environmental conditions.

The main steps are:

* Calculation of the [photosynthetic environment](photosynthetic_environment).
* Calculation of [leaf $\ce{CO2}$ variables](optimal_chi).
* Constraints on [light use efficiency (LUE)](lue_limitation).
* Estimation of [gross primary productivity](estimating-productivity).

A [worked example](global_example.md) shows the use of the P Model on
global gridded data. Additional pages provide further information on:

* Approaches to incorporate [soil moisture effects](soil_moisture.md)
* Model behaviour in [extreme environments](extreme_values.md)
* Comparison to the [`rpmodel` implementation](rpmodel.md)

```{eval-rst}
.. toctree::
  :maxdepth: 4
  :hidden:

  photosynthetic_environment.md
  optimal_chi.md
  lue_limitation.md
  envt_variation_outputs.md
  soil_moisture.md
  extreme_values.md
  global_example.md
  rpmodel.md
```
