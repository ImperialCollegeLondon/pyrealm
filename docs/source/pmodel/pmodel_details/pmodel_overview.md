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

<!-- markdownlint-disable-next-line MD041 -->
(pmodel_overview)=

# The P Model

This page provides an overview of the theory of the P Model
{cite:p}`Prentice:2014bc,Wang:2017go`, and how to use the implementation in the
`pyrealm` package. The details of calculations and the API for the package code are
shown in the [module reference documentation](../../api/pmodel_api).

* Approaches to incorporate [soil moisture effects](soil_moisture.md)
* Model behaviour in [extreme environments](extreme_values.md)
* Comparison to the [`rpmodel` implementation](rpmodel.md)

## Overview

The P-model is a model of carbon capture and water use by plants. Four
variables are used to define the environment that the plant experiences:

* temperature (`tc`, °C),
* vapor pressure deficit (`vpd`, Pa),
* atmospheric $\ce{CO2}$ concentration (`co2`, ppm), and
* atmospheric pressure (`patm`, Pa).

From these inputs, the model breaks down into four broad stages, each of which
is described in more detail in the link for each stage

The main steps are:

* Calculation of the [photosynthetic environment](photosynthetic_environment).
* Calculation of [leaf $\ce{CO2}$ variables](optimal_chi).
* Constraints on [light use efficiency (LUE)](lue_limitation).
* Estimation of [gross primary productivity](estimating-productivity).

### Photosynthetic environment

The environmental variables are used to calculate variables describing the
photosynthetic environment of a plant (see
[details](photosynthetic_environment)).

### Calculation of leaf $\ce{CO2}$ variables

The photosynthetic environment is then used to calculate the optimal ratio of
internal to external CO2 concentration ($chi$), along with $\ce{CO2}$ partial
pressures and limitation factors (see [details](optimal_chi)).

This step also governs the main differences between C3 and C4 photosynthesis.

### Limitation of light use efficiency (LUE)

The calculation of light use efficiency can be subjected to a number of
constraints. (see [details](lue_limitation)).

* Theoretical limitations to the maximum rates of Rubsico regeneration
   ($J_{max}$) and maximum carboxylation capacity ($V_{cmax}$)

* Temperature sensitivity of the quantum yield efficiency of photosynthesis
(`kphio`, $\phi_0$).

* Soil moisture stress.

### Estimation of GPP

Once LUE has been calculated, estimates of absorbed photosynthetically active
radiation, can be used to predict gross primary productivity (GPP) and other key
rates within the leaf (see [details](estimating-productivity)).

### Worked code examples

Two examples of how to use the {mod}`~pyrealm` package to fit the P Model can be seen in
the [worked examples](worked_examples) page.

### Variable graph

The graph below shows these broad model areas in terms of model inputs (blue)
and modelled outputs (red) used in the P-model. Optional inputs and internal
variables are shown with a dashed edge.

![pmodel.svg](pmodel.svg)

## Array inputs

The `pyrealm` package uses the `numpy` package and expects arrays of data to be be
passed to all inputs. Input arrays can be a single scalar value, but all non-scalar
inputs must be **arrays with the same shape**: the `pyrealm` packages does not attempt
to resolve the broadcasting of array dimensions.

## Extreme values

The four photosynthetic environment variables and the effect of temperature
on the temperature dependence of quantum yield efficiency are all calculated
directly from the input forcing variables. While the majority of those calculations
behave smoothly with extreme values of temperature and atmospheric pressure,
the calculation of the relative viscosity of water ($\eta^*$) does not handle
low temperatures well. The behaviour of these functions with extreme values
is shown [here](extreme_values).

```{eval-rst}
.. toctree::
  :maxdepth: 4
  :hidden:

  worked_examples.md
  photosynthetic_environment.md
  optimal_chi.md
  lue_limitation.md
  envt_variation_outputs.md
  soil_moisture.md
  extreme_values.md
  rpmodel.md
```
