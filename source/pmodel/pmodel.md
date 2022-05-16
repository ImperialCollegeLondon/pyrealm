---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.6.0
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(pmodel_overview)=
# PModel overview and example use

This module provides a Python implementation of the P-model
(:{cite}`Prentice:2014bc`, :{cite}`Wang:2017go`). This provides an overview of
using the model, showing typical code and outputs, and the details of
calculations are included in the [module reference documentation](pmodel_reference).

The implementation in this module draws from the `rpmodel` implementation of the
model ({cite}`Stocker:2020dh`) and development matches the predictions of the
two implementation for most - but not all use cases (see [here](rpmodel)
for discussion).

## Overview

The P-model is a model of carbon capture and water use by plants. Four 
variables are used to define the environment that the plant experiences:

- temperature (`tc`, °C),
- vapor pressure deficit (`vpd`, Pa),
- atmospheric $\ce{CO2}$ concentration (`co2`, ppm), and
- atmospheric pressure (`patm`, Pa).

From thes inputs, the model breaks down into four broad stages, each of which 
is described in more detail in the link for each stage


### Step 1. Photosynthetic environment

The environmental variables are used to calculate variables describing the
photosynthetic environment of a plant (see
[details](photosynthetic_environment)).

### Step 2. Calculation of leaf $\ce{CO2}$ variables

The photosynthetic environment is then used to calculate the optimal ratio of
internal to external CO2 concentration ($chi$), along with $\ce{CO2}$ partial
pressures and limitation factors (see [details](optimal_chi)). 
   
This step also governs the main differences between C3 and C4 photosynthesis.

### Step 3. Limitation of light use efficiency (LUE)

The calculation of light use efficiency can be subjected to a number of
constraints. (see [details](lue_limitation)). 

* Theoretical limitations to the maximum rates of Rubsico regeneration
   ($J_{max}$) and maximum carboxylation capacity ($V_{cmax}$)

* Temperature sensitivity of the quantum yield efficiency of photosynthesis
(`kphio`, $\phi_0$). 

* Soil moisture stress. 

### Step 4. Estimation of GPP

Once LUE has been calculated, estimates of absorbed photosynthetically active
radiation, can be used to predict gross primary productivity (GPP) and other key
rates within the leaf (see [details](estimating-productivity)).

### Variable graph

The graph below shows these broad model areas in terms of model inputs (blue)
and modelled outputs (red) used in the P-model. Optional inputs and internal
variables are shown with a dashed edge.

![pmodel.svg](pmodel.svg)

## Example use

The first step is to use estimates of environmental variables to calculate the
photosynthetic environment for the model ({class}`~pyrealm.pmodel.PModelEnvironment`). 

The code below shows the steps required using a single site with:

   * a temperature of 20°C,
   * standard atmospheric at sea level (101325 Pa),
   * a vapour pressure deficit of 0.82 kPa (~ 65% relative humidity), and
   * an atmospheric $\ce{CO2}$ concentration of 400 ppm.

### Estimate photosynthetic environment

```{code-cell} ipython3
from pyrealm import pmodel
env  = pmodel.PModelEnvironment(tc=20.0, patm=101325.0, vpd=820, co2=400)
```

The `env` object now holds the photosynthetic environment, which can be re-used 
with different P Model settings. The representation of `env` is deliberately 
terse - just the shape of the data - but the 
{meth}`~pyrealm.pmodel.PModelEnvironment.summarize` method provides a 
more detailed summary of the attributes.  

```{code-cell} ipython3
env
```

```{code-cell} ipython3
env.summarize()
```

### Fitting the P Model

Next, the P Model can be fitted to the photosynthetic environment using the
({class}`~pyrealm.pmodel.PModel`) class:

```{code-cell} ipython3
model = pmodel.PModel(env)
```

The returned model object holds a lot of information. The representation of the
model object shows a terse display of the settings used to run the model:

```{code-cell} ipython3
model
```

A P model also has a {meth}`~pyrealm.pmodel.PModel.summarize` method 
that summarizes settings and displays a summary of calculated predictions.
Initially, this shows two measures of photosynthetic efficiency: the intrinsic 
water use efficiency (``iwue``) and the light use efficiency (``lue``).

```{code-cell} ipython3
model.summarize()
```

### $\chi$ estimates and $\ce{CO2}$ limitation

The P Model also contains a {class}`~pyrealm.pmodel.CalcOptimalChi` object,
recording key parameters from the [calculation of $\chi$](optimal_chi). This
object also has a {meth}`~pyrealm.pmodel.CalcOptimalChi.summarize` method:

```{code-cell} ipython3
model.optchi.summarize()
```

### Estimating productivity outputs

The productivity of the model can be calculated using estimates of the fraction
of absorbed photosynthetically active radiation ($f_{APAR}$, `fapar`, unitless)
and the photosynthetic photon flux density (PPFD,`ppfd`, µmol m-2 s-1), using the
{meth}`~pyrealm.pmodel.PModel.estimate_productivity` method. 

Here we are using:

* An absorption fraction of 0.91 (-), and
* a PPFD of 834 µmol m-2 s-1.

```{code-cell} ipython3
model.estimate_productivity(fapar=0.91, ppfd=834)
model.summarize()
```

```{warning}

To use {meth}`~pyrealm.pmodel.PModel.estimate_productivity`, the estimated PPFD
must be expressed as **µmol m-2 s-1**.

Estimates of PPFD sometimes use different temporal or spatial scales - for
example daily moles of photons per hectare. Although GPP can also be expressed
with different units, many other predictions of the P Model ($J_{max}$,
$V_{cmax}$, $g_s$ and $r_d$) _must_ be expressed as µmol m-2 s-1 and so this
standard unit must also be used for PPFD.
```

## Array inputs

The `pyrealm` package uses the `numpy` package and arrays 
of data can be passed to all inputs. If arrays are being used, then all inputs
must either be scalars or **arrays with the same shape**: the PModel does not
attempt to apply calculations across combinations of different dimensions.

The example below repeats the model above for a range of temperature values and
plots the resulting light use efficiency curve.

```{code-cell} ipython3
from matplotlib import pyplot
import numpy as np

# Create a sequence of temperatures and fit the model
tc = np.linspace(20, 30, 101)
env = pmodel.PModelEnvironment(tc=tc, patm=101325.0, vpd=820, co2=400)
model_array = pmodel.PModel(env)

# Plot TC against LUE
pyplot.plot(tc, model_array.lue)
pyplot.xlabel('Temperature °C')
pyplot.ylabel('Light use efficiency')
pyplot.show()
```

## Elevation data

The {func}`~pyrealm.pmodel.calc_patm` function can be used to convert elevation
data to atmospheric pressure, for use in the {class}`~pyrealm.pmodel.PModel`
class. The example below repeats the model at an elevation of 3000 metres and
compares the resulting light use efficiencies.



```{code-cell} ipython3
patm = pmodel.calc_patm(3000)
env = pmodel.PModelEnvironment(tc=20, patm=patm, vpd=820, co2=400)
model_3000 = pmodel.PModel(env)

# Tiny change in LUE
np.array([model.lue, model_3000.lue])
```


## Extreme values

The four photosynthetic environment variables and the effect of temperature
on the temperature dependence of quantum yield efficiency are all calculated
directly from the input forcing variables. While the majority of those calculations
behave smoothly with extreme values of temperature and atmospheric pressure, 
the calculation of the relative viscosity of water ($\eta^*$) does not handle
low temperatures well. The behaviour of these functions with extreme values
is shown [here](extreme_values).
