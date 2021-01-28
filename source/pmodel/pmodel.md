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

# The P-Model

This module provides a Python implementation of the P-model
(:{cite}`Prentice:2014bc`, :{cite}`Wang:2017go`). These summary documents give
an overview of using the model and show typical code and outputs. Details of
calculations are typically included in the function and class descriptions in
the API documentation ([here](pmodel_reference)) and links to these are included
in the summary descriptions.

The implementation in this module is based very heavily on the `rpmodel`
implementation of the model ({cite}`Stocker:2020dh`) and gives identical outputs
but the Python implementation has some differences in the code structure (see
[here](rpmodel) for discussion).

## Overview

The P-model uses four core environmental variables to build a model of carbon
capture and water use:

- temperature (`tc`),
- vapor pressure deficit (`vpd`),
- atmospheric $\ce{CO2}$ concentration (`co2`), and
- atmospheric pressure (`patm`)

The model breaks down into four broad stages, each of which is described in more
detail in the link for each stage

1. Conversion of environmental variables to photosynthetic parameters
   (see [details](optimal_chi#photosynthetic-parameters)).
2. Calculation $\ce{CO2}$ pressures and limitation factors
   (see [details](optimal_chi#calculation-of-optimal-chi)) 
3. Calculation of light use efficiency and maximum carboxylation capacity
   (see [details](lue_vcmax)). 
4. Optionally, scaling of key outputs to the total absorbed irradiance (see [below](#scaling-to-absorbed-irradiance)).


### Variable graph

The graph below shows these broad model areas in terms of model inputs (blue)
and modelled outputs (red) used in the P-model. Optional inputs and internal
variables are shown with a dashed edge.

![pmodel.svg](pmodel.svg)

## Example use

Running the P Model will typically just use the {class}`~pyrealm.pmodel.PModel`,
possibly with some preprocessing of optional inputs. The code below illustrates
different use cases.


### Simple use

Fitting the model for:

   * a temperature of 20°C,
   * standard atmospheric at sea level (101325 Pa),
   * a vapour pressure deficit of 0.82 kPa (~ 65% relative humidity), and
   * an atmospheric $\ce{CO2}$ concentration of 400 ppm.

```{code-cell} ipython3
from pyrealm import pmodel
model = pmodel.PModel(tc=20.0, patm=101325.0, vpd=0.82, co2=400)
```

The returned model object holds a lot of information. The model object itself contains calculated values of photosynthetic parameters, model options and intrinsic water use efficiency:

```{code-cell} ipython3
model
```

There is also a {class}`CalcOptimalChi` object, containing details of the calculation of optimal ratio of internal to ambient $\ce{CO2}$ pressure ($\chi$) calculations and $\ce{CO2}$ limitation factors to both light assimilation ($m_j$) and carboxylation ($m_c$).

```{code-cell} ipython3
model.optchi
```

Last, the object contains `unit_iabs`,  an instance of class
{class}`~pyrealm.pmodel.IabsScaled` that contains six variables that scale with
absorbed irradiance, including the light use efficiency (LUE) and maximum
carboxylation rate ($V_{cmax}$). The values in `unit_iabs` are reported as
values **per unit absorbed irradiance**:

```{code-cell} ipython3
model.unit_iabs
```

### Scaling to absorbed irradiance

Since the values stored in class {class}`~pyrealm.pmodel.IabsScaled` scale
linearly with absorbed irradiance ($I_{abs}$), it is  straightforward to convert
them to absolute values. The {class}`~pyrealm.pmodel.IabsScaled` class contains
a convenience method {meth}`~pyrealm.pmodel.IabsScaled.scale_iabs` that will
automatically scale all six variables, given the fraction of absorbed
photosynthetically active radiation (`fapar`) and the photosynthetic photon flux
density (`ppfd`).

Note that the units of PPFD determine the units of outputs. The example below
uses representative values for tropical rainforest, with PPFD expressed as
$\text{mol}\,m^{-2}\,\text{month}^{-1}$: GPP is therefore $g\,C\,m^{-2}
\text{month}^{-1}$:

```{code-cell} ipython3
model.unit_iabs.scale_iabs(fapar=0.91, ppfd=834)
```

### Array inputs

The `pyrealm` package uses the `numpy` package for most calculation, and arrays
of data can be passed to all inputs. If arrays are being used, then all inputs
must either be scalars or **arrays with the same shape**: the PModel does not
attempt to apply calculations across combinations of different dimensions.

The example below repeats the model above for a range of temperature values and
plots the resulting light use efficiency curve per unit absorbed irradiance.

```{code-cell} ipython3
from matplotlib import pyplot
import numpy as np
%matplotlib inline

# Create a sequence of temperatures and fit the model
tc = np.linspace(20, 30, 101)
model_array = pmodel.PModel(tc=tc, patm=101325.0, vpd=0.82, co2=400)

# Plot TC against LUE
pyplot.plot(tc, model_array.unit_iabs.lue)
pyplot.xlabel('Temperature °C')
pyplot.ylabel('Light use efficiency')
pyplot.show()
```

### Elevation data

The {func}`~pyrealm.pmodel.calc_patm` function can be used to convert elevation
data to atmospheric pressure, for use in the {class}`~pyrealm.pmodel.PModel`
class. The example below repeats the model at an elevation of 3000 metres and
compares the resulting light use efficiencies.

```{code-cell} ipython3
patm = pmodel.calc_patm(3000)
model_3000 = pmodel.PModel(tc=20, patm=patm, vpd=0.82, co2=400)

# Tiny change in LUE
np.array([model.unit_iabs.lue, model_3000.unit_iabs.lue])
```

### Apparent quantum yield efficiency

A default value for apparent quantum yield efficiency (`kphio`, $\phi_0$) is set
automatically. Following {cite}`Stocker:2020dh`, the default value depends on
(see also {class}`~pyrealm.pmodel.PModel`):

1. is a soil moisture stress factor being applied, and
1. is the temperature dependency of $\phi_0$ being applied.

The user can however specify their own value. This is used here and then below
to faciliate comparisons across other optional settings.

```{code-cell} ipython3
model_fixkphio = pmodel.PModel(tc=20, patm=patm, vpd=0.82, 
                               co2=400, kphio=0.08)
```

### Soil moisture stress

{cite}`Stocker:2020dh` includes an optional soil moisture stress factor, used to
modify the light use efficency and calculated from the relative soil moisture
and an aridity index. In the {mod}`~pyrealm.pmodel` module, this factor is
calculated by the use in advance with the function
{func}`~pyrealm.pmodel.calc_soilmstress` and passed to
{class}`~pyrealm.pmodel.PModel`. 

See the documentation {func}`~pyrealm.pmodel.calc_soilmstress` and class
{class}`~pyrealm.pmodel.CalcLUEVcmax` for the details.

```{code-cell} ipython3
# soil moisture stress factor
soilmstress = pmodel.calc_soilmstress(soilm=0.4, meanalpha=0.9)
soilmstress
```

```{code-cell} ipython3
model_soil = pmodel.PModel(tc=20, patm=101325.0, vpd=0.82, 
                           co2=400, kphio=0.08, soilmstress=soilmstress)

# Compare LUE
np.array([model_fixkphio.unit_iabs.lue, model_soil.unit_iabs.lue])
```

### Temperature dependence of $\phi_0$

By default, {class}`~pyrealm.pmodel.PModel` automatically calculates a
correction factor for the temperature dependence of $\phi_0$ (see
{func}`calc_ftemp_kphio`) and passes this factor to
{class}`~pyrealm.pmodel.CalcLUEVcmax`. The factor is stored in the model object:

```{code-cell} ipython3
model_fixkphio.ftemp_kphio
```

This correction can be removed using the argument `do_ftemp_kphio=False`:

```{code-cell} ipython3
model_no_td_kphio = pmodel.PModel(tc=20, patm=patm, vpd=0.82, co2=400,
                                  kphio=0.08, do_ftemp_kphio=False)

# Compare LUE
np.array([model_fixkphio.unit_iabs.lue, model_no_td_kphio.unit_iabs.lue])
```

### Limitation of electron transfer rate ($J_{max}$)

The {class}`~pyrealm.pmodel.PModel` automatically uses the method described by
{cite}`Wang:2017go` to account for limitation of the maximum rate of electron
transfer ($J_{max}$ limitation). Using the argument `method_jmaxlim`, this
correction can either be omitted (`method_jmaxlim='none'`) or the alternative
formulation of {cite}`Smith:2019dh` can be used (`method_jmaxlim='smith19'`).

```{code-cell} ipython3
model_jmax_none = pmodel.PModel(tc=20, patm=patm, vpd=0.82, co2=400,
                                kphio=0.08, method_jmaxlim='none')
model_jmax_smith19 = pmodel.PModel(tc=20, patm=patm, vpd=0.82, co2=400,
                                   kphio=0.08, method_jmaxlim='smith19')

# Compare LUE from the three methods
np.array([model_fixkphio.unit_iabs.lue, 
          model_jmax_none.unit_iabs.lue,
          model_jmax_smith19.unit_iabs.lue])
```

