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

# Environmental determination of key photosynthetic parameters

This document describes the calculation of four key photosynthetic variables
used to calculate optimal $\chi$ in the P-model. These calculations are the
first performed in the P-model implementation and all are simple direct
calculations from input variables.

1. The photorespiratory compensation point ($\Gamma^*$)
2. The Michaelis-Menten coefficient for photosynthesis ($K_{mm}$)
3. The relative viscosity of water, given a standard at 25°C ($\eta^*$)
4. The partial pressure of $\ce{CO2}$ in ambient air.

This documentation is a Jupyter notebook and uses code examples to demonstrate
the behaviour of these parameters with environmental inputs. The code below
loads required packages and then creates a representative range of values of the
core variables. Note that the ranges are created (`_1d`) but are also cast to
two dimensional arrays of repeating values (`_2d`) to make it easy to create
response surfaces of the functions.

```{code-cell} python
from matplotlib import pyplot
import numpy as np
from pyrealm import pmodel
from pyrealm.params import PARAM
%matplotlib inline
```

```{code-cell} python
n_pts = 101

# Create a range of representative values for key inputs.
tc_1d = np.linspace(0, 50, n_pts)
patm_1d = np.linspace(60000, 106000, n_pts) # ~ tropical tree line
co2_1d = np.linspace(200, 500, n_pts)

# Broadcast the range into arrays with repeated values.
tc_2d = np.broadcast_to(tc_1d, (n_pts, n_pts))
patm_2d = np.broadcast_to(patm_1d, (n_pts, n_pts))
co2_2d = np.broadcast_to(co2_1d, (n_pts, n_pts))
```


## Photorespiratory compensation point ($\Gamma^*$)

$\Gamma^*$ varies with temperature and atmospheric pressure and is calculated by
{func}`pyrealm.pmodel.calc_gammastar`. The figure shows how $\Gamma^*$ varies
with different inputs and function documentation is shown below the figure:

```{code-cell} python
# Calculate gammastar
gammastar = pmodel.calc_gammastar(tc_2d, patm_2d.transpose())

# Create a contour plot of gamma
fig, ax = pyplot.subplots()
CS = ax.contour(tc_1d, patm_1d, gammastar, levels=10, colors='black')
ax.clabel(CS, inline=1, fontsize=10)
ax.set_title('Gamma star')
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Atmospheric pressure (Pa)')
pyplot.show()
```

```{eval-rst}
.. autofunction:: pyrealm.pmodel.calc_gammastar
```


## Michaelis-Menten coefficient for photosynthesis ($K_{mm}$)

$K_{mm}$ also varies with temperature and atmospheric pressure and is calculated by
{func}`pyrealm.pmodel.calc_kmm`. The figure shows how $K_{mm}$ varies
with different inputs and the function documentation is shown below the figure:


```{code-cell} python
# Calculate K_mm
kmm = pmodel.calc_kmm(tc_2d, patm_2d.transpose())

# Contour plot of calculated values
fig, ax = pyplot.subplots()
CS = ax.contour(tc_1d, patm_1d, kmm, levels=[10,25,50,100,200,400], colors='black')
ax.clabel(CS, inline=1, fontsize=10)
ax.set_title('KMM')
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Atmospheric pressure (Pa)')
pyplot.show()
```

```{eval-rst}
.. autofunction:: pyrealm.pmodel.calc_kmm
```

## Relative viscosity of water ($\eta^*$)

The density ($\rho$) and viscosity ($\mu$) of water both vary with temperature
and atmospheric pressure and are calculated using
{func}`pyrealm.pmodel.calc_density_h2o` and
{func}`pyrealm.pmodel.calc_viscosity_h2o`. Together, these are used to
calculate the  viscosity of water relative to its viscosity at standard
temperature and pressure ($\eta^*$).

The figure shows how $\eta^*$ varies with temperature and pressure and the
function documentation is shown below the figure:

```{code-cell} python
# Calculate the viscosity under the range of values and the standard 
# temperature and pressure
viscosity = pmodel.calc_viscosity_h2o(tc_2d, patm_2d.transpose())
viscosity_std = pmodel.calc_viscosity_h2o(PARAM.k.To, PARAM.k.Po)

# Calculate the relative viscosity
ns_star = viscosity / viscosity_std

# Plot ns_star
fig, ax = pyplot.subplots()
CS = ax.contour(tc_1d, patm_1d, ns_star, colors='black')
ax.clabel(CS, inline=1, fontsize=10)
ax.set_title('NS star')
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Atmospheric pressure (Pa)')
pyplot.show()
```

```{eval-rst}
.. autofunction:: pyrealm.pmodel.calc_density_h2o
```
```{eval-rst}
.. autofunction:: pyrealm.pmodel.calc_viscosity_h2o
```


## Partial pressure of $\ce{CO2}$ ($c_a$)

The partial pressure of $\ce{CO2}$ is a function of the atmospheric concentration of $\ce{CO2}$ in parts per million and the atmospheric pressure, and is calculated by {func}`calc_co2_to_ca`. 

```{code-cell} python
# Variation in partial pressure
ca = pmodel.calc_co2_to_ca(co2_2d, patm_2d.transpose())
# Plot contour plot of values
fig, ax = pyplot.subplots()
CS = ax.contour(co2_1d, patm_1d, ca, colors='black')
ax.clabel(CS, inline=1, fontsize=10)
ax.set_title('CO2')
ax.set_xlabel('Atmospheric CO2 (ppm)')
ax.set_ylabel('Atmospheric pressure (Pa)')
pyplot.show()
```

```{eval-rst}
.. autofunction:: pyrealm.pmodel.calc_co2_to_ca
```
