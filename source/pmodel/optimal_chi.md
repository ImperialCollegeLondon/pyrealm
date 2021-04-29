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

# Calculation of optimal chi

The first step in the P-model is to use the provided values of temperature,
pressure, $\ce{CO2}$ concentration to calculate values for four key
environmentally determined photosynthetic variables.

## Photosynthetic parameters

The four key variables are the: 

1. photorespiratory compensation point ($\Gamma^*$),
2. Michaelis-Menten coefficient for photosynthesis ($K_{mm}$),
3. relative viscosity of water, given a standard at 25°C ($\eta^*$), and
4. partial pressure of $\ce{CO2}$ in ambient air ($c_a$).

The descriptions below show the typical ranges of these values under common environmental inputs along with links to the more detailed documentation of the key functions.

```{code-cell} python
:tags: [hide-input]
# This code loads required packages and then creates a representative range of
# values of the core variables to use in function plots.
#
# Note that the ranges are created (`_1d`) but are also cast to two dimensional
# arrays of repeating values (`_2d`) to generate response surfaces for functions
# with multuple inputs.

from matplotlib import pyplot
import numpy as np
from pyrealm import pmodel
%matplotlib inline

# get the default set of P Model parameters
pmodel_param = pmodel.PModelParams()

# Set the resolution of examples
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

### Photorespiratory compensation point ($\Gamma^*$)

Details: {func}`pyrealm.pmodel.calc_gammastar`

The photorespiratory compensation point ($\Gamma^*$) varies with as a function
of temperature and atmospheric pressure:

```{code-cell} python
:tags: [hide-input]
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

### Michaelis-Menten coefficient for photosynthesis ($K_{mm}$)

Details: {func}`pyrealm.pmodel.calc_kmm`

The Michaelis-Menten coefficient for photosynthesis ($K_{mm}$ )also varies with
temperature and atmospheric pressure:

```{code-cell} python
:tags: [hide-input]
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

### Relative viscosity of water ($\eta^*$)

Details: {func}`pyrealm.pmodel.calc_density_h2o`, {func}`pyrealm.pmodel.calc_viscosity_h2o`

The density ($\rho$) and viscosity ($\mu$) of water both vary with temperature
and atmospheric pressure. Together, these functions are used to calculate the
viscosity of water relative to its viscosity at standard temperature and
pressure ($\eta^*$).

The figure shows how $\eta^*$ varies with temperature and pressure.

```{code-cell} python
:tags: [hide-input]
# Calculate the viscosity under the range of values and the standard 
# temperature and pressure
viscosity = pmodel.calc_viscosity_h2o(tc_2d, patm_2d.transpose())
viscosity_std = pmodel.calc_viscosity_h2o(pmodel_param.k_To, pmodel_param.k_Po)

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

### Partial pressure of $\ce{CO2}$ ($c_a$)

Details: {func}`pyrealm.pmodel.calc_co2_to_ca`

The partial pressure of $\ce{CO2}$ is a function of the atmospheric concentration of $\ce{CO2}$ in parts per million and the atmospheric pressure: 

```{code-cell} python
:tags: [hide-input]
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

## Calculation of optimal chi

Details: {class}`pyrealm.pmodel.CalcOptimalChi`

The next step is to calculate the following factors:

- The optimal ratio of leaf internal to ambient $\ce{CO2}$ partial
  pressure ($\chi = c_i/c_a$).
- The $\ce{CO2}$ limitation term of light use efficiency ($m_j$).
- The limitation term for $V_{cmax}$ ($m_{joc}$). 

The class supports two methods: `prentice14` and `c4`. 

### The `c4` method

This method simply sets all three variables ($\chi$, $m_j$ and $m_{joc}$) to 1,
to reflect the lack of $\ce{CO2}$ limitation in C4 plants.

### The `prentice14` method

This method calculates values for $\chi$ ({cite}`Prentice:2014bc`),  $m_j$
({cite}`Wang:2017go`) and $m_{joc}$ (???). The plots below show how these
parameters change with different environmental inputs.


```{code-cell} python
:tags: [hide-input]
# Create inputs for a temperature curve at two atmospheric pressures
patm_1d = pmodel.calc_patm(np.array([0, 3000]))
tc_2d = np.broadcast_to(tc_1d, (2, n_pts))
patm_2d = np.broadcast_to(patm_1d, (n_pts, 2)).transpose()

# Pass those through the intermediate steps to get inputs for CalcOptimalChi
gammastar = pmodel.calc_gammastar(tc_2d, patm=patm_2d)
kmm = pmodel.calc_kmm(tc_2d, patm=patm_2d)
viscosity = pmodel.calc_viscosity_h2o(tc_2d, patm=patm_2d)
viscosity_std = pmodel.calc_viscosity_h2o(pmodel_param.k_To, pmodel_param.k_Po)
ns_star = viscosity / viscosity_std

# Compare four scenarios of differing CO2 and VPD
ch = pmodel.calc_co2_to_ca(co2=410, patm=patm_2d)
cl = pmodel.calc_co2_to_ca(co2=280, patm=patm_2d)
optchi_ch_vh = pmodel.CalcOptimalChi(kmm=kmm, gammastar=gammastar, 
                                     ns_star=ns_star, ca=ch, vpd = 1)
optchi_ch_vl = pmodel.CalcOptimalChi(kmm=kmm, gammastar=gammastar, 
                                     ns_star=ns_star, ca=ch, vpd=0.5)
optchi_cl_vh = pmodel.CalcOptimalChi(kmm=kmm, gammastar=gammastar, 
                                     ns_star=ns_star, ca=cl, vpd = 1)
optchi_cl_vl = pmodel.CalcOptimalChi(kmm=kmm, gammastar=gammastar, 
                                     ns_star=ns_star, ca=cl, vpd=0.5)

# Create line plots of optimal chi
pyplot.plot(tc_1d, optchi_ch_vh.chi[0, ], label='0m, 410 ppm, VPD 1')
pyplot.plot(tc_1d, optchi_ch_vh.chi[1, ], label='3000m, 410 ppm, VPD 1')
pyplot.plot(tc_1d, optchi_ch_vl.chi[0, ], label='0m, 410 ppm, VPD 0.5')
pyplot.plot(tc_1d, optchi_ch_vl.chi[1, ], label='3000m, 410 ppm, VPD 0.5')
pyplot.title('Variation in optimal chi')
pyplot.xlabel('Temperature °C')
pyplot.ylabel('Optimal chi')
pyplot.legend()
pyplot.show()

# Create line plots of mj
pyplot.plot(tc_1d, optchi_ch_vh.mj[0, ], label='0m, 410 ppm, VPD 1')
pyplot.plot(tc_1d, optchi_cl_vh.mj[0, ], label='0m, 280 ppm, VPD 1')
pyplot.title('Variation in m_j')
pyplot.xlabel('Temperature °C')
pyplot.ylabel('m_j')
pyplot.legend()
pyplot.show()

# Create line plots of mj
pyplot.plot(tc_1d, optchi_ch_vh.mjoc[0, ], label='0m, 410 ppm, VPD 1')
pyplot.plot(tc_1d, optchi_ch_vh.mjoc[1, ], label='3000m, 410 ppm, VPD 1')
pyplot.plot(tc_1d, optchi_cl_vh.mjoc[0, ], label='0m, 280 ppm, VPD 1')
pyplot.plot(tc_1d, optchi_cl_vh.mjoc[1, ], label='3000m, 280 ppm, VPD 1')
pyplot.title('Variation in m_joc')
pyplot.xlabel('Temperature °C')
pyplot.ylabel('m_joc')
pyplot.legend()
pyplot.show()

```






