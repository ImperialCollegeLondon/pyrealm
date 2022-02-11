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

# Calculation of Light Use Efficiency

Once key photosynthetic parameters have been calculated
([Details](optimal_chi)), the P model then calculates the light use efficiency
(LUE). In its simplest form, this is:

$$
  \text{LUE} = \phi_0 m_j M_C
$$

where $\phi_0$ is the quantum yield efficiency, $M_C$ is the molar mass of
carbon and $m_j$ is the $\ce{CO2}$ limitation term of light use efficiency from
the calculation of optimal $\chi$.

However, the implementation in {mod}`pyrealm.pmodel` also incorporates three
optional limiting factors: 

- a factor ($\phi_0(T)$) to capture the temperature dependence of $\phi_0$, 
- soil moisture stress ($\beta$) and
- a term ($m_{jlim}$) to capture $J_{max}$ limitation of $m_j$.

$$
  \text{LUE} = \phi_0 \phi_0(T) m_j m_{jlim} M_C \beta
$$

The Rubisco carboxylation capacity ($V_{cmax}$) of the system can be back
calculated from LUE as:

$$
  V_{cmax} = \frac{\text{LUE}}{m_c M_C},
$$

where $m_c$ is the  $\ce{CO2}$ limitation term for Rubisco assimilation.

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
n_pts = 201

# Create a range of representative values for key inputs.
tc_1d = np.linspace(-25, 50, n_pts)
soilm_1d = np.linspace(0, 1, n_pts)
meanalpha_1d = np.linspace(0, 1, n_pts) 
co2_1d = np.linspace(200, 500, n_pts)

# Broadcast the range into arrays with repeated values.
tc_2d = np.broadcast_to(tc_1d, (n_pts, n_pts))
soilm_2d = np.broadcast_to(soilm_1d, (n_pts, n_pts))
meanalpha_2d = np.broadcast_to(meanalpha_1d, (n_pts, n_pts))
co2_2d = np.broadcast_to(co2_1d, (n_pts, n_pts))
```

## Temperature dependence of quantum yield efficiency

The P-model uses a single variable to capture apparent quantum yield efficiency
(`kphio`). The default values of `kphio` vary with the model options, corresponding
to the empirically fitted values presented for three setups in {cite}`Stocker:2020dh`.

- If `do_ftemp_kphio = False`, then $\phi_0 = 0.049977$.
- If `do_ftemp_kphio = True` and
    - soil moisture stress is being used `kphio = 0.087182` (FULL),
    - soil moisture stress is not being used `kphio = 0.081785` (BRC).

The P-model also incorporates temperature dependence of `kphio` ($\phi_0(T)$),
determined by {cite}`Bernacchi:2003dc` and implemented in the function
{func}`pyrealm.pmodel.calc_ftemp_kphio`. This can be disabled by setting
`do_ftemp_kphio=False`. The scaling of this temperature dependence uses
different scalings for C3 and C4 plants.


```{code-cell} python
:tags: [hide-input]
# Calculate temperature dependence of quantum yield efficiency
fkphio_c3 = pmodel.calc_ftemp_kphio(tc_1d, c4=False)
fkphio_c4 = pmodel.calc_ftemp_kphio(tc_1d, c4=True)

# Create a line plot of ftemp kphio
pyplot.plot(tc_1d, fkphio_c3, label='C3')
pyplot.plot(tc_1d, fkphio_c4, label='C4')

pyplot.title('Temperature dependence of quantum yield efficiency')
pyplot.xlabel('Temperature °C')
pyplot.ylabel('Limitation factor')
pyplot.legend()
pyplot.show()
```

## Soil moisture stress factor

The P model implements an empirically derived factor ($\beta \in [0,1]$,
:{cite}`Stocker:2018be`, :{cite}`Stocker:2020dh`) that describes the response of
LUE to soil moisture stress. To implement this, you can provide the
{func}`~pyrealm.pmodel.PModel` class with values of $\beta$ estimated using
the function {func}`pyrealm.pmodel.calc_soilmstress`. This requires estimates
of:

* relative soil moisture (`soilm`), as the fraction of field capacity, and
* aridity (`meanalpha`), as the average annual ratio of AET to PET.

The calculation includes an upper bound in relative soil moisture
(`soilmstress_thetastar`), above which $\beta$ is always 1, corresponding
to no loss of light use efficiency.

```{code-cell} python
:tags: [hide-input]
# Calculate soil moisture stress factor
soilm = pmodel.calc_soilmstress(soilm_2d, meanalpha_2d.transpose())

# Create a contour plot of gamma
fig, ax = pyplot.subplots()
CS = ax.contour(soilm_1d, meanalpha_1d, soilm, colors='black',
                levels=np.append(np.linspace(0, 0.9, 10), [0.99, 0.999]))
ax.clabel(CS, inline=1, fontsize=10)
ax.set_title('Soil moisture stress factor')
ax.set_xlabel('Soil moisture fraction')
ax.set_ylabel('AET/PET')
pyplot.show()
```

## $J_{max}$ limitation

$J_{max}$ limitation is used to capture temperature dependency in the maximum
rate of RuBP regeneration. Three methods are implemented in the class
{class}`pyrealm.pmodel.CalcLUEVcmax` and set using the `method_jmaxlim`
argument:

- `wang17` (default, Wang Han et al. 2017, {meth}`pyrealm.pmodel.CalcLUEVcmax.wang17`)
- `smith19` (Smith et al., 2019, {meth}`pyrealm.pmodel.CalcLUEVcmax.smith19`)
- `none` (removes $J_{max}$ limitation, {meth}`pyrealm.pmodel.CalcLUEVcmax.none`)

Each method calculates a value for $m_{jlim}$, with $m_{jlim} = 1.0$ for the method
`none`. The plot below shows the effects of each method on the overall combined
impacts of light-limitation on assimilation: $m_j \cdot m_{jlim}$. In this
example, only temperature varies ($P=101325.0 , \ce{CO2}= 410 \text{ppm},
\text{VPD}=1.0$) and $\phi_0=0.05$.

```{code-cell} python
:tags: [hide-input]
# Calculate variation in m_jlim with temperature
# - calculate optimal chi under a temperature gradient
gammastar = pmodel.calc_gammastar(tc_1d, patm=pmodel_param.k_Po)
kmm = pmodel.calc_kmm(tc_1d, patm=pmodel_param.k_Po)
viscosity = pmodel.calc_viscosity_h2o(tc_1d, patm=pmodel_param.k_Po)
viscosity_std = pmodel.calc_viscosity_h2o(pmodel_param.k_To, pmodel_param.k_Po)
ns_star = viscosity / viscosity_std
ca = pmodel.calc_co2_to_ca(co2=410, patm=pmodel_param.k_Po)

# Compare Wang17 and Smith19
optchi = pmodel.CalcOptimalChi(kmm=kmm, gammastar=gammastar, 
                               ns_star=ns_star, ca=ca, vpd = 1)

lue_wang17 = pmodel.CalcLUEVcmax(optchi, kphio=0.05, method='wang17')
lue_smith19 = pmodel.CalcLUEVcmax(optchi, kphio=0.05, method='smith19')

# Create a line plot of the resulting values of m_j
pyplot.plot(tc_1d, optchi.mj, label='None')
pyplot.plot(tc_1d, optchi.mj * lue_wang17.mjlim, label='wang17')
pyplot.plot(tc_1d, optchi.mj * lue_smith19.mjlim, label='smith19')

pyplot.title('Effects of J_max limitation')
pyplot.xlabel('Temperature °C')
pyplot.ylabel('Light-assimilated limitation (m_j) factor')
pyplot.legend()
pyplot.show()
```



