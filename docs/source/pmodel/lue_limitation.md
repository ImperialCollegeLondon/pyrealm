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

# Step 3: LUE Limitation

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

Once key [photosynthetic parameters](photosynthetic_environment) and [optimal
chi](optimal_chi) have been calculated, the {class}`~pyrealm.pmodel.PModel`
class can report estimates of:

* the light use efficiency (LUE), as grams of carbon per mole of photons, and
* the intrinsic water use efficiency (IWUE), as micromoles per mole of photons.

## Light use efficiency

In its simplest form:

$$
  \text{LUE} = \phi_0 \cdot M_C \cdot m_j
$$

where $\phi_0$ is the quantum yield efficiency of photosynthesis, $M_C$ is the
molar mass of carbon and $m_j$ is the $\ce{CO2}$ limitation term of light use
efficiency from the calculation of optimal $\chi$.

However, the {mod}`pyrealm.pmodel` module also incorporates three further
factors:

* temperature (t) dependence of $\phi_0$,
* $J_{max}$ limitation of $m_j$ by a factor $f_v$ and
* an empirical soil moisture stress penalty on LUE.

$$
  \text{LUE} = \phi_0(t) \cdot M_C \cdot m_j \cdot f_v \cdot \beta(\theta)$
$$

### $\phi_0$ and temperature dependency

The {class}`~pyrealm.pmodel.PModel` uses a single variable to capture the
apparent quantum yield efficiency of photosynthesis (`kphio`, $\phi_0$).

```{warning}

Note that $\phi_0$ is sometimes used to refer to the quantum yield of electron
transfer, which is exactly four times larger than the quantum yield of
photosynthesis. 

```

The value of $\phi_0$ shows temperature dependence, which is modelled
following {cite}`Bernacchi:2003dc` for C3 plants and {cite}`cai:2020a` for C4
plants (see {func}`calc_ftemp_kphio`). The temperature dependency is applied by
default but can be turned off using the {class}`~pyrealm.pmodel.PModel` argument
`do_ftemp_kphio=False`.

The default values of `kphio` vary with the model options, corresponding
to the empirically fitted values presented for three setups in {cite}`Stocker:2020dh`.

1. If the temperature dependence of $\phi_0$ is **not** applied,
    $\phi_0 = 0.049977$,
1. otherwise, if an [empirical soil moisture stress factor](soil_moisture)
   is being applied, $\phi_0 = 0.87182$
1. otherwise, with no soil moisture stress and temperature dependence
   $\phi_0 = 0.081785$

The initial value of $\phi_0$ and the values used in calculations are stored in
the `init_kphio` and  `kphio` attributes of the {class}`~pyrealm.pmodel.PModel`
object.  The code examples compare models with and without temperature
dependency of $\phi_0$.

```{code-cell} ipython3
env = pmodel.PModelEnvironment(tc=30, patm=101325, vpd=820, co2=400)
model_fixkphio = pmodel.PModel(env, kphio=0.08, do_ftemp_kphio=False)
np.array([model_fixkphio.init_kphio, model_fixkphio.kphio])
```

```{code-cell} ipython3
model_tempkphio = pmodel.PModel(env, kphio=0.08, do_ftemp_kphio=True)
np.array([model_tempkphio.init_kphio, model_tempkphio.kphio])
```

The scaling of temperature dependence varies for C3 and C4 plants and the
function {func}`calc_ftemp_kphio` is used to calculate a limitation factor that
is applied to $\phi_0$.

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

### Limitation of electron transfer rate ($J_{max}$) and carboxylation capacity ($V_{cmax}$)

The {class}`~pyrealm.pmodel.PModel` implements three alternative approaches to
the calculation of $J_{max}$ and $V_{cmax}$, using the argument
`method_jmaxlim`. These options set the calculation of two factor ($f_j$ and
$f_v$) which are applied to the calculation of $J_{max}$ and $V_{cmax}$. The
options for this setting are:

* `simple`: These are the 'simple' formulations of the P Model, with $f_j = f_v
  = 1$.
* `wang17`: This is the default setting for `method_jmaxlim` and applies the
  calculations describe in  {cite}`Wang:2017go`. The calculation details can be
  seen in the {meth}`~pyrealm.pmodel.JmaxLimitation.wang17` method.

* `smith19`: This is an alternate calculation for optimal values of $J_{max}$
  and $V_{cmax}$ described in {cite}`Smith:2019dv`. The calculation details can be
  seen in the {meth}`~pyrealm.pmodel.JmaxLimitation.smith19` method.

```{code-cell} ipython3
model_jmax_simple = pmodel.PModel(env, kphio=0.08, method_jmaxlim='simple')
model_jmax_wang17 = pmodel.PModel(env, kphio=0.08, method_jmaxlim='wang17')
model_jmax_smith19 = pmodel.PModel(env,  kphio=0.08, method_jmaxlim='smith19')

# Compare LUE from the three methods
np.array([model_jmax_simple.lue,
          model_jmax_wang17.lue,
          model_jmax_smith19.lue])
```

The plot below shows the effects of each method on the LUE across a temperature
gradient ($P=101325.0 , \ce{CO2}= 400 \text{ppm}, \text{VPD}=820$) and $\phi_0=0.05$).

```{code-cell} python
:tags: [hide-input]
# Calculate variation in m_jlim with temperature
env = pmodel.PModelEnvironment(tc = tc_1d, patm=101325, vpd=820, co2=400)
model_tc_wang17 = pmodel.PModel(env, kphio=0.08, do_ftemp_kphio=False)
model_tc_simple = pmodel.PModel(env, kphio=0.08, do_ftemp_kphio=False, method_jmaxlim='simple')
model_tc_smith19 = pmodel.PModel(env, kphio=0.08, do_ftemp_kphio=False, method_jmaxlim='smith19')

# Create a line plot of the resulting values of m_j
pyplot.plot(tc_1d, model_tc_simple.lue, label='simple')
pyplot.plot(tc_1d, model_tc_wang17.lue, label='wang17')
pyplot.plot(tc_1d, model_tc_smith19.lue, label='smith19')

pyplot.title('Effects of J_max limitation')
pyplot.xlabel('Temperature °C')
pyplot.ylabel('Light Use Efficiency (g C mol-1)')
pyplot.legend()
pyplot.show()
```

### Soil moisture stress

This approach to handling soil moisture effects is presented
[here](soil_moisture.md).