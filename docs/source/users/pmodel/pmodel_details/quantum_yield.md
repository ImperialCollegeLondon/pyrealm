---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
language_info:
  codemirror_mode:
    name: ipython
    version: 3
  file_extension: .py
  mimetype: text/x-python
  name: python
  nbconvert_exporter: python
  pygments_lexer: ipython3
  version: 3.11.9
---

# Quantum yield efficiency of photosynthesis

```{code-cell} ipython3
:tags: [hide-input]

# This code loads required packages and then creates a representative range of
# values of the core variables to use in function plots.
#
# Note that the ranges are created (`_1d`) but are also cast to two dimensional
# arrays of repeating values (`_2d`) to generate response surfaces for functions
# with multuple inputs.

import matplotlib.pyplot as plt
import numpy as np
from pyrealm.pmodel import PModelEnvironment
from pyrealm.pmodel.pmodel import PModel
from pyrealm.pmodel.quantum_yield import QuantumYieldTemperature, QuantumYieldSandoval

# Set the resolution of examples
n_pts = 201

# Create a range of representative values for key inputs.
tc_1d = np.linspace(-25, 50, n_pts)
meanalpha_1d = np.linspace(0, 1, n_pts)
co2_1d = np.linspace(200, 500, n_pts)

# Broadcast the range into arrays with repeated values.
tc_2d = np.broadcast_to(tc_1d, (n_pts, n_pts))
meanalpha_2d = np.broadcast_to(meanalpha_1d, (n_pts, n_pts))
co2_2d = np.broadcast_to(co2_1d, (n_pts, n_pts))
```

:::{warning}

Note that $\phi_0$ is also sometimes used to refer to the quantum yield of electron
transfer, which is exactly four times larger than the quantum yield of photosynthesis.

:::

The value of $\phi_0$ captures the conversion rate of moles photosynthetically active
photons into moles of $\ce{CO2}$. The theoretical maximum for this value is 1/9, in the
absence of a Q cycle, or 1/8 when a Q cycle is operating {cite}`long:1993a`. These
theoretical maxima are not necessarily directly used in calculating light use
efficiency:

* The values of $\phi_0$ are often adjusted to include other components of light
capture. For example, {cite:t}`Stocker:2020dh` include a factor for incomplete leaf
absorptance in their estimation of $\phi_0$ and argue that $\phi_0$ should be treated as
a parameter representing canopy-scale effective quantum yield.

* The maximum quantum yield can vary with environmental conditions, such as temperature
variation in $\phi_0$ {cite}`Bernacchi:2003dc`.

For these reasons, the {class}`~pyrealm.pmodel.pmodel.PModel` provides alternative
approaches to estimating the value of $\phi{0}$, using the `method_kphio` argument. The
currently implemented approaches are described below.

All of the approaches share the same baseline reference value of $\phi{0}= 1/8$ as the
theoretical maximum quantum yield of photosynthesis. This can be overridden got all
analsyes using a given {class}`~pyrealm.pmodel.pmodel_environment.PModelEnvironmen` by
changing the value of
{attr}`PModelConst.maximum_phi0<pyrealm.constants.pmodel_const.PModelConst.maximum_phi0`,
but it can also be overriden for a single P Model using the `reference_kphio` argument.
This can be needed when specific methods have been calibrated using a 'tuned' value for
$\phi_0$: an example here is the soil moisture stress corrected GPP estimates of
:cite:t:`Stocker:2020dh` which use different $\phi_0$ values (see
{meth}`~pyrealm.pmodel.functions.calc_soilmstress_stocker`).

## Temperature dependent $\phi_0$

The default approach (`method_kphio='temperature'`) applies a temperature dependent
estimate of $\phi_0$, following {cite:t}`Bernacchi:2003dc` for C3 plants and
{cite:t}`cai:2020a` for C4 plants.

```{code-cell} ipython3
:tags: [hide-input]

# Calculate temperature dependence of quantum yield efficiency
env = PModelEnvironment(tc=tc_1d, patm=101325, vpd=820, co2=40, fapar=1, ppfd=1)

fkphio_c3 = QuantumYieldTemperature(env=env, use_c4=False)
fkphio_c4 = QuantumYieldTemperature(env=env, use_c4=True)

# Create a line plot of ftemp kphio
plt.plot(tc_1d, fkphio_c3.kphio, label="C3")
plt.plot(tc_1d, fkphio_c4.kphio, label="C4")

plt.title("Temperature dependence of quantum yield efficiency")
plt.xlabel("Temperature °C")
plt.ylabel("Quantum yield efficiency ($\phi_0$)")
plt.legend()
plt.show()
```

## Fixed $\phi_0$

This approach (`method_kphio='fixed'`) applies a fixed value of $\phi_0$ in the
calculation of light use efficiency.

However, the fixed method will also accept $\phi_0$ values for each observation being
fitted in the PModel. This option is provided to allow users to experiment with
alternative per-observation estimation of $\phi_0$ that are not currently implemented.
You will need to provide an array of values that has the same shape as the other driver
variables and these values are then used within the calculations for each observation.

In the code and plot below, this approach is used to provide a simple linear series of
$\phi_0$ values to an otherwise constant environment. As you would expect given
$\text{LUE} = \phi_0 \cdot M_C \cdot m_j$, light use efficiency changes linearly along
this gradient of $\phi_0$ values.

```{code-cell} ipython3
:tags: [hide-input]

# A constant environment to show a range of kphio values
kphio_values = np.arange(0, 0.126, step=0.001)
n_vals = len(kphio_values)

env = PModelEnvironment(
    tc=np.repeat(20, n_vals),
    patm=np.repeat(101325, n_vals),
    vpd=np.repeat(820, n_vals),
    co2=np.repeat(400, n_vals),
    fapar=np.repeat(1, n_vals),
    ppfd=np.repeat(1, n_vals),
)
model_var_kphio = PModel(env, method_kphio="fixed", reference_kphio=kphio_values)

# Create a line plot of ftemp kphio
plt.plot(kphio_values, model_var_kphio.lue)
plt.title("Variation in LUE with changing $\phi_0$")
plt.xlabel("$\phi_0$")
plt.ylabel("LUE")
plt.show()
```

## Temperature and aridity effects on $\phi_0$

The option `method_kphio='sandoval'` implements an experimental calculation
{cite}`sandoval:in_prep` of $\phi_0$ as a function of a local aridity index (P/PET), the
mean growth temperature and the air temperature {cite}`sandoval:in_prep`. You will
need to provide the aridity index and mean growing temperature for observations when
creating the `PModelEnvironment`.

First, the aridity index is used to adjust the reference value ($\phi_{0R}$) using a
double exponential function to calculate a new maximum value given the climatological
aridity ($\phi_{0A}$):

$\phi_{0A} = \dfrac{\phi_{0R}}{(1 + \textrm{AI}^m) ^ n}$

This captures a decrease in maximum $\phi_0$ in arid conditions, as shown below.

```{code-cell} ipython3
:tags: [hide-input]

n_vals = 51
aridity_index = np.logspace(-2, 1.5, num=n_vals)

env = PModelEnvironment(
    tc=np.repeat(20, n_vals),
    patm=np.repeat(101325, n_vals),
    vpd=np.repeat(820, n_vals),
    co2=np.repeat(400, n_vals),
    fapar=np.repeat(1, n_vals),
    ppfd=np.repeat(1, n_vals),
    aridity_index=aridity_index,
    mean_growth_temperature=np.repeat(20, n_vals),
)

sandoval_kphio = QuantumYieldSandoval(env)

fig, ax = plt.subplots(1, 1)
ax.plot(aridity_index, sandoval_kphio.kphio)
ax.set_title("Change in $\phi_0$ with aridity index (P/PET).")
ax.set_ylabel("$\phi_0$")
ax.set_xlabel("Aridity Index")
ax.set_xscale("log")
plt.show()
```

In addition to capping the peak $\phi_0$ as a function of the aridity index, this
approach also alters the temperature at which $\phi_0$ is maximised as a function of the
mean growth temperature ($T_g$) in a location. The plot below shows how aridity and mean
growth temperature interact to change the location and height of the peak $\phi_0$.

```{code-cell} ipython3
:tags: [hide-input]

n_vals = 51
mean_growth_values = np.array([10, 22, 24, 25])
aridity_values = np.array([1.0, 1.5, 5.0])
tc_values = np.linspace(0, 40, n_vals)

shape = (n_vals, len(aridity_values), len(mean_growth_values))

ai3, tc3, mg3 = np.meshgrid(aridity_values, tc_values, mean_growth_values)


env = PModelEnvironment(
    tc=tc3,
    patm=np.full(shape, 101325),
    vpd=np.full(shape, 820),
    co2=np.full(shape, 400),
    fapar=np.full(shape, 1),
    ppfd=np.full(shape, 1),
    aridity_index=ai3,
    mean_growth_temperature=mg3,
)

sandoval_kphio = QuantumYieldSandoval(env)

fig, axes = plt.subplots(ncols=3, nrows=1, sharey=True, figsize=(10, 6))

for ai_idx, (ax, ai_val) in enumerate(zip(axes, aridity_values)):

    for mg_idx, mg_val in enumerate(mean_growth_values):
        ax.plot(
            env.tc[:, ai_idx, mg_idx],
            sandoval_kphio.kphio[:, ai_idx, mg_idx],
            label=f"$T_{{g}}$ = {mg_val}",
        )
        ax.set_title(f"AI = {ai_val}")
        ax.set_ylabel("$\phi_0$")
        ax.set_xlabel("Observed temperature")


ax.legend(frameon=False)
plt.show()
```
