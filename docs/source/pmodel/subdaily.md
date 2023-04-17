---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: pyrealm_python3
  language: python
  name: pyrealm_python3
---

# The P Model with fast and slow responses

The standard [P Model](pmodel.md) assumes that plants are able to instantaneously
respond optimally to their environmental conditions. This is a reasonable approximation
when using forcing variables that capture average conditions over longer time scales
such as weekly, monthly or coarser time steps. However, at finer temporal scales - and
particularly when trying to describe photosynthetic behaviour at subdaily timescales -
it is essential to account for fast and slow responses to changing environmental
conditions.

Fast responses
: At timescales of minutes,  include the opening and closing of stomata in response to
  changing conditions and changes in enzyme kinetics in response to changing conditions.

Slow responses
: At timescales of around a fortnight, include the adaptation of three key parameters
  that acclimate over time to changing environmental conditions.

* $\xi$: the sensitivity of the optimal $\chi$ ratio to the vapour pressure deficit.
* $V_{cmax25}$: the maximum rate of carboxylation at standard temperature,
* $J_{max25}$: the maximum rate of electron transport at standard temperature, and

  The $\xi$ parameter captures the plant response to changing hygroscopic conditions.
  The other parameters control the shape of the $A$/$C_i$ curve for photosynthesis:
  $V_{cmax25}$ constrains the Rubisco-limited assimilation rate ($A_c$) and $J_{max25}$
  constrains the electron transport rate limited assimilation rate ($A_J$).

The approach has the following steps:

* The [photosynthetic environment](pmodel_details/photosynthetic_environment) for the
  data is calculated as for the standard P Model. This estimates the fast responses of
  $\Gamma^*$, $K_{mm}$, $\eta^*$, and $c_a$.

* A [daily window](memory_effect.md#the-acclimation-window) is then set to define the
  conditions towards which the slow responses will acclimate, typically noon conditions
  that optimise light use efficiency during the daily period of highest photosynthetic
  photon flux density (PPFD). This is used to calculate a daily time series of average
  conditions during this acclimation window.

* A standard P model is used to estimate *optimal* behaviour during the daily
  acclimation conditions. A [memory
  effect](memory_effect.md#estimating-realised-responses) is then applied to the optimal
  daily estimates of $\xi$, $V_{cmax25}$ using a rolling weighted mean to estimate the
  slow *realised* responses of these parameters.

* The daily realised values are then
  [interpolated](memory_effect.md#interpolation-of-realised-values-to-subdaily-timescales)
  back to the subdaily time scale and combined with the estimated fast responses of
  other variables to calculate GPP at subdaily timescales.

This implementation largely follows the weighted average method of
{cite:t}`mengoli:2022a`, but is modified to include slow responses in $\xi$.

```{code-cell} ipython3
:tags: [hide-input]

from importlib import resources

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

from pyrealm.pmodel import (
    FastSlowScaler,
    memory_effect,
    FastSlowPModel,
    PModelEnvironment,
    PModel,
)
from pyrealm.constants import PModelConst
from pyrealm.pmodel.functions import calc_ftemp_arrh, calc_ftemp_kphio
```

## Demonstration dataset

The code below uses half hourly data from 2014 for the [BE-Vie FluxNET
site](https://fluxnet.org/doi/FLUXNET2015/BE-Vie), which was also used as a
demonstration in {cite:t}`mengoli:2022a`.

```{code-cell} ipython3

data_path = resources.files("pyrealm").parent / "data" / "subdaily_BE_Vie_2014.csv"
data = np.genfromtxt(
    data_path,
    names=True,
    delimiter=",",
    dtype=None,
    encoding="UTF8",
    missing_values = "NA",
)

# Extract the key half hourly timestep variables
temp_subdaily = data["ta"]
vpd_subdaily = data["vpd"]
co2_subdaily = data["co2"]
patm_subdaily = data["patm"]
ppfd_subdaily = data["ppfd"]
fapar_subdaily = data["fapar"]
datetime_subdaily = data["time"].astype(np.datetime64)
```

This dataset can then be used to calculate the photosynthetic environment at the
subdaily timescale. The code below also estimates GPP under the standard P Model with no
slow responses for comparison.

```{code-cell} ipython3
# Calculate the photosynthetic environment
subdaily_env = PModelEnvironment(
    tc=temp_subdaily,
    vpd=vpd_subdaily,
    co2=co2_subdaily,
    patm=patm_subdaily,
)

# Fit the standard P Model
pmodel_subdaily = PModel(subdaily_env, kphio=1 / 8)
pmodel_subdaily.estimate_productivity(ppfd=ppfd_subdaily, fapar=fapar_subdaily)
pmodel_subdaily.summarize()
```

The code below then fits a P Model including slow responses, which requires the
definition of a daily acclimation window, identifying the daily conditions that will
lead to optimal overall productivity. While acclimating to average daytime environment
might give better overall *light use efficiency* across the day, *productivity* is
optimised by acclimating to the conditions when PPFD is high.

A decision needs to be made about when those conditions occur during the day and how
best to sample those conditions. Typically those might be the observed environmental
conditions at the observation closest to noon, or the mean environmental conditions in a
window around noon.

```{code-cell} ipython3
# Create the fast slow scaler
fsscaler = FastSlowScaler(datetime_subdaily)

# Set the acclimation window as the values within a one hour window centred on noon
fsscaler.set_window(
    window_center=np.timedelta64(12, "h"),
    half_width=np.timedelta64(30, "m"),
)

# Fit the P Model with fast and slow responses
pmodel_fastslow = FastSlowPModel(
    env=subdaily_env,
    fs_scaler=fsscaler,
    ppfd=ppfd_subdaily,
    fapar=fapar_subdaily,
)
```

```{code-cell} ipython3
:tags: [hide-input]

idx = np.arange(48 * 120, 48 * 130)
plt.figure(figsize=(10, 4))
plt.plot(
    datetime_subdaily[idx], pmodel_subdaily.gpp[idx], label="Instantaneous model"
)
plt.plot(datetime_subdaily[idx], pmodel_fastslow.gpp[idx], "r-", label="Slow responses")
plt.ylabel = "GPP"
plt.legend(frameon=False)
plt.show()
```

## Calculation of GPP using fast and slow responses

The {class}`~pyrealm.pmodel.subdaily.FastSlowPModel` implements the calculations used to
estimate GPP using slow responses, but the details of these calculations are shown
below.

### Optimal responses during the acclimation window

The daily average conditions during the acclimation window can be sampled and used as
inputs to the standard P Model to calculate the optimal behaviour of plants under those
conditions.

```{code-cell} ipython3
# Get the daily acclimation conditions for the forcing variables
temp_acclim = fsscaler.get_daily_means(temp_subdaily)
co2_acclim = fsscaler.get_daily_means(co2_subdaily)
vpd_acclim = fsscaler.get_daily_means(vpd_subdaily)
patm_acclim = fsscaler.get_daily_means(patm_subdaily)
ppfd_acclim = fsscaler.get_daily_means(ppfd_subdaily)
fapar_acclim = fsscaler.get_daily_means(fapar_subdaily)

# Fit the P Model to the acclimation conditions
daily_acclim_env = PModelEnvironment(
    tc=temp_acclim, vpd=vpd_acclim, co2=co2_acclim, patm=patm_acclim
)

pmodel_acclim = PModel(daily_acclim_env, kphio=1 / 8)
pmodel_acclim.estimate_productivity(fapar=fapar_acclim, ppfd=ppfd_acclim)
```

### Slow responses of $\xi$, $J_{max25}$ and $V_{cmax25}$

Rather than being able to instantaneously adopt optimal values, the  $\xi$, $J_{max25}$
and $V_{cmax25}$ parameters are assumed to acclimate towards optimal values with a
lagged response using a [memory effect](memory_effect.md#estimating-realised-responses).

#### Calculation of $J_{max}$ and $V_{cmax}$ at standard temperature

The daily optimal acclimation values are obviously calculated under a range of
temperatures so $J_{max}$ and $V_{cmax}$ must first be standardised to expected values
at 25°C. This is acheived by multiplying by the reciprocal of the exponential part of
the Arrhenius equation ($h^{-1}$ in {cite}`mengoli:2022a`).

```{code-cell} ipython3
# Are these any of the existing values in the constants?
ha_vcmax25 = 65330
ha_jmax25 = 43900

tk_acclim = temp_acclim + pmodel_subdaily.const.k_CtoK
vcmax25_acclim = pmodel_acclim.vcmax * (1 / calc_ftemp_arrh(tk_acclim, ha_vcmax25))
jmax25_acclim = pmodel_acclim.jmax * (1 / calc_ftemp_arrh(tk_acclim, ha_jmax25))
```

#### Calculation of realised values

The memory effect can now be applied to the three parameters with slow
responses to calculate realised values, here using the default 15 day window.

```{code-cell} ipython3
# Calculation of memory effect in xi, vcmax25 and jmax25
xi_real = memory_effect(pmodel_acclim.optchi.xi, alpha=1 / 15)
vcmax25_real = memory_effect(vcmax25_acclim, alpha=1 / 15)
jmax25_real = memory_effect(jmax25_acclim, alpha=1 / 15)
```

The plots below show the instantaneously acclimated values for  $J_{max25}$,
$V_{cmax25}$ and $\xi$ in grey along with the realised slow reponses.
applied.

```{code-cell} ipython3
:tags: [hide-input]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for (ax, inst, mem, title) in zip(
    axes,
    (vcmax25_acclim, jmax25_acclim, pmodel_acclim.optchi.xi),
    (vcmax25_real, jmax25_real, xi_real),
    (r"$V_{cmax25}$", r"$J_{max25}$", r"$\xi$"),
):

    ax.plot(fsscaler.observation_dates, inst, "0.8", label="Optimal")
    ax.plot(fsscaler.observation_dates, mem, "r-", label="Realised")
    ax.set_title(title)
    ax.legend(frameon=False)
```

### Subdaily model including fast and slow responses

The last stage is to recalculate P model predictions on the subdaily timescale using
the realised slow responses for $\xi$, $J_{max25}$ and $V_{cmax25}$.

#### Calculation of fast responses in $J_{max}$ and $V_{cmax}$

Although the maximum rates at standard temperature $J_{max25}$ and $V_{cmax25}$ exhibit
slow reponses, the values of $J_{max}$ and $V_{cmax}$ will respond to changes in
temperature at fast scales:

* The realised daily values of $J_{max25}$ and $V_{cmax25}$ are interpolated from the
  acclimation window to the subdaily time scale.
* These values are adjusted to the actual half hourly temperatures to give the fast responses
  of $J_{max}$ and $V_{cmax}$.

```{code-cell} ipython3
tk_subdaily = subdaily_env.tc + pmodel_subdaily.const.k_CtoK

# Fill the realised jmax and vcmax from subdaily to daily
vcmax25_subdaily = fsscaler.fill_daily_to_subdaily(vcmax25_real)
jmax25_subdaily = fsscaler.fill_daily_to_subdaily(jmax25_real)

# Adjust to actual temperature at subdaily timescale
vcmax_subdaily = vcmax25_subdaily * calc_ftemp_arrh(tk=tk_subdaily, ha=ha_vcmax25)
jmax_subdaily = jmax25_subdaily * calc_ftemp_arrh(tk=tk_subdaily, ha=ha_jmax25)
```

#### Calculation of $c_i$

The subdaily variation in $c_i$ can now be calculated using $c_a$ and fast reponses in
$\Gamma^\ast$ with the realised slow responses of $\xi$. The original implementation of
{cite:t}`mengoli:2022a` here used  optimal values from the acclimation window of $\xi$,
$\Gamma^{\ast}$ and $c_a$, interpolated to the subdaily timescale and the actual
subdaily variation in VPD.

```{code-cell} ipython3
# Interpolate xi to subdaily scale
xi_subdaily = fsscaler.fill_daily_to_subdaily(xi_real)

# Calculate ci
ci_subdaily = (
    xi_subdaily * subdaily_env.ca + subdaily_env.gammastar * np.sqrt(subdaily_env.vpd)
) / (xi_subdaily + np.sqrt(subdaily_env.vpd))
```

#### Calculation of assimilation and GPP

Predictions for $A_j$, $A_c$ and GPP can then now be calculated as in the standard P
Model, where $c_i$ includes the slow responses of $\xi$ and $V_{cmax}$ and $J_{max}$
include the slow responses of $V_{cmax25}$ and $J_{max25}$ and fast responses to
temperature.

```{code-cell} ipython3
# Calculate Ac
Ac_subdaily = (
    vcmax_subdaily
    * (ci_subdaily - subdaily_env.gammastar)
    / (ci_subdaily + subdaily_env.kmm)
)

# Calculate J and Aj
phi = (1 / 8) * calc_ftemp_kphio(tc=temp_subdaily)
iabs = fapar_subdaily * ppfd_subdaily

J_subdaily = (4 * phi * iabs) / np.sqrt(1 + ((4 * phi * iabs) / jmax_subdaily) ** 2)

Aj_subdaily = (
    (J_subdaily / 4)
    * (ci_subdaily - subdaily_env.gammastar)
    / (ci_subdaily + 2 * subdaily_env.gammastar)
)

# Calculate GPP and convert from micromols to micrograms
GPP_subdaily = np.minimum(Ac_subdaily, Aj_subdaily) * PModelConst.k_c_molmass

# Compare to the FastSlowPModel outputs
diff = GPP_subdaily - pmodel_fastslow.gpp
print(np.nanmin(diff), np.nanmax(diff))
```
