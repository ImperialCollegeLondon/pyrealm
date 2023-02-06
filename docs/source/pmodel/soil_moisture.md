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
  name: python3
---

# Soil moisture stress

At present, there are three approaches for incorporating soil moisture effects on
photosynthesis:

* The Stocker $\beta(\theta)$ factor applied to light use efficiency, described below.
* The experimental `rootzonestress` argument to {class}`~pyrealm.pmodel.pmodel.PModel`,
  see below.
* The `lavergne20_c3` and `lavergne20_c4` methods for
  {class}`~pyrealm.params.pmodel.CalcOptimalChi`, which use an empirical model of the
  change in $\beta$ with soil moisture. See [here](optimal_chi) for details.

## Stocker $\beta(\theta)$

This is an empirically derived factor ($\beta(\theta) \in [0,1]$,
:{cite}`Stocker:2018be`, :{cite}`Stocker:2020dh`) that captures the response of light
use efficiency (LUE) and hence gross primary productivity (GPP)  to soil moisture
stress. The calculated value of $\beta(\theta)$ is applied directly as a penalty factor
to LUE and hence to estimates of GPP.

The factor requires estimates of:

* relative soil moisture ($m_s$, `soilm`), as the fraction of field capacity, and
* a measure of local mean aridity ($\bar{\alpha}$, `meanalpha`), as the average annual
  ratio of AET to PET.

The functions to calculate $\beta(\theta)$ are based on four parameters, derived from
experimental data and set in {class}`~pyrealm.params.PModelParams`:

* An upper bound in relative soil moisture ($\theta^\ast$, `soilmstress_thetastar`),
  above which $\beta$ is always 1, corresponding to no loss of light use efficiency.
* An lower bound in relative soil moisture ($\theta_0$, `soilmstress_theta0`),
  below which LUE is always zero.
* An intercept (a, `soilmstress_a`) for the aridity sensitivity parameter $q$.
* A slope (b, `soilmstress_b`) for the aridity sensitivity parameter $q$.

The aridity measure (($\bar{\alpha}$) is first used to set an aridity sensitivity
parameter ($q$), which sets the speed with which $\beta(\theta) \to 0$ as $m_s$
decreases.

$$
    q = (1 - (a + b \bar{\alpha}))/(\theta^\ast - \theta_{0})^2
$$

Then, relative soil moisture ($m_s$) is used to calculate the soil moisture factor:

$$`
    \beta(\theta) = q ( m_s - \theta^\ast) ^ 2  + 1
$$

The figures below shows how the aridity sensitivity parameter ($q$) changes with
differing aridity measures and then how the soil moisture factor $\beta(\theta)$
varies with changing soil moisture for some different values of mean aridity. In
the examples below, the default $\theta_0 = 0$ has been changed to $\theta_0 =
0.1$ to make the lower bound more obvious.

```{code-cell}
:tags: [hide-input]

from matplotlib import pyplot as plt
import numpy as np
from pyrealm import pmodel
from pyrealm.param_classes import PModelParams

# change default theta0 parameter
par_def = PModelParams(soilmstress_theta0=0.1)

# Calculate q
mean_alpha_seq = np.linspace(0, 1, 101)

q = (1 - (par_def.soilmstress_a + par_def.soilmstress_b * mean_alpha_seq)) / (
    par_def.soilmstress_thetastar - par_def.soilmstress_theta0
) ** 2

# Create a 1x2 plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot q ~ mean_alpha
ax1.plot(mean_alpha_seq, q)
ax1.set_xlabel(r"Mean aridity, $\bar{\alpha}$")
ax1.set_ylabel(r"Aridity sensitivity parameter, $q$")

# Plot beta(theta) ~ m_s for 5 values of mean alpha
soilm = np.linspace(0, 0.7, 101)

for mean_alpha in [0.9, 0.5, 0.3, 0.1, 0.0]:

    soilmstress = pmodel.calc_soilmstress(
        soilm=soilm, meanalpha=mean_alpha, pmodel_params=par_def
    )
    ax2.plot(soilm, soilmstress, label=r"$\bar{{\alpha}}$ = {}".format(mean_alpha))

ax2.axvline(x=par_def.soilmstress_thetastar, linestyle="--", color="black")
ax2.axvline(x=par_def.soilmstress_theta0, linestyle="--", color="black")

secax = ax2.secondary_xaxis("top")
secax.set_xticks(
    ticks=[par_def.soilmstress_thetastar, par_def.soilmstress_theta0],
    labels=[r"$\theta^\ast$", r"$\theta_0$"],
)

ax2.legend()
ax2.set_xlabel(r"Relative soil moisture, $m_s$")
ax2.set_ylabel(r"Empirical soil moisture factor, $\beta(\theta)$")


plt.show()
```

### Application of the factor

The factor can be applied to the P Model by using
{func}`~pyrealm.pmodel.calc_soilmstress` to calculate the factor values and then
passing them into {class}`~pyrealm.pmodel.pmodel.PModel` using the `soilmstress`
argument. The example below shows how the predicted light use efficiency from
the P Model changes across an aridity gradient both with and without the
soil moisture factor.

```{code-cell}
# Calculate a model without water stress in a constant environment
# and across an soil moisture gradient.

tc = np.array([20] * 101)
sm_gradient = np.linspace(0, 0.7, 101)
sm_stress = pmodel.calc_soilmstress(soilm=sm_gradient, meanalpha=0.5)

env = pmodel.PModelEnvironment(tc=tc, patm=101325.0, vpd=820, co2=400)

# Fix the kphio as the defaults change when soilmstress is used
model = pmodel.PModel(env, kphio=0.08)
model.estimate_productivity(fapar=1, ppfd=1000)

model_stress = pmodel.PModel(env, soilmstress=sm_stress, kphio=0.08)
model_stress.estimate_productivity(fapar=1, ppfd=1000)
```

```{code-cell}
:tags: [hide-input]

plt.plot(sm_gradient, model.lue, label="No soil moisture stress")
plt.plot(sm_gradient, model_stress.lue, label="Soil moisture stress applied")

plt.xlabel(r"Relative soil moisture, $m_s$, -")
plt.ylabel(r"Light use efficiency, gC mol-1")

plt.show()
```

```{warning}
In the `rpmodel` implementation, the soil moisture factor is also used
to modify $V_{cmax}$ and $J_{max}$, so that these values are congruent
with the resulting penalised LUE and GPP.

This is **not implemented in {class}`~pyrealm.pmodel.pmodel.PModel`**. The
empirical correction is applied only to LUE and hence GPP. A warning is generated
when accessing $V_{cmax}$, $J_{max}$ and predictions deriving from those
values if this soil moisture stress factor has been applied.
```

```{code-cell}
# Jmax warns that it has not been corrected for soil moisture
print(model_stress.jmax[0])
```

```{code-cell}
# Vcmax warns that it has not been corrected for soil moisture
print(model_stress.vcmax[0])
```

## The `rootzonestress` factor

```{warning}
This approach is **an experimental feature** - see the
{class}`~pyrealm.pmodel.pmodel.PModel` documentation. Essentially, the values for
`rootzonestress` apply a penalty factor directly to $\beta$ in the calculation of
optimal $\chi$. This factor is currently calculated externally to the `pyrealm` package
and is not documented here.
```
