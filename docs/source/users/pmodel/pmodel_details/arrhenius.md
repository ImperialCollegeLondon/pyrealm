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
---

# Arrhenius scaling in the P Model

:::{warning}
This document discusses the form of the Arrhenius scaling used for estimating
temperature scaling of $V_{cmax}$ and $J_{max}$ in the P Model. Although `pyrealm`
provides flexibility for different forms, this is an active research area.

We currently **strongly** recommend the use of the `simple` method for Arrhenius scaling
when fitting a P Model. The `kattge_knorr` method scaling is implemented for
experimental purposes only.
:::

```{code-cell}
:tags: [hide-input]

import matplotlib.pyplot as plt
import numpy as np

from pyrealm.constants import CoreConst, PModelConst
from pyrealm.pmodel.functions import (
    calculate_simple_arrhenius_factor,
    calculate_kattge_knorr_arrhenius_factor,
)
```

The rates of enzyme kinetics in the `PModel` and `SubdailyPModel` vary with the
temperature of the enzyme reactions following an Arrhenius relationship scaling.

In most cases, the rates are described using a "simple" Arrhenius relationship. Given
the activation energy for the system ($H_a$), the universal gas constant $R$ and a
constant $c$, the rate at temperature $T$ is:

$$r(T) = \exp(c - H_a / (T R))$$

In general, rate calculations in the P Model use rate factors ($f$), relative to a fixed
reference temperature $T_0$:

$$
    \begin{align*}
        f &= \frac{r(T)}{r(T_0)} \\
          &= \exp \left( \frac{ H_a}{R} \cdot
                \left(\frac{1}{T_0} - \frac{1}{T}\right)\right)
    \end{align*}
$$

This simple scaling factor is appropriate for calculating temperature scaling of two key
enzyme systems:

* The photorespiratory CO2 compensation point (`gammastar`, $\Gamma^\ast$,
  {meth}`~pyrealm.pmodel.functions.calc_gammastar`).
* The Michaelis Menten coefficient of Rubisco-limited assimilation (`kmm`, $K_{MM}$,
  {meth}`~pyrealm.pmodel.functions.calc_kmm`).

## Scaling of $V_{cmax}$ and $J_{max}$

The simple scaling factor above is also used in the original description of the P Model
{cite:p}`Prentice:2014bc,Wang:2017go` for calculating Arrhenius scaling with temperature
of $V_{cmax}$ and $J_{max}$. This scaling is not required to calculate GPP, but *is*
used to calculate representative values of those rates at standard temperatures
($V_{cmax25}$ and $J_{max25}$) when required.

The simple scaling is also used in the original description of the subdaily P Model
{cite}`mengoli:2022a`. Here, the scaling is more central as Arrhenius scaling is used to
convert between realised daily estimates of acclimating $V_{cmax}$ and $J_{max}$ and the
values of $V_{cmax}$ and $J_{max}$ for subdaily observations:

1. Daily realised rates at daily representative temperatures are converted to daily
   realised rates **at standard temperatures** (daily realised $V_{cmax25}$ and
   $J_{max25}$).
2. Resulting predicted subdaily estimates at standard temperatures back to **observed
   temperature** ($V_{cmax}$ and $J_{max}$ at subdaily scales).

However, there is some discussion about the form of the scaling of reaction rates of
$V_{cmax}$ and $J_{max}$ with temperature, particularly the suggestion that these rates
should be "peaked", declining at higher temperatures. {cite:t}`Kattge:2007db` presented
a form of this peaked relationship, using the growing temperature ($T_g$) of the plant
to define the location of a peak. This form is used in the `rpmodel` implementation of
the P Model, although `rpmodel` currently sets the growth temperature to be equal to the
observed temperature (leaf or air temperature).

The plot below shows some examples of Arrhenius factor curves using these different
approaches. Two separate growth temperatures are used with the `kattge_knorr` method. At
present, the implementation of the standard P Model in `rpmodel` uses the form of the
`kattge_knorr` method but sets $t_g=T$, rather than having a fixed growth temperature.
This leads to the curve labelled `rpmodel` in the plot, which does not have a peak.

```{code-cell}
:tags: [hide-input]

# Define constants and a temperature range
pmodel_const = PModelConst()
core_const = CoreConst()
tc = np.arange(0, 40, 0.1)
tk = tc + core_const.k_CtoK

# Calculate the simple scaling factor
simple = calculate_simple_arrhenius_factor(
    tk=tk,
    tk_ref=pmodel_const.tk_ref,
    ha=pmodel_const.arrhenius_vcmax["simple"]["ha"],
)

# Calculate the Kattge Knorr curve under three conditions:
# 1) t_g = 10°C
coef = pmodel_const.arrhenius_vcmax["kattge_knorr"]
kattge_knorr_10 = calculate_kattge_knorr_arrhenius_factor(
    tk_leaf=tk,
    tc_growth=10,
    tk_ref=pmodel_const.tk_ref,
    coef=coef,
)

# 2) t_g = 20°C
kattge_knorr_20 = calculate_kattge_knorr_arrhenius_factor(
    tk_leaf=tk,
    tc_growth=20,
    tk_ref=pmodel_const.tk_ref,
    coef=coef,
)

# 3) rpmodel: t_g == T_leaf
rpmodel = calculate_kattge_knorr_arrhenius_factor(
    tk_leaf=tk,
    tc_growth=tk,
    tk_ref=pmodel_const.tk_ref,
    coef=coef,
)

plt.plot(tc, simple, label="Simple")
plt.plot(tc, kattge_knorr_10, label="Kattge Knorr ($t_g=10$°C)")
plt.plot(tc, kattge_knorr_20, label="Kattge Knorr ($t_g=20$°C)")
plt.plot(
    tc, rpmodel, linestyle="--", color="grey", label="Kattge Knorr in rpmodel ($t_g=T$)"
)
plt.legend(frameon=False)
plt.xlabel("Leaf temperature (°C)")
plt.tight_layout()
```

Note that we do not the additional factor $T/T_0$ in the calculation of the peaked form.
This was initially suggested by {cite:t}`murphy:2021a`, but was later agreed to be
unnecessary {cite:p}`stinziano:2021a,yin:2021a`

## Using different Arrhenius scaling

The `method_arrhenius` argument to `PModel` and `SubdailyPModel` allows the form of the
scaling of $V_{cmax}$ and $J_{max}$ to be switched between different models. At present,
`pyrealm` implements two alternative options:

* `simple`: This uses the equation shown above, implemented in the function
  {meth}`~pyrealm.pmodel.functions.calculate_simple_arrhenius_factor`.
* `kattge_knorr`: This uses a peaked form of the relationship, implemented in the
  function {meth}`~pyrealm.pmodel.functions.calculate_kattge_knorr_arrhenius_factor`. To
  use this method in a P Model, users need to provide values for the growth temperature
  $t_g$, which modulates the location of the peak in the relationship, when creating a
  PModelEnvironment instance. To duplicate the current behaviour of `rpmodel`, where
  $T=t_g$, the `mean_growth_temperature` can just be set to be the same as the
  air temperature.

  As noted at the top of the page, we **do not recommend this option** for normal use in
  the P Model.
