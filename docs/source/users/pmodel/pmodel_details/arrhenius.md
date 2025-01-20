---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.5
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

<!-- markdownlint-disable MD041-->

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np

from pyrealm.constants.core_const import CoreConst
from pyrealm.constants.pmodel_const import PModelConst
from pyrealm.pmodel.functions import (
    calculate_simple_arrhenius_factor,
    calculate_kattge_knorr_arrhenius_factor,
)
```

<!-- markdownlint-enable MD041-->

# Arrhenius scaling in the P Model

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

These scaling factors are used extensively to calculate temperature scaling of:

* gammastar
* TODO

## Scaling of $V_{cmax}$ and $J_{max}$

One critical component of the P Model that requires Arrhenius scaling is the temperature
scaling of $V_{cmax}$ and $J_{max}$. This is central to the subdaily form of the P
Model, as Arrhenius factors are required twice to fit the model:

1. Conversion of daily realised rates at daily temperature to daily realised rates **at
   standard temperatures** ($V_{cmax25}$ and $J_{max25}$).
2. Conversion of subdaily estimates at standard temperatures back to **observed
   temperature** ($V_{cmax}$ and $J_{max}$).

The standard model calculates $V_{cmax}$ and $J_{max}$ directly, but the standardised
versions at 25°C are also often required and use the same conversion.

There is some discussion about the form of the scaling of reaction rates of $V_{cmax}$
and $J_{max}$ with temperature. The original description of the P Model {cite}`TODO`
used the simple scaling shown here, but other research suggests that these rates may
show peaked relationships with temperature (e.g. {citep}`kattgeknorr2007`).

The `method_arrhenius` argument to `PModel` and `SubdailyPModel` allows the form of the
scaling of $V_{cmax}$ and $J_{max}$ to be switched between different models. At present,
`pyrealm` implements two alternative options:

* `simple`: This uses the equation shown above, implemented in the function
  {meth}`~pyrealm.pmodel.functions.calculate_simple_arrhenius_factor`.
* `kattge_knorr`: This uses a peaked form of the relationship, implemented in the
  function {meth}`~pyrealm.pmodel.functions.calculate_kattge_knorr_arrhenius_factor`.
  This form requires the users to specify the growth temperature $t_g$, which modulates
  the location of the peak in the relationship.

:::{warning}
We currently **strongly** recommend the use of the `simple` method for day
to day use. The `kattge_knorr` method scaling is implemented for experimental purposes
only.
:::

The plot below shows the calculated factor using both of these forms. Two separate
growth temperatures are used with the `kattge_knorr` method. At present, the
implementation of the standard P Model in `rpmodel` uses the form of the `kattge_knorr`
method but sets $t_g=T$, rather than having a fixed growth temperature. This leads to
the curve labelled `rpmodel` in the plot, which does not have a peak.

```{code-cell} ipython3
pmodel_const = PModelConst()
core_const = CoreConst()

tc = np.arange(0, 40, 0.1)

simple = calculate_simple_arrhenius_factor(
    tk=tc + core_const.k_CtoK,
    tk_ref=pmodel_const.plant_T_ref + core_const.k_CtoK,
    ha=pmodel_const.arrhenius_vcmax["simple"]["ha"],
)


coef = pmodel_const.arrhenius_vcmax["kattge_knorr"]
kattge_knorr_10 = calculate_kattge_knorr_arrhenius_factor(
    tk_leaf=tc + core_const.k_CtoK,
    tc_growth=10,
    tk_ref=pmodel_const.plant_T_ref + core_const.k_CtoK,
    ha=coef["ha"],
    hd=coef["hd"],
    entropy_intercept=coef["entropy_intercept"],
    entropy_slope=coef["entropy_slope"],
)

kattge_knorr_20 = calculate_kattge_knorr_arrhenius_factor(
    tk_leaf=tc + core_const.k_CtoK,
    tc_growth=20,
    tk_ref=pmodel_const.plant_T_ref + core_const.k_CtoK,
    ha=coef["ha"],
    hd=coef["hd"],
    entropy_intercept=coef["entropy_intercept"],
    entropy_slope=coef["entropy_slope"],
)


rpmodel = calculate_kattge_knorr_arrhenius_factor(
    tk_leaf=tc + core_const.k_CtoK,
    tc_growth=tc + core_const.k_CtoK,  # t_g == T
    tk_ref=pmodel_const.plant_T_ref + core_const.k_CtoK,
    ha=coef["ha"],
    hd=coef["hd"],
    entropy_intercept=coef["entropy_intercept"],
    entropy_slope=coef["entropy_slope"],
)
```

```{code-cell} ipython3
plt.plot(tc, simple, label="Simple")
plt.plot(tc, kattge_knorr_10, label="Kattge Knorr ($t_g=10$°C)")
plt.plot(tc, kattge_knorr_20, label="Kattge Knorr ($t_g=20$°C)")
plt.plot(tc, rpmodel, linestyle="--", color="grey", label="rpmodel")
plt.legend(frameon=False)
plt.xlabel("Leaf temperature (°C)")
plt.tight_layout()
```
