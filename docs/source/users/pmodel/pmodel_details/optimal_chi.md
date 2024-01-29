---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: pyrealm_python3
  language: python
  name: pyrealm_python3
---

# Optimal $\chi$ and leaf $\ce{CO2}$

The next step is to estimate the following parameters:

* The ratio of carboxylation to transpiration cost factors (``beta``, $\beta$).
  In some approaches, this is taken as a fixed value, but other approaches apply
  penalties to $\beta$ based on environmental conditions. At present, these
  approaches are the `{cite:t}`lavergne:2020a` modifications with soil moisture content
  (``theta``, $\theta$) and the experimental root zone stress approaches.
* The value $\chi = c_i/c_a$, which is the unitless ratio of leaf internal $\ce{CO2}$
  partial pressure ($c_i$, Pa) to ambient $\ce{CO2}$ partial pressure ($c_a$, Pa).
* A parameter ($\xi$) describing the sensitivity of $\chi$ to vapour pressure deficit
  (VPD).
* $\ce{CO2}$ limitation factors to both light assimilation ($m_j$) and carboxylation
  ($m_c$) along with their ratio ($m_{joc} = m_j / m_c$).

The  {class}`~pyrealm.pmodel.optimal_chi` module implements the following methods for
calculating these values, providing options to handle C3 and C4 photosynthesis and
different implementations of water stress. The method names are used to select an
approach when fitting a {class}`~pyrealm.pmodel.pmodel.PModel`, and each is implemented
in a specific model class that provides the appropriate calculations. It is possible for
users to extend these methods to provide additional approaches.

```{list-table}
:header-rows: 1

* - Method name
  * Method class
* - `prentice14`
  * {class}`~pyrealm.pmodel.optimal_chi.OptimalChiPrentice14`
* - `c4`
  * {class}`~pyrealm.pmodel.optimal_chi.OptimalChiC4`
* - `c4_no_gamma`
  * {class}`~pyrealm.pmodel.optimal_chi.OptimalChiC4NoGamma`
* - `lavergne20_c3`
  * {class}`~pyrealm.pmodel.optimal_chi.OptimalChiLavergne20C3`
* - `lavergne20_c4`
  * {class}`~pyrealm.pmodel.optimal_chi.OptimalChiLavergne20C4`
* - `prentice14_rootzonestress`
  * {class}`~pyrealm.pmodel.optimal_chi.OptimalChiPrentice14RootzoneStress`
* - `c4_rootzonestress`
  * {class}`~pyrealm.pmodel.optimal_chi.OptimalChiC4RootzoneStress`
* - `c4_no_gamma_rootzonestress`
  * {class}`~pyrealm.pmodel.optimal_chi.OptimalChiC4NoGammaRootzoneStress`
```

```{code-cell} ipython3
:tags: [hide-input]

from itertools import product

import numpy as np
from matplotlib import pyplot
from matplotlib.lines import Line2D

from pyrealm.pmodel.optimal_chi import OptimalChiPrentice14
from pyrealm.pmodel import PModelEnvironment, PModel

# Create inputs for a temperature curve at:
# - two atmospheric pressures
# - two CO2 concentrations
# - two VPD values

n_pts = 31

tc_1d = np.linspace(-10, 40, n_pts)
patm_1d = np.array([101325, 80000])
vpd_1d = np.array([500, 2000])
co2_1d = np.array([280, 410])

tc_4d, patm_4d, vpd_4d, co2_4d = np.meshgrid(tc_1d, patm_1d, vpd_1d, co2_1d)

# Calculate the photosynthetic environment
pmodel_env = PModelEnvironment(tc=tc_4d, patm=patm_4d, vpd=vpd_4d, co2=co2_4d)

# A plotter function for a model
def plot_opt_chi(mod):

    # Create a list of combinations and line formats
    # (line col: PATM, style: CO2, marker used for VPD)

    idx_vals = {
        "vpd": zip([0, 1], vpd_1d),
        "patm": zip([0, 1], patm_1d),
        "co2": zip([0, 1], co2_1d),
    }

    idx_combos = list(product(*idx_vals.values()))
    line_formats = ["r-", "r--", "b-", "b--"] * 2

    # Create side by side subplots
    fig, (ax1, ax2, ax3) = pyplot.subplots(1, 3, figsize=(16, 5))

    # Loop over the variables
    for ax, var, label in (
        (ax1, "chi", "Optimal $\chi$"),
        (ax2, "mj", "$m_j$"),
        (ax3, "mc", "$m_c$"),
    ):

        # Get the variable to be plotted
        var_values = getattr(mod.optchi, var)

        # Plot line combinations
        for ((vdx, vvl), (pdx, pvl), (cdx, cvl)), lfmt in zip(idx_combos, line_formats):

            ax.plot(tc_1d, var_values[pdx, :, vdx, cdx], lfmt)

        # Annotate graph and force common y limits
        ax.set_title(f"Variation in {label}")
        ax.set_xlabel("Temperature °C")
        ax.set_ylabel(label)
        ax.set_ylim([-0.02, 1.02])

        # Add markers to note the two VPD inputs
        for vdx, mrkr in zip([0, 1], ["o", "^"]):

            mean_at_min_temp = var_values[:, 0, vdx, :].mean()
            ax.scatter(
                tc_1d[0] - 2,
                mean_at_min_temp,
                marker=mrkr,
                s=60,
                c="none",
                edgecolor="black",
            )

    # create a legend showing the combinations
    blnk = Line2D([], [], color="none")
    rd = Line2D([], [], linestyle="-", color="r")
    bl = Line2D([], [], linestyle="-", color="b")
    sld = Line2D([], [], linestyle="-", color="k")
    dsh = Line2D([], [], linestyle="--", color="k")
    circ = Line2D(
        [],
        [],
        marker="o",
        linestyle="",
        markersize=10,
        markeredgecolor="k",
        markerfacecolor="none",
    )
    trng = Line2D(
        [],
        [],
        marker="^",
        linestyle="",
        markersize=10,
        markeredgecolor="k",
        markerfacecolor="none",
    )

    ax1.legend(
        [blnk, blnk, blnk, rd, sld, circ, bl, dsh, trng],
        [
            "patm",
            "co2",
            "vpd",
            f"{patm_1d[0]} Pa",
            f"{co2_1d[0]} ppm",
            f"{vpd_1d[0]} Pa",
            f"{patm_1d[1]} Pa",
            f"{co2_1d[1]} ppm",
            f"{vpd_1d[1]} Pa",
        ],
        ncol=3,
        loc="upper left",
        frameon=False,
    )

    # pyplot.tight_layout();
```

## The `prentice14` method

This **C3 method** follows the approach detailed in {cite:t}`Prentice:2014bc`, see
{class}`~pyrealm.pmodel.optimal_chi.OptimalChiPrentice14` for details.

```{code-cell} ipython3
:tags: [hide-input]

# Run the P Model and plot predictions
pmodel_c3 = PModel(pmodel_env, method_optchi="prentice14")
plot_opt_chi(pmodel_c3)
```

## The `C4` method

This **C4_method** follows the approach detailed in {cite:t}`Prentice:2014bc`, but uses
a C4 specific version of the unit cost ratio ($\beta$). It also sets $m_j = m_c = 1$.
See {class}`~pyrealm.pmodel.optimal_chi.OptimalChiC4` for details.

```{code-cell} ipython3
:tags: [hide-input]

# Run the P Model and plot predictions
pmodel_c4 = PModel(pmodel_env, method_optchi="c4")
plot_opt_chi(pmodel_c4)
```

## The `c4_no_gamma` method

This method drops terms from the approach given in {cite:t}`Prentice:2014bc` to reflect
the assumption that photorespiration ($\Gamma^\ast$) is negligible in C4 photosynthesis.
It uses the same $\beta$ estimate as
{class}`~pyrealm.pmodel.optimal_chi.OptimalChiC4`
and also also sets $m_j = 1$, but $m_c$ is calculated as in
{class}`~pyrealm.pmodel.optimal_chi.OptimalChiPrentice14`. See
{meth}`~pyrealm.pmodel.optimal_chi.OptimalChiC4NoGamma` for details.

```{code-cell} ipython3
:tags: [hide-input]

# Run the P Model and plot predictions
pmodel_c4 = PModel(pmodel_env, method_optchi="c4_no_gamma")
plot_opt_chi(pmodel_c4)
```

## The `lavergne20_c3` and `lavergne20_c4` methods

These methods follow the approach detailed in {cite:t}`lavergne:2020a`, which fitted
an empirical model of $\beta$ for C3 plants as a function of volumetric soil moisture
($\theta$, m3/m3), using data from leaf gas exchange measurements. The C4 method takes
the same approach but with modified empirical parameters giving predictions of
$\beta_{C3} = 9 \times \beta_{C4}$. Following the approach of
{class}`~pyrealm.pmodel.optimal_chi.OptimalChiC4NoGamma`, $m_c$ is calculated
but $m_j=1$.

```{warning}
Note that {cite:t}`lavergne:2020a` found **no relationship** between C4 $\beta$
values and soil moisture in leaf gas exchange data  The
{class}`~pyrealm.pmodel.optimal_chi.OptimalChiLavergne20C4` method is **an 
experimental
feature** - see the documentation for the
{class}`~pyrealm.pmodel.optimal_chi.OptimalChiLavergne20C4` and
{class}`~pyrealm.pmodel.optimal_chi.OptimalChiC4` methods for the theoretical 
rationale.
```

### Variation in $\beta$ with soil moisture

The calculation details are provided in the description of the
{class}`~pyrealm.pmodel.optimal_chi.OptimalChiLavergne20C3` method, but the
variation in $\beta$ with $\theta$ is shown below.

```{code-cell} ipython3
:tags: [hide-input]

# Theta is required for the calculation of beta
from pyrealm.pmodel.optimal_chi import OptimalChiLavergne20C3, OptimalChiLavergne20C4

pmodel_env_theta_range = PModelEnvironment(
    tc=25, patm=101325, vpd=0, co2=400, theta=np.linspace(0, 0.8, 81)
)
opt_chi_lavergne20_c3 = OptimalChiLavergne20C3(pmodel_env_theta_range)
opt_chi_lavergne20_c4 = OptimalChiLavergne20C4(pmodel_env_theta_range)

# Plot the predictions
fig, ax1 = pyplot.subplots(1, 1, figsize=(6, 4))
ax1.plot(pmodel_env_theta_range.theta, opt_chi_lavergne20_c4.beta, label="C3")
ax1.plot(pmodel_env_theta_range.theta, opt_chi_lavergne20_c3.beta, label="C4")
ax1.set_xlabel(r"Soil moisture ($\theta$, m3/m3)")
ax1.set_ylabel(r"Unit cost ratio ($\beta$, -)")
ax1.legend()
pyplot.tight_layout()
```

### Estimation of optimal $\chi$

The plots below show the impacts on optimal $\chi$ across a temperature gradient for two
values of VPD and soil moisture, with constant atmospheric pressure (101325 Pa) and CO2
(280 ppm).

```{code-cell} ipython3
:tags: [hide-input]

# Environments with high and low soil moisture
theta_hi = 0.6
theta_lo = 0.1
pmodel_env_hi = PModelEnvironment(
    tc=tc_4d, patm=101325, vpd=vpd_4d, co2=co2_4d, theta=theta_hi
)
pmodel_env_lo = PModelEnvironment(
    tc=tc_4d, patm=patm_4d, vpd=vpd_4d, co2=co2_4d, theta=theta_lo
)

# Run the P Model and plot predictions
chi_lavc3_hi = OptimalChiLavergne20C3(pmodel_env_hi)
chi_lavc4_hi = OptimalChiLavergne20C4(pmodel_env_hi)
chi_lavc3_lo = OptimalChiLavergne20C3(pmodel_env_lo)
chi_lavc4_lo = OptimalChiLavergne20C4(pmodel_env_lo)

fig, (ax1, ax2) = pyplot.subplots(1, 2, figsize=(10, 5), sharey=True)

ax1.plot(
    tc_1d,
    chi_lavc3_hi.chi[0, :, 0, 0],
    "b-",
    label=f"VPD = {vpd_1d[0]}, theta={theta_hi}",
)
ax1.plot(
    tc_1d,
    chi_lavc3_hi.chi[0, :, 1, 0],
    "b--",
    label=f"VPD = {vpd_1d[1]}, theta={theta_hi}",
)
ax1.plot(
    tc_1d,
    chi_lavc3_lo.chi[0, :, 0, 0],
    "r-",
    label=f"VPD = {vpd_1d[0]}, theta={theta_lo}",
)
ax1.plot(
    tc_1d,
    chi_lavc3_lo.chi[0, :, 1, 0],
    "r--",
    label=f"VPD = {vpd_1d[1]}, theta={theta_lo}",
)
ax1.set_title(f"C3 plants (`lavergne20_c3`)")
ax1.set_ylabel(r"Optimal $\chi$")
ax1.set_xlabel("Temperature (°C)")
ax1.legend(frameon=False)

ax2.plot(tc_1d, chi_lavc4_hi.chi[0, :, 0, 0], "b-")
ax2.plot(tc_1d, chi_lavc4_hi.chi[0, :, 1, 0], "b--")
ax2.plot(tc_1d, chi_lavc4_lo.chi[0, :, 0, 0], "r-")
ax2.plot(tc_1d, chi_lavc4_lo.chi[0, :, 1, 0], "r--")
ax2.set_title(f"C4 plants (`lavergne20_c4`)")
ax2.set_xlabel("Temperature (°C)")

pyplot.tight_layout()
```

### $m_j$ and $m_c$

The plots below illustrate the impact of temperature and  $\theta$ on  $m_j$ and $m_c$,
again with constant atmospheric pressure (101325 Pa) and CO2 (280 ppm).

```{code-cell} ipython3
:tags: [hide-input]

fig, ((ax1, ax3), (ax2, ax4)) = pyplot.subplots(2, 2, figsize=(10, 10), sharey=True)

ax1.plot(
    tc_1d,
    chi_lavc3_hi.mj[0, :, 0, 0],
    "b-",
    label=f"VPD = {vpd_1d[0]}, theta={theta_hi}",
)
ax1.plot(
    tc_1d,
    chi_lavc3_hi.mj[0, :, 1, 0],
    "b--",
    label=f"VPD = {vpd_1d[1]}, theta={theta_hi}",
)
ax1.plot(
    tc_1d,
    chi_lavc3_lo.mj[0, :, 0, 0],
    "r-",
    label=f"VPD = {vpd_1d[0]}, theta={theta_lo}",
)
ax1.plot(
    tc_1d,
    chi_lavc3_lo.mj[0, :, 1, 0],
    "r--",
    label=f"VPD = {vpd_1d[1]}, theta={theta_lo}",
)
ax1.set_title(f"Variation in $m_j$ for C3 plants (`lavergne20_c3`)")
ax1.set_ylabel(r"$m_j$")
ax1.set_xlabel("Temperature (°C)")
ax1.legend(frameon=False)

ax2.plot(tc_1d, chi_lavc3_hi.mc[0, :, 0, 0], "b-")
ax2.plot(tc_1d, chi_lavc3_hi.mc[0, :, 1, 0], "b--")
ax2.plot(tc_1d, chi_lavc3_lo.mc[0, :, 0, 0], "r-")
ax2.plot(tc_1d, chi_lavc3_lo.mc[0, :, 1, 0], "r--")
ax2.set_title(f"Variation in $m_c$ for C3 plants (`lavergne20_c3`)")
ax2.set_ylabel(r"$m_c$")
ax2.set_xlabel("Temperature (°C)")

ax3.plot(
    tc_1d,
    chi_lavc4_hi.mj[0, :, 0, 0],
    "b-",
    label=f"VPD = {vpd_1d[0]}, theta={theta_hi}",
)
ax3.plot(
    tc_1d,
    chi_lavc4_hi.mj[0, :, 1, 0],
    "b--",
    label=f"VPD = {vpd_1d[1]}, theta={theta_hi}",
)
ax3.plot(
    tc_1d,
    chi_lavc4_lo.mj[0, :, 0, 0],
    "r-",
    label=f"VPD = {vpd_1d[0]}, theta={theta_lo}",
)
ax3.plot(
    tc_1d,
    chi_lavc4_lo.mj[0, :, 1, 0],
    "r--",
    label=f"VPD = {vpd_1d[1]}, theta={theta_lo}",
)
ax3.set_title(f"Variation in $m_j$ for C4 plants (`lavergne20_c4`)")
ax3.set_ylabel(r"$m_j$")
ax3.set_xlabel("Temperature (°C)")

ax4.plot(tc_1d, chi_lavc4_hi.mc[0, :, 0, 0], "b-")
ax4.plot(tc_1d, chi_lavc4_hi.mc[0, :, 1, 0], "b--")
ax4.plot(tc_1d, chi_lavc4_lo.mc[0, :, 0, 0], "r-")
ax4.plot(tc_1d, chi_lavc4_lo.mc[0, :, 1, 0], "r--")
ax4.set_title(f"Variation in $m_c$ for C4 plants (`lavergne20_c4`)")
ax4.set_ylabel(r"$m_c$")
ax4.set_xlabel("Temperature (°C)")

pyplot.tight_layout()
```
