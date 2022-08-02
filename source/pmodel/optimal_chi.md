---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Step 2: Optimal $\chi$ and leaf $\ce{CO2}$

The next step is to estimate the following parameters:

* The value $\chi = c_i/c_a$, which is the unitless ratio of leaf internal $\ce{CO2}$
  partial pressure ($c_i$, Pa) to ambient $\ce{CO2}$ partial pressure ($c_a$, Pa).
* A parameter ($\xi$) describing the sensitivity of $\chi$ to vapour pressure deficit
  (VPD).
* $\ce{CO2}$ limitation factors to both light assimilation ($m_j$) and carboxylation
  ($m_c$) along with their ratio ($m_{joc} = m_j / m_c$).

The details of these calculations are in {class}`~pyrealm.pmodel.CalcOptimalChi`, which
implements a number of approaches to calculating these values:

```{code-cell}
:tags: [hide-input]

from itertools import product
from pyrealm import pmodel
import numpy as np
from matplotlib import pyplot
from matplotlib.lines import Line2D

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
pmodel_env = pmodel.PModelEnvironment(tc=tc_4d, patm=patm_4d, vpd=vpd_4d, co2=co2_4d)

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

## Method {meth}`~pyrealm.pmodel.CalcOptimalChi.prentice14`

This **C3 method** follows the approach detailed in {cite}`Prentice:2014bc`, see
{meth}`~pyrealm.pmodel.CalcOptimalChi.prentice14` for details.

```{code-cell}
:tags: [hide-input]

# Run the P Model and plot predictions
pmodel_c3 = pmodel.PModel(pmodel_env, method_optchi="prentice14")
plot_opt_chi(pmodel_c3)
```

## Method {meth}`~pyrealm.pmodel.CalcOptimalChi.c4`

This **C4_method** follows the approach detailed in {cite}`Prentice:2014bc`, but
uses a C4 specific version of the unit cost ratio ($\beta$). It also sets
$m_j = m_c = 1$.

See {meth}`~pyrealm.pmodel.CalcOptimalChi.c4` for details.

```{code-cell}
:tags: [hide-input]

# Run the P Model and plot predictions
pmodel_c4 = pmodel.PModel(pmodel_env, method_optchi="c4")
plot_opt_chi(pmodel_c4)
```

## Method {meth}`~pyrealm.pmodel.CalcOptimalChi.c4_no_gamma`

This method drops terms from the {cite}`Prentice:2014bc` to reflect the
assumption that photorespirations ($\Gamma^\ast$) is negligible in C4
photosynthesis. It uses the same $\beta$ estimate as
{meth}`~pyrealm.pmodel.CalcOptimalChi.c4` and also also sets $m_j = 1$, but
$m_c$ is calculated as in {meth}`~pyrealm.pmodel.CalcOptimalChi.prentice14`.

See {meth}`~pyrealm.pmodel.CalcOptimalChi.c4_no_gamma` for details.

```{code-cell}
:tags: [hide-input]

# Run the P Model and plot predictions
pmodel_c4 = pmodel.PModel(pmodel_env, method_optchi="c4_no_gamma")
plot_opt_chi(pmodel_c4)
```

## Methods {meth}`~pyrealm.pmodel.CalcOptimalChi.lavergne20_c3` and {meth}`~pyrealm.pmodel.CalcOptimalChi.lavergne20_c4`

These methods follow the approach detailed in {cite}`lavergne:2020a`, which fitted
an empirical model of $\beta$ for C3 plants as a function of volumetric soil moisture
($\theta$, m3/m3), using data from leaf gas exchange measurements. The C4 method takes
the same approach but with modified empirical parameters giving predictions of
$\beta_{C3} = 9 \times \beta_{C4}$, as in the {meth}`~pyrealm.pmodel.CalcOptimalChi.c4`
method.

```{warning}
Note that {cite}`lavergne:2020a` found **no relationship** between C4 $\beta$
values and soil moisture in leaf gas exchange data  The
{meth}`~pyrealm.pmodel.CalcOptimalChi.lavergne20_c4` method is **an experimental
feature** - see the documentation for the
{meth}`~pyrealm.pmodel.CalcOptimalChi.lavergne20_c4` and
{meth}`~pyrealm.pmodel.CalcOptimalChi.c4` methods for the theoretical rationale.
```

The calculation details are provided in the description of the
{meth}`~pyrealm.pmodel.CalcOptimalChi.lavergne20_c3` method, but the variation in
$\beta$ with $\theta$ is shown below.

```{code-cell}
:tags: [hide-input]

# Only theta is used in the calculation of beta
pmodel_env_theta_range = pmodel.PModelEnvironment(
    tc=25, patm=101325, vpd=0, co2=400, theta=np.linspace(0, 0.8, 81)
)
opt_chi_lavergne20_c3 = pmodel.CalcOptimalChi(
    pmodel_env_theta_range, method="lavergne20_c3"
)
opt_chi_lavergne20_c4 = pmodel.CalcOptimalChi(
    pmodel_env_theta_range, method="lavergne20_c4"
)
# Plot the predictions
fig, ax1 = pyplot.subplots(1, 1, figsize=(6, 4))
ax1.plot(pmodel_env_theta_range.theta, opt_chi_lavergne20_c4.beta, label="C3")
ax1.plot(pmodel_env_theta_range.theta, opt_chi_lavergne20_c3.beta, label="C4")
ax1.set_xlabel(r"Soil moisture ($\theta$, m3/m3)")
ax1.set_ylabel(r"Unit cost ratio ($\beta$, -)")
ax1.legend()
pyplot.tight_layout()
```

### Optimal $\chi$

The plots below show the impacts on optimal $\chi$ across a temperature gradient for two
values of VPD and soil moisture, with constant atmospheric pressure (101325 Pa) and CO2
(280 ppm).

```{code-cell}
:tags: [hide-input]

# Environments with high and low soil moisture
theta_hi = 0.6
theta_lo = 0.1
pmodel_env_hi = pmodel.PModelEnvironment(
    tc=tc_4d, patm=101325, vpd=vpd_4d, co2=co2_4d, theta=theta_hi
)
pmodel_env_lo = pmodel.PModelEnvironment(
    tc=tc_4d, patm=patm_4d, vpd=vpd_4d, co2=co2_4d, theta=theta_lo
)

# Run the P Model and plot predictions
chi_lavc3_hi = pmodel.CalcOptimalChi(pmodel_env_hi, method="lavergne20_c3")
chi_lavc4_hi = pmodel.CalcOptimalChi(pmodel_env_hi, method="lavergne20_c4")
chi_lavc3_lo = pmodel.CalcOptimalChi(pmodel_env_lo, method="lavergne20_c3")
chi_lavc4_lo = pmodel.CalcOptimalChi(pmodel_env_lo, method="lavergne20_c4")

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

As with the {meth}`~pyrealm.pmodel.CalcOptimalChi.c4` method, the
{meth}`~pyrealm.pmodel.CalcOptimalChi.lavergne20_c4` method set $m_j=m_c=1$, but the
plots below illustrate the impact of temperature and  $\theta$ on  $m_j$ and $m_c$ for
C3 plants, again with constant atmospheric pressure (101325 Pa) and CO2
(280 ppm).

```{code-cell}
:tags: [hide-input]

fig, (ax1, ax2) = pyplot.subplots(1, 2, figsize=(10, 5), sharey=False)

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

pyplot.tight_layout()
```
