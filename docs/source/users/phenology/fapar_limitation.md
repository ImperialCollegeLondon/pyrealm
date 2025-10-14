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

# Estimating maximum annual $f_{APAR}$

```{code-cell} ipython3
:tags: [hide-input]

from importlib import resources

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pyrealm.constants import PhenologyConst
from pyrealm.phenology.fapar_limitation import FaparLimitation
from pyrealm.pmodel import PModelEnvironment, PModel
```

The {class}`~pyrealm.phenology.fapar_limitation.FaparLimitation` class is used to
calculate annual maximum values for both $f_{APAR}$ and leaf area index ($L$), following
the method described by {cite:t}`cai:2025a`.

The maximum annual fAPAR is limited by the ability of plants to assimilate carbon for
constructing leaves and this can be limited either by the availability of light energy
($f_{APAR_{c}}$) or by the availability of water ($f_{APAR_{w}}$). The maximum annual
fAPAR is calculated as the minimum of those two terms. The equations are:

$$
\begin{align*}
f_{APAR_{c}} &= 1 - \frac{z}{k A_0}\\
f_{APAR_{w}} &= \left(\frac{ c_a \left( 1 - \chi \right)}{ 1.6 D }\right)
        \left(\frac{ f_0 P }{ A_0 }\right) \\
f_{APAR_{max}} &= \min{\left(f_{APAR_{c}}, f_{APAR_{w}}\right)}
\end{align*}
$$

In addition to $f_{APAR_{max}}$, the approach also estimates the maximum annual leaf
area index ($L_{max}$) and the steady-state annual ratio of leaf area index to GPP
($m$):

$$
\begin{align*}
        L_{max} &= - ( 1 / k ) \ln {1 -f_{APAR_{max}}} \\
        m &= \frac{ \sigma \cdot G  \cdot L_{max}}{A_0  \cdot f_{APAR_{max}}}
\end{align*}
$$

Six of the terms above are annual estimates for locations of:

* The ambient CO2 partial pressure during the growing season ($c_a$, Pa).
* The annual mean vapour pressure deficit during the growing season ($D$, Pa)
* The annual total precipitation ($P$, $\text{mol m}^{-2} \text{year}^{-1}$)
* The annual mean ratio of ambient to leaf CO2 partial during the growing season
  ($\chi$, Pa)
* The annual total potential GPP expressed in moles of Carbon ($A_0$, $\text{mol C
  m}^{-2} \text{year}^{-1}$).
* The length of the growing season in days for each year ($G$, days)

The first three variables ($c_a, D, P$) would typically be estimated from climate data
but the last two ($\chi, A_0$) are predicted values from the P Model. In addition, there
are four constants:

* $z$ estimates the growth and maintenance costs of leaves,
* $k$ is the light extinction coefficient of leaves, and
* $f_0$ is the ratio of annual total transpiration to annual total precipitation. This
  value is calculated using an empirical parameterisation from the aridity index for a
  location (see
  {meth}`PhenologyConst.calculate_f0<pyrealm.constants.phenology_const.PhenologyConst.calculate_f0>`).
* $\sigma$ is an empirically-derived penalty factor that captures loss of potential leaf
  area from delays deploying and dropping the canopy during the growing season.

The default values for these constants are set in the
{class}`~pyrealm.constants.phenology_const.PhenologyConst` constants object:

```{code-cell} ipython3
phenology_constants = PhenologyConst()
phenology_constants
```

## Example calculation

The example below uses a time series of 11 years of annual values from the [`DE_Gri`
Fluxnet site](https://fluxnet.org/doi/FLUXNET2015/DE-Gri). The code below loads some
site constants and then a data frame of annual values for the required variables,
including $A_0$ and $\chi$ from a P Model.

```{code-cell} ipython3
# Load site data
site_data_path = (
    resources.files("pyrealm_build_data.phenology") / "DE-GRI_site_data.json"
)
with open(site_data_path) as json_src:
    site_data = json.load(json_src)

# Load annual estimates
annual_data_path = (
    resources.files("pyrealm_build_data.phenology.fortnightly_example")
    / "annual_outputs.csv"
)
annual_data = pd.read_csv(annual_data_path).iloc[:, 0:9]
annual_data["time"] = annual_data["time"].to_numpy().astype("datetime64[Y]")
```

The `site_data` provides the following constants, including aridity index estimates (AI).

```{code-cell} ipython3
site_data
```

The plots below show the time series for the annual variables:

```{code-cell} ipython3
:tags: [hide-input]

fig, axes = plt.subplots(ncols=2, nrows=3, sharex=True, figsize=(10, 8))
axis_fmt_year = mdates.DateFormatter("%Y")

plot_vars = (
    ("annual_precip_molar", r"Total annual precipitation (moles)"),
    ("N_growing_days", r"Number of growing days"),
    ("annual_mean_ca_in_GS", r"Mean $c_a$ in growing season (Pa)"),
    ("annual_mean_chi_in_GS", r"Mean $\chi$ in growing season"),
    ("annual_mean_VPD_in_GS", r"Mean VPD in growing season"),
    ("ann_total_A0", "Total annual potential GPP (moles)"),
)

for (input_var, axis_label), axis in zip(plot_vars, axes.flatten()):

    axis.plot(annual_data["time"], annual_data[input_var])
    axis.set_ylabel(axis_label)
    axis.set_title(input_var)
    axis.xaxis.set_major_formatter(axis_fmt_year)

plt.tight_layout()
```

The code below then shows the use of the
{class}`~pyrealm.phenology.fapar_limitation.FaparLimitation` class to calculate
$f_{APAR_{max}}, L_{max}, m$ and prints a summary of the calculated values.

```{code-cell} ipython3
faparlim = FaparLimitation(
    annual_total_potential_gpp=annual_data["ann_total_A0"].to_numpy(),
    annual_mean_ca=annual_data["annual_mean_ca_in_GS"].to_numpy(),
    annual_mean_chi=annual_data["annual_mean_chi_in_GS"].to_numpy(),
    annual_mean_vpd=annual_data["annual_mean_VPD_in_GS"].to_numpy(),
    annual_total_precip=annual_data["annual_precip_molar"].to_numpy(),
    annual_growing_season_length=annual_data["N_growing_days"].to_numpy(),
    years=annual_data["time"].to_numpy().astype("datetime64[Y]"),
    aridity_index=site_data["AI_from_cruts"],
)

faparlim.summarize()
```

The resulting time series of annual values of $f_{APAR_{max}}, L_{max}, m$ are shown below:

```{code-cell} ipython3
:tags: [hide-input]

fig, axes = plt.subplots(ncols=1, nrows=3, sharex=True, figsize=(5, 8))

plot_vars = (
    (faparlim.fapar_max, r"$f_{APAR_{max}}$", "Maximum annual fAPAR"),
    (faparlim.lai_max, r"$L_{max}$", "Maximum annual LAI"),
    (faparlim.lai_to_gpp_ratio_m, r"$m$", "LAI to GPP ratio"),
)

for (input_var, axis_label, title), axis in zip(plot_vars, axes.flatten()):

    axis.plot(faparlim.years, input_var)
    axis.set_ylabel(axis_label)
    axis.set_title(title)
    axis.xaxis.set_major_formatter(axis_fmt_year)

plt.tight_layout()
```

## Fapar limitation from a PModel

Calculating maximum $f_{APAR}$ requires predictions of $A_0$ and $\chi$ from a P Model.
Since fitting a P Model _also_ requires estimates of VPD and CO2 concentration, much of
the data required to calculate maximum $f_{APAR}$ is stored within a fitted P Model. The
{meth}`FaparLimitation.from_pmodel<pyrealm.phenology.fapar_limitation.FaparLimitation.from_pmodel>`
method is provided to calculate maximum $f_{APAR}$ directly from an existing P
Model.

P models are typically fitted to observations at faster temporal scales: typically
monthly to weekly observations for the [standard P
Model](../pmodel/pmodel_details/pmodel_overview.md)
and subdaily observations for the [subdaily P
Model](../pmodel/subdaily_details/subdaily_overview.md). The
{meth}`~pyrealm.phenology.fapar_limitation.FaparLimitation.from_pmodel` method
automates the calculation of the required annual summary statistics using the dates and
times of the observations used in the P Model (see the
{class}`~pyrealm.core.time_series.AnnualValueCalculator` class for details).

The example here uses fortnightly summary data for the `DE_Gri` dataset to fit a P
Model. The data provides 287 observations of fortnightly average conditions for the site
over 11 years.

```{code-cell} ipython3
# Load fortnightly data
fn_data_path = (
    resources.files("pyrealm_build_data.phenology.fortnightly_example")
    / "fortnightly_data.csv"
)
fn_data = pd.read_csv(fn_data_path)

fn_data["time"] = pd.to_datetime(fn_data["time"])
fn_data.info()
```

The plot below shows the time series for temperature.

```{code-cell} ipython3
:tags: [hide-input]

plt.plot(fn_data["time"], fn_data["tc_mean"])
plt.axhline(0, linewidth=0.4, color="red")
_ = plt.ylabel("Temperature (°C)")
```

That data can then be used to fit a P model for the site, setting $f_{APAR} = 1$ to
calculate potential GPP.

```{code-cell} ipython3
pmodel_env = PModelEnvironment(
    tc=fn_data["tc_mean"].to_numpy(),
    vpd=fn_data["vpd_mean"].to_numpy(),
    patm=fn_data["patm_mean"].to_numpy(),
    co2=fn_data["co2_mean"].to_numpy(),
    ppfd=fn_data["ppfd_mean"].to_numpy(),
    fapar=np.array(1),
)
pmodel = PModel(pmodel_env)

pmodel.summarize()
```

The plot below shows the resulting predictions of mean GPP in µg C m-2 s-1 for each
fortnight.

```{code-cell} ipython3
:tags: [hide-input]

plt.plot(fn_data["time"], pmodel.gpp)
_ = plt.ylabel("Potential GPP (µg C m-2 s-1)")
```

The P model contains most of the information needed to estimate maximum $f_{APAR}$ for
each year, but the method requires four additional arguments:

`growing_season`
: The calculation of $f_{APAR_{max}}$ requires estimates of $D, c_a$ and $\chi$ **during
  the growing season**. The `growing_season` input is an array of boolean (`TRUE` or
  `FALSE`) values that indicates if each observation was during the growing season.
  There are different approaches for estimating the start and end of the growing season
  and so you need to create this variable according to the approach you want to use -
  it is often simply if the temperature exceeds a certain threshold.

`precip`
: The P Model does not require precipitation data, so this must be added when using
  `from_pmodel`. You will need to compile data for the total precipitation during each
  observation, expressed as moles of water per m2.

`aridity_index`
: You need to provide a climatological aridity index estimates for sites. This could be
  a single value or an array of site specific values.

`datetimes`
: This is only needed if you are using a standard PModel, as the SubdailyPModel already
  includes observation datetimes. These values are used to map the observations onto
   years and to scale per second rates from the P Model up to the annual time scale.

The code below then calculates $f_{APAR_{max}}$ for the observations.

```{code-cell} ipython3
faparlim_pmodel = FaparLimitation.from_pmodel(
    pmodel=pmodel,
    growing_season=fn_data["growing_season"].to_numpy(),
    precip=fn_data["precip_molar_sum"].to_numpy(),
    aridity_index=site_data["AI"],
    datetimes=fn_data["time"].to_numpy(),
)
```

The resulting time series of annual values of $f_{APAR_{max}}, L_{max}, m$ are shown
below and are basically identical to the calculations from the pre-calculated annual
values shown above

```{code-cell} ipython3
:tags: [hide-input]

fig, axes = plt.subplots(ncols=1, nrows=3, sharex=True, figsize=(5, 8))

plot_vars = (
    (faparlim_pmodel.fapar_max, r"$f_{APAR_{max}}$", "Maximum annual fAPAR"),
    (faparlim_pmodel.lai_max, r"$L_{max}$", "Maximum annual LAI"),
    (faparlim_pmodel.lai_to_gpp_ratio_m, r"$m$", "LAI to GPP ratio"),
)

for (input_var, axis_label, title), axis in zip(plot_vars, axes.flatten()):

    axis.plot(faparlim_pmodel.years, input_var)
    axis.set_ylabel(axis_label)
    axis.set_title(title)
    axis.xaxis.set_major_formatter(axis_fmt_year)

plt.tight_layout()
```

```{code-cell} ipython3

```
