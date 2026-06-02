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
  name: python
  version: 3.12.3
  mimetype: text/x-python
  codemirror_mode:
    name: ipython
    version: 3
  pygments_lexer: ipython3
  nbconvert_exporter: python
  file_extension: .py
---

# Estimating maximum annual $f_{APAR}$

```{admonition} Run this notebook
:class: hint

* Read the guide on setting up your computer to [run Jupyter
  notebooks](../getting_started.md)
* Download {nb-download}`this notebook<./fapar_limitation.ipynb>` as a Jupyter notebook.

```

```{code-cell} ipython3
:tags: [hide-input]

from importlib import resources

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pyrealm.core.datasets import get_pyrealm_data
from pyrealm.constants import PhenologyConst
from pyrealm.phenology.fapar_limitation_new import FaparLimitationNew
from pyrealm.pmodel import PModelEnvironment, PModel, SubdailyPModel, AcclimationModel
```

The {class}`~pyrealm.phenology.fapar_limitation_new.FaparLimitationNew` class calculates
annual maximum values for both $f_{APAR}$ and leaf area index ($L$). Two alternative
approaches are available via the `method` argument:

1. the option `method=cai` implements the approach of {cite:t}`cai:2025a`, and
2. the option `method=zhu` implements the approach of {cite:t}`zhu:2026a`.

In both cases, the maximum annual fAPAR is limited by the ability of plants to
assimilate carbon for constructing leaves. This can be limited either by the
availability of light energy ($f_{APAR_{c}}$) or by the availability of water
($f_{APAR_{w}}$). The equations are:

$$
\begin{align*}
f_{APAR_{c}} &= 1 - \frac{z}{k A_0}\\
f_{APAR_{w}} &= \left(\frac{ c_a \left( 1 - \chi \right)}{ 1.6 D }\right)
        \left(\frac{ f_0 P }{ A_0 }\right) \\
\end{align*}
$$

Five of the terms above are annual estimates for a site of:

* The ambient CO2 partial pressure during the growing season ($c_a$, Pa).
* The annual mean vapour pressure deficit during the growing season ($D$, Pa)
* The annual total precipitation ($P$, $\text{mol m}^{-2} \text{year}^{-1}$)
* The annual mean ratio of ambient to leaf CO2 partial during the growing season
  ($\chi$, Pa)
* The annual total potential GPP expressed in moles of Carbon ($A_0$, $\text{mol C
  m}^{-2} \text{year}^{-1}$).

The remaining variables are:

* the light extinction coefficient of leaves ($k$),
* the growth and maintenance costs of leaves ($z$), and
* the ratio of annual total transpiration to annual total precipitation ($f_0$).

The methods differ in the values used for $z$ and $f_0$ and in the calculation of
annual maximum fAPAR from the water limited and energy limited terms.

* The option `method=cai` uses a fixed value for $z$ but calculates $f_0$ as a function
  of the site-specific long term aridity index expressed as PET/P (see
  {class}`FaparLimitationMethodCai.set_z_and_f0<pyrealm.phenology.fapar_limitation_new.FaparLimitationMethodCai.set_z_and_f0>`
  for details). The maximum annual fAPAR for a site is then simply the minimum of the
  energy and water limited terms for that site.

  Using this method requires that users provide those site specific aridity values in
  addition to the other variables required for calculation.

* The option `method=zhu` uses fixed values for both $z$ and $f_0$, but the maximum
  annual fAPAR is a function of the energy limited and water limited terms, following
  the model of the Budyko curve {cite:p}`roderick:2011a` (see
  {class}`FaparLimitationMethodZhu.calculate_maximum_fapar<pyrealm.phenology.fapar_limitation_new.FaparLimitationMethodZhu.calculate_maximum_fapar>`
  for details).

In both cases, best fit global values of $z$ and $f_0$ (or parameters of the function
for $f_0$) were estimated from satellite derived fAPAR and environmental data. These
values are defined in the
{class}`~pyrealm.constants.phenology_const.PhenologyConst` constants object, which can
be updated with user defined values.

From the maximum annual fAPAR, the maximum leaf area index ($L_{max}$) is
approximated using Beer's law:

$$
        L_{max} = - ( 1 / k ) \ln {1 -f_{APAR_{max}}}
$$

## Example calculations

The examples below uses a time series of half hourly data from the  [`DE_Gri`
Fluxnet site](https://fluxnet.org/doi/FLUXNET2015/DE-Gri) from 2004 to 2014 that
has been resampled to show different workflows for calculating annual maximum fAPAR.
The different workflows use some site specific scalar values, loaded in the following
code.

```{code-cell} ipython3
# Load site data
site_data_path = get_pyrealm_data("phenology/inputs/source/DE-GRI_site_data.json")

with open(site_data_path) as json_src:
    site_data = json.load(json_src)
```

### Direct calculation from annual values

Directly calculating annual maximum fAPAR using
{class}`~pyrealm.phenology.fapar_limitation_new.FaparLimitationNew` requires you to
provide the calculated annual summary statistics given in the equations above. The
example data contains estimates of these annual values calculated from the original half
hourly FluxNET data and from a separately fitted model of assimilation.

```{code-cell} ipython3
# Load annual estimates
annual_data_path = get_pyrealm_data("phenology/inputs/fortnightly/annual_inputs.csv")
annual_data = pd.read_csv(annual_data_path)
annual_data["time"] = annual_data["year"].to_numpy().astype(str).astype("datetime64[Y]")
```

The plots below show the annual estimates of the required variables from the
`annual_data` dataset:

```{code-cell} ipython3
:tags: [hide-input]

fig, axes = plt.subplots(ncols=2, nrows=3, sharex=True, figsize=(8, 8))
axis_fmt_year = mdates.DateFormatter("%Y")

plot_vars = (
    ("annual_precip_molar", r"Total annual precipitation (moles)"),
    ("N_growing_days", r"Number of growing days"),
    ("annual_mean_ca_in_GS", r"Mean $c_a$ in growing season (Pa)"),
    ("annual_mean_chi_in_GS", r"Mean $\chi$ in growing season"),
    ("annual_mean_VPD_in_GS", r"Mean VPD in growing season"),
    ("annual_total_A0", "Total annual potential GPP (moles)"),
)

for (input_var, axis_label), axis in zip(plot_vars, axes.flatten()):

    axis.plot(annual_data["time"], annual_data[input_var])
    axis.set_ylabel(axis_label)
    axis.set_title(input_var)
    axis.xaxis.set_major_formatter(axis_fmt_year)

plt.tight_layout()
```

The code below then shows the use of the different methods available within the
{class}`~pyrealm.phenology.fapar_limitation_new.FaparLimitationNew` class to calculate
$f_{APAR_{max}}$ and $L_{max}$. Note that `method="cai"` requires additional
`aridity_index` data and that these should be an array providing a single climatological
estimate per site. For example, a dataset covering a 5x5 grid of sites over 20 years
would have annual data with the shape `(20, 5, 5)` and so the aridity data would be an
array with shape `(5, 5)`. When there is only a single site, the site data should be a
"scalar" array of size `(1,)` as in the example below.

```{code-cell} ipython3
faparlim_cai = FaparLimitationNew(
    method="cai",
    annual_total_potential_gpp=annual_data["annual_total_A0"].to_numpy(),
    annual_mean_ca=annual_data["annual_mean_ca_in_GS"].to_numpy(),
    annual_mean_chi=annual_data["annual_mean_chi_in_GS"].to_numpy(),
    annual_mean_vpd=annual_data["annual_mean_VPD_in_GS"].to_numpy(),
    annual_total_precip=annual_data["annual_precip_molar"].to_numpy(),
    annual_growing_season_length=annual_data["N_growing_days"].to_numpy(),
    years=annual_data["time"].to_numpy().astype("datetime64[Y]"),
    aridity_index=np.array([site_data["AI_from_cruts"]]),
)

faparlim_cai.summarize()
```

```{code-cell} ipython3
faparlim_zhu = FaparLimitationNew(
    method="zhu",
    annual_total_potential_gpp=annual_data["annual_total_A0"].to_numpy(),
    annual_mean_ca=annual_data["annual_mean_ca_in_GS"].to_numpy(),
    annual_mean_chi=annual_data["annual_mean_chi_in_GS"].to_numpy(),
    annual_mean_vpd=annual_data["annual_mean_VPD_in_GS"].to_numpy(),
    annual_total_precip=annual_data["annual_precip_molar"].to_numpy(),
    annual_growing_season_length=annual_data["N_growing_days"].to_numpy(),
    years=annual_data["time"].to_numpy().astype("datetime64[Y]"),
)

faparlim_zhu.summarize()
```

The resulting time series of annual values of $f_{APAR_{max}}$ and $L_{max}$ are shown below:

```{code-cell} ipython3
:tags: [hide-input]

fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(8, 5))

plot_vars = (
    ("fapar_max", r"$f_{APAR_{max}}$", "Maximum annual fAPAR"),
    ("lai_max", r"$L_{max}$", "Maximum annual LAI"),
)
methods = (
    ("cai", faparlim_cai),
    ("zhu", faparlim_zhu),
)


for (input_var, axis_label, title), axis in zip(plot_vars, axes.flatten()):
    for method, faparlim in methods:
        axis.plot(
            faparlim.years, getattr(faparlim, input_var), label=f"method={method}"
        )
        axis.set_ylabel(axis_label)
        axis.set_title(title)
        axis.xaxis.set_major_formatter(axis_fmt_year)
    axis.legend(frameon=False)

plt.tight_layout()
```

### Calculation from a fitted PModel

Calculating annual maximum fAPAR _requires_ the use of a P Model to provide estimates of
$A_0$ and $\chi$ and fitting a P Model also requires data on VPD and CO2 concentration.
However, a P model is typically fitted to data at faster temporal scales than the annual
values required to calculate maximum fAPAR.

* The [standard P Model](../pmodel/pmodel_details/pmodel_overview.md) typically uses
  monthly to weekly observations
* The [subdaily P Model](../pmodel/subdaily_details/subdaily_overview.md) is fitted to
  observations at subdaily frequencies.

To make it easier to estimate maximum $f_{APAR}$ the
{meth}`FaparLimitationNew.from_pmodel<pyrealm.phenology.fapar_limitation_new.FaparLimitationNew.from_pmodel>`
method can be used to automatically calculate the required annual summary statistics
using the dates and times of the observations used in the P Model (see the
{class}`~pyrealm.core.time_series.AnnualValueCalculator` class for details). The method
also handles all of unit conversion needed to estimate annual maximum fAPAR.

The method does require some additional data that is not required when fitting a P model:

`growing_season`
: The calculation of $f_{APAR_{max}}$ requires estimates of $D, c_a$ and $\chi$ **during
  the growing season**. The `growing_season` input is an array of boolean (`True` or
  `False`) values that indicates if each observation was during the growing season.
  There are different approaches for estimating the start and end of the growing season
  and so you need to create this variable according to the approach you want to use -
  it is often simply if the temperature exceeds a certain threshold.

`precip`
: The P Model does not require precipitation data, so this must be added when
  using `from_pmodel`. You will need to compile data for the total precipitation during
  each observation, expressed as the total precipitation during each period of
  observation in moles of water per m2.

`datetimes`
: Datestamps are needed to map the data in the PModel onto years to map the observations
  onto years and to scale per second rates from the P Model up to the annual time
  scale. This is only needed when using the standard P Model (see below)

+++

#### The standard P Model

The example here uses fortnightly summary data for the `DE_Gri` dataset to fit a
standard P Model. The data provides 287 observations of fortnightly average conditions
for the site over 11 years.

```{code-cell} ipython3
# Load fortnightly data
fortnightly_data_path = get_pyrealm_data(
    "phenology/inputs/fortnightly/pmodel_inputs.csv"
)
fortnightly_data = pd.read_csv(fortnightly_data_path)
fortnightly_data["time"] = pd.to_datetime(fortnightly_data["time"])
```

The plot below shows the time series for temperature.

```{code-cell} ipython3
:tags: [hide-input]

fig, ax = plt.subplots()
ax.plot(fortnightly_data["time"], fortnightly_data["tc"])
ax.axhline(0, linewidth=0.4, color="red")
ax.xaxis.set_major_formatter(axis_fmt_year)
_ = ax.set_ylabel("Temperature (°C)")
```

That data can then be used to fit a P model for the site. Note that setting $f_{APAR} =
1$ is required to calculate potential GPP for estimating maximum fAPAR.

```{code-cell} ipython3
pmodel_env = PModelEnvironment(
    tc=fortnightly_data["tc"].to_numpy(),
    vpd=fortnightly_data["vpd"].to_numpy(),
    patm=fortnightly_data["patm"].to_numpy(),
    co2=fortnightly_data["co2"].to_numpy(),
    ppfd=fortnightly_data["ppfd"].to_numpy(),
    fapar=np.array([1]),
)
pmodel = PModel(pmodel_env)

pmodel.summarize()
```

The plot below shows the resulting predictions of mean GPP in µg C m-2 s-1 for each
fortnight.

```{code-cell} ipython3
:tags: [hide-input]

fig, ax = plt.subplots()
ax.plot(fortnightly_data["time"], pmodel.gpp)
ax.xaxis.set_major_formatter(axis_fmt_year)
_ = ax.set_ylabel("Potential GPP (µg C m-2 s-1)")
```

The code below then uses the `from_pmodel` method to calculate maximum annual fAPAR
using two different methods. Note that `method="cai"` also requires the additional
variable `aridity_index` providing estimates of site specific aridity (PET/P), which
should be an array of single climatological values per site, described above.

```{code-cell} ipython3
# Method = "cai", with additional aridity data
faparlim_pmodel_cai = FaparLimitationNew.from_pmodel(
    method="cai",
    pmodel=pmodel,
    growing_season=fortnightly_data["growing_season"].to_numpy(),
    precip=fortnightly_data["precip_molar"].to_numpy(),
    datetimes=fortnightly_data["time"].to_numpy(),
    aridity_index=np.array([site_data["AI"]]),
)

# Method = "zhu"
faparlim_pmodel_zhu = FaparLimitationNew.from_pmodel(
    method="zhu",
    pmodel=pmodel,
    growing_season=fortnightly_data["growing_season"].to_numpy(),
    precip=fortnightly_data["precip_molar"].to_numpy(),
    datetimes=fortnightly_data["time"].to_numpy(),
)
```

The resulting time series of annual values of $f_{APAR_{max}}, L_{max}$ are shown
below and are basically identical to the calculations from the pre-calculated annual
values shown above.

```{code-cell} ipython3
:tags: [hide-input]

fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(8, 4))

plot_vars = (
    ("fapar_max", r"$f_{APAR_{max}}$", "Maximum annual fAPAR"),
    ("lai_max", r"$L_{max}$", "Maximum annual LAI"),
)
methods = (
    ("cai", faparlim_pmodel_cai),
    ("zhu", faparlim_pmodel_zhu),
)


for (input_var, axis_label, title), axis in zip(plot_vars, axes.flatten()):
    for method, faparlim in methods:
        axis.plot(
            faparlim.years, getattr(faparlim, input_var), label=f"method={method}"
        )
        axis.set_ylabel(axis_label)
        axis.set_title(title)
        axis.xaxis.set_major_formatter(axis_fmt_year)
    axis.legend(frameon=False)

plt.tight_layout()
```

#### The SubdailyPModel

The code below loads the original half hourly FluxNET data (2004-2014, 192864 values).

```{code-cell} ipython3
# Load subdaily data
subdaily_data_path = get_pyrealm_data("phenology/inputs/subdaily/pmodel_inputs.csv")
subdaily_data = pd.read_csv(subdaily_data_path)
subdaily_data["time"] = pd.to_datetime(subdaily_data["time"])
```

This dataset is then used to build predictions using the subdaily P Model:

```{code-cell} ipython3
# Calculate the P Model environment
subdaily_pmodel_env = PModelEnvironment(
    tc=subdaily_data["tc"].to_numpy(),
    vpd=subdaily_data["vpd"].to_numpy(),
    patm=subdaily_data["patm"].to_numpy(),
    co2=subdaily_data["co2"].to_numpy(),
    ppfd=subdaily_data["ppfd"].to_numpy(),
    fapar=np.array([1]),
)

# Define the acclimation window for the model
acclim_model = AcclimationModel(datetimes=subdaily_data["time"].to_numpy())
acclim_model.set_window(
    window_center=np.timedelta64(12, "h"), half_width=np.timedelta64(1, "h")
)

# Fit the subdaily P Model
subdaily_pmodel = SubdailyPModel(
    env=subdaily_pmodel_env,
    acclim_model=acclim_model,
)
```

Predicting annual maximum fAPAR is then almost identical to using the standard P Model.
The only difference is that the acclimation model used to fit subdaily P Models already
provides datetimes for each observation, so the `datetimes` argument is not used:

```{code-cell} ipython3
# Method = "cai", with additional aridity data
faparlim_subdaily_pmodel_cai = FaparLimitationNew.from_pmodel(
    method="cai",
    pmodel=subdaily_pmodel,
    growing_season=subdaily_data["growing_season"].to_numpy(),
    precip=subdaily_data["precip_molar"].to_numpy(),
    aridity_index=np.array([site_data["AI"]]),
)

# Method = "zhu"
faparlim_subdaily_pmodel_zhu = FaparLimitationNew.from_pmodel(
    method="zhu",
    pmodel=subdaily_pmodel,
    growing_season=subdaily_data["growing_season"].to_numpy(),
    precip=subdaily_data["precip_molar"].to_numpy(),
)
```

The plots of the annual predicted maximum fAPAR again look very similar.

```{code-cell} ipython3
:tags: [hide-input]

fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(8, 4))

plot_vars = (
    ("fapar_max", r"$f_{APAR_{max}}$", "Maximum annual fAPAR"),
    ("lai_max", r"$L_{max}$", "Maximum annual LAI"),
)
methods = (
    ("cai", faparlim_subdaily_pmodel_cai),
    ("zhu", faparlim_subdaily_pmodel_zhu),
)


for (input_var, axis_label, title), axis in zip(plot_vars, axes.flatten()):
    for method, faparlim in methods:
        axis.plot(
            faparlim.years, getattr(faparlim, input_var), label=f"method={method}"
        )
        axis.set_ylabel(axis_label)
        axis.set_title(title)
        axis.xaxis.set_major_formatter(axis_fmt_year)
    axis.legend(frameon=False)

plt.tight_layout()
```
