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

```{code-cell} ipython3
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
constucting leaves and this can be limited either by the availability of light energy
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
site constants and then a data frame of annual values for the required variables.

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

```{code-cell} ipython3
site_data
```

```{code-cell} ipython3
annual_data
```

The plots below show the annual variation in these inputs:

```{code-cell} ipython3
fig, axes = plt.subplots(ncols=2, nrows=3, sharex=True, figsize=(10, 8))
axis_fmt_year = mdates.DateFormatter("%Y")

plot_vars = (
    ("annual_precip_molar", "TBD"),
    ("N_growing_days", "TBD"),
    ("annual_mean_ca_in_GS", "TBD"),
    ("annual_mean_chi_in_GS", "TBD"),
    ("annual_mean_VPD_in_GS", "TBD"),
    ("ann_total_A0", "TBD"),
)

for (input_var, axis_label), axis in zip(plot_vars, axes.flatten()):

    axis.plot(annual_data["time"], annual_data[input_var])
    axis.set_ylabel(axis_label)
    axis.set_title(input_var)
    axis.xaxis.set_major_formatter(axis_fmt_year)

plt.tight_layout()
```

The code below then uses the  {class}`~pyrealm.phenology.fapar_limitation.FaparLimitation`
class to calculate $f_{APAR_{max}}, L_{max}, m$.

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
```

```{code-cell} ipython3
faparlim.summarize()
```

The resulting time series of annual values of $f_{APAR_{max}}, L_{max}, m$ are shown below:

```{code-cell} ipython3
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
    ncols=2, nrows=2, sharex=True, figsize=(8, 6)
)
ax1.plot(faparlim.years, faparlim.fapar_max)
ax2.plot(faparlim.years, faparlim.lai_max)
ax3.plot(faparlim.years, faparlim.lai_to_gpp_ratio_m)
plt.tight_layout()
```

## Fapar limitation from a PModel

To make it easier to calculate $f_{APAR}$ limitation, the
{meth}`FaparLimitation.from_pmodel<pyrealm.phenology.fapar_limitation.FaparLimitation.from_pmodel>`
method calculates the values above directly from an existing P Model. The P Model
provides estimates of GPP for use in calculating $A_0$ and values of $\chi$ for
calculating mean annual $\chi$. In addition, fitting a P Model requires estimates of
vapor pressure deficit and ambient CO2 concentration, so  annual mean $D$ and $c_a$ can
also be calculated from a P Model.

Fitting a P Model does not require estimates of precipitation, so this needs to be
provided for each observation in the P Model.

The code below loads fortnightly summary data for the `DE_Gri` dataset:

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

The data provides 287 observations of fortnightly average conditions for the site over
11 years. The plot below shows the temperature values.

```{code-cell} ipython3
plt.plot(data["time"], data["tc_mean"])
_ = plt.axhline(0, linewidth=0.4, color="red")
```

That data can then be used to fit a P model for the site:

```{code-cell} ipython3
pmodel_env = PModelEnvironment(
    tc=data["tc_mean"].to_numpy(),
    vpd=data["vpd_mean"].to_numpy(),
    patm=data["patm_mean"].to_numpy(),
    co2=data["co2_mean"].to_numpy(),
    ppfd=data["ppfd_mean"].to_numpy(),
    fapar=np.array(1),
)
pmodel = PModel(pmodel_env)
```

```{code-cell} ipython3
pmodel.summarize()
```

The plot below shows the resulting predictions of mean GPP in µg C m2 s-1 for each
fortnight.

```{code-cell} ipython3
_ = plt.plot(data["time"], pmodel.gpp)
```

The P model contains most of the information needed to estimate maximum $f_{APAR}$ for
each year, but some additional data is needed.

1. The calculation of $f_{APAR_{max}}$ requires estimates of $D, c_a$ and $\chi$
   **during the growing season**. The function therefore needs an additional
   `growing_season` argument that indicates - for each observation - if that observation
   was during the growing season. This needs to be provide a boolean (or logical) value
   for each observation. There are different approaches for estimating the start and end
   of the growing season and so you need to create this variable according to the
   approach you want to use - it is often simply if the temperature exceeds a certain
   threshold.

2. The calculation also requires precipitation data: you will need to compile data for
   the total precipitation during each observation, expressed as moles of water per m2.

3. If you are using a standard PModel, rather than a SubdailyPModel, you will also need
   to provide datetimes for the observations. These values are used to map the
   observations onto years and to scale per second rates from the P Model up to the
   annual time scale. The SubdailyPModel requires datetimes for observations, so these
   are already defined for SubdailyPModel inputs.

4. Lastly, the aridity index is needed for sites.

The code below then calculates $f_{APAR_{max}}$ for the observations.

```{code-cell} ipython3
faparlim_pmodel = FaparLimitation.from_pmodel(
    pmodel=pmodel,
    growing_season=data["growing_season"].to_numpy(),
    precip=data["precip_molar_sum"].to_numpy(),
    aridity_index=site_data["AI"],
    datetimes=data["time"].to_numpy(),
)
```

The resulting time series of annual values of $f_{APAR_{max}}, L_{max}, m$ are shown
below and are basically identical to the calculations from the pre-calculated annual
values shown above

```{code-cell} ipython3
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
    ncols=2, nrows=2, sharex=True, figsize=(8, 6)
)
ax1.plot(faparlim_pmodel.years, faparlim_pmodel.fapar_max)
ax2.plot(faparlim_pmodel.years, faparlim_pmodel.lai_max)
ax3.plot(faparlim_pmodel.years, faparlim_pmodel.lai_to_gpp_ratio_m)
plt.tight_layout()
```
