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

# Worked example of the Subdaily P Model

```{code-cell} ipython3
from importlib import resources

import xarray
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import matplotlib.dates as mdates

from pyrealm.pmodel.pmodel import PModel, SubdailyPModel
from pyrealm.pmodel.pmodel_environment import PModelEnvironment
from pyrealm.pmodel.acclimation import AcclimationModel

from pyrealm.core.hygro import convert_sh_to_vpd
```

This notebook shows example analyses fitting P Models that include fast and slow
photosynthetic responses to subdaily variation in environmental conditions. The dataset
is taken from WFDE5 v2 and provides 1 hourly resolution data on a 0.5° spatial grid. The
fAPAR data is interpolated to the same spatial and temporal resolution from MODIS data.

* time: 2018-06-01 00:00 to 2018-07-31 23:00 at 1 hour resolution.
* longitude: 10°W to 4°E at 0.5° resolution.
* latitude: 49°N to 60°N at 0.5° resolution

This notebook demonstrates fitting subdaily P Models in the `pyrealm` package. Model
fitting basically takes all of the same arguments as the standard
{class}`~pyrealm.pmodel.pmodel.PModel` class. There are three additional things
to set:

* The timing of the observations and the daily window that should be used to estimate
  [acclimation of slow responses](acclimation.md#the-acclimation-model).
* How rapidly plants [acclimate to daily optimal
  conditions](./acclimation.md#estimating-realised-responses).
* Approaches to handling any [missing data](./subdaily_model_and_missing_data.md): since
  estimating acclimation involves a recursive function, the subdaily model can be
  derailed by missing or undefined data.

The test data use some UK WFDE data for three sites in order to compare predictions
over a time series.

```{code-cell} ipython3
# Loading the example dataset:
dpath = (
    resources.files("pyrealm_build_data.uk_data") / "UK_WFDE5_FAPAR_2018_JuneJuly.nc"
)
ds = xarray.load_dataset(dpath)

datetimes = ds["time"].to_numpy()

# Define three sites for showing time series
sites = xarray.Dataset(
    data_vars=dict(
        lat=(["stid"], [50.73, 53.93, 58.03]), lon=(["stid"], [-3.52, -0.79, -4.41])
    ),
    coords=dict(stid=(["Exeter", "Pocklington", "Lairg"])),
)
```

The WFDE data need some conversion for use in the PModel, along with the definition of
the atmospheric CO2 concentration.

```{code-cell} ipython3
# Variable set up
# Air temperature in °C from Tair in Kelvin
tc = (ds["Tair"] - 273.15).to_numpy()
# Atmospheric pressure in Pascals
patm = ds["PSurf"].to_numpy()
# Convert specific huidity to VPD and remove negative values
vpd = convert_sh_to_vpd(sh=ds["Qair"].to_numpy(), ta=tc, patm=patm / 1000) * 1000
vpd = np.clip(vpd, 0, np.inf)
# Extract fAPAR (unitless)
fapar = ds["fAPAR"].to_numpy()
# Convert SW downwelling radiation from W/m^2 to PPFD µmole/m2/s1
ppfd = ds["SWdown"].to_numpy() * 2.04
# Define atmospheric CO2 concentration (ppm)
co2 = np.ones_like(tc) * 400
```

The code below then calculates the photosynthetic environment.

```{code-cell} ipython3
# Generate and check the PModelEnvironment
pm_env = PModelEnvironment(tc=tc, patm=patm, vpd=vpd, co2=co2, fapar=fapar, ppfd=ppfd)
pm_env.summarize()
```

## Instantaneous C3 and C4 P Models

The standard implementation of the P Model used below assumes that plants can
instantaneously adopt optimal behaviour.

```{code-cell} ipython3
# Standard PModels
pmodC3 = PModel(
    env=pm_env,
    method_kphio="fixed",
    method_optchi="prentice14",
)
pmodC3.summarize()
```

```{code-cell} ipython3
pmodC4 = PModel(
    env=pm_env,
    method_kphio="fixed",
    method_optchi="c4_no_gamma",
)

pmodC4.summarize()
```

```{code-cell} ipython3
np.nanmean(pmodC3.iwue)
```

```{code-cell} ipython3
np.nanmean((pmodC3.gpp / pmodC3.env.core_const.k_c_molmass) / pmodC3.gs)
```

## Subdaily P Models

The code below then refits these models, with slow responses in $\xi$, $V_{cmax25}$ and
$J_{max25}$. The `allow_holdover` argument allows the estimation of realised optimal
values to holdover previous realised values to cover missing data within the
calculations: essentially the plant does not acclimate until the optimal values can be
calculated again to update those realised estimates.

```{code-cell} ipython3
# Set the acclimation window to an hour either side of noon
acclim_model = AcclimationModel(datetimes, alpha=1 / 15, allow_holdover=True)
acclim_model.set_window(
    window_center=np.timedelta64(12, "h"),
    half_width=np.timedelta64(1, "h"),
)

# Fit C3 and C4 with the Subdaily P Model
subdailyC3 = SubdailyPModel(
    env=pm_env,
    acclim_model=acclim_model,
    method_optchi="prentice14",
    method_kphio="fixed",
)
subdailyC4 = SubdailyPModel(
    env=pm_env,
    acclim_model=acclim_model,
    method_optchi="c4_no_gamma",
    method_kphio="fixed",
)
```

## Time series predictions

The code below then extracts the time series for the two months from the three sites
shown above and plots the instantaneous predictions against predictions including slow
photosynthetic responses.

```{code-cell} ipython3
# Store the predictions in the xarray Dataset to use indexing
ds["GPP_pmodC3"] = (ds["Tair"].dims, pmodC3.gpp)
ds["GPP_subdailyC3"] = (ds["Tair"].dims, subdailyC3.gpp)
ds["GPP_pmodC4"] = (ds["Tair"].dims, pmodC4.gpp)
ds["GPP_subdailyC4"] = (ds["Tair"].dims, subdailyC4.gpp)

# Get three sites to show time series for locations
site_ds = ds.sel(sites, method="nearest")

# Set up subplots
fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharey=True)

# Plot the time series for the two approaches for each site
for (ax1, ax2), st in zip(axes, sites["stid"].values):

    ax1.plot(
        datetimes,
        site_ds["GPP_pmodC3"].sel(stid=st),
        label="Instantaneous",
        color="0.4",
    )
    ax1.plot(
        datetimes,
        site_ds["GPP_subdailyC3"].sel(stid=st),
        label="Subdaily",
        color="red",
        alpha=0.7,
    )
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax1.text(0.02, 0.90, st + " - C3", transform=ax1.transAxes)

    ax2.plot(
        datetimes,
        site_ds["GPP_pmodC4"].sel(stid=st),
        label="Instantaneous",
        color="0.4",
    )
    ax2.plot(
        datetimes,
        site_ds["GPP_subdailyC4"].sel(stid=st),
        label="Subdaily",
        color="red",
        alpha=0.7,
    )
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax2.text(0.02, 0.90, st + " - C4", transform=ax2.transAxes)

axes[0][0].legend(loc="lower center", bbox_to_anchor=[0.5, 1], ncols=2, frameon=False)
plt.tight_layout()
```

## Converting models

The subdaily models can also be obtained directly from the standard models, using the
{meth}`PModel.to_subdaily<pyrealm.pmodel.pmodel.PModel.to_subdaily>` method:

```{code-cell} ipython3
# Convert standard C3 model
converted_C3 = pmodC3.to_subdaily(acclim_model=acclim_model)

# Convert standard C4 model
converted_C4 = pmodC4.to_subdaily(acclim_model=acclim_model)
```

This produces the same outputs as the `SubdailyPModel` class, but is convenient and more
compact when the two models are going to be compared.

```{code-cell} ipython3
:tags: [remove-cell]

# This cell is here to force a docs build failure if these values
# are _not_ identical. The 'remove-cell' tag is applied to hide this
# in built docs.
from numpy.testing import assert_allclose

assert_allclose(subdailyC3.gpp, converted_C3.gpp)
assert_allclose(subdailyC4.gpp, converted_C4.gpp)
```
