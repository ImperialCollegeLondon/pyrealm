---
jupytext:
  formats: md:myst
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

# Draft implementation of general Subdaily model

```{code-cell} ipython3
from importlib import resources

import xarray
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import matplotlib.dates as mdates

from pyrealm.pmodel import PModel, FastSlowPModel, PModelEnvironment, FastSlowScaler
from pyrealm.pmodel.new_subdaily import SubdailyPModel, convert_pmodel_to_subdaily

from pyrealm.core.hygro import convert_sh_to_vpd
```

This notebook demonstrates a new implementation of the subdaily P Model.
The current implementation (`pmodel.FastSlowPModel`) is a conversion of
Giulia's original R code used for the JULES paper, although it extends that
code to add slow acclimation of the `xi` parameter. The equations in that
implementation were hard-coded to the {cite:t}`Prentice:2014bc` equations for the
C3 pathway.

This implementation takes all of the same arguments as the standard
{class}`~pyrealm.pmodel.pmodel.PModel` class, used to fit a model that exhibits
instantaneously optimal behaviour. This allows users to select between the different
optimal chi estimation options, including those representing C4 pathways and water
stress.

This uses the new optimal chi calculations, which provide the option to assert the `xi`
values to be used. In the normal P Model, the expected `xi` is used, but now we can also
take the same forcing variables and recalculate with lagged `xi`.

So the new implementation takes a PModel environment, calculates the daily optimal
values during the acclimation window (as before), applies a lag (as before), but can now
use the {class}`~pyrealm.pmodel.optimal_chi.OptimalChiABC` methods to estimate `chi`,
`ci`, `mc` and `mj` using lagged `xi`. The lagged `vcmax` and `jmax` are calculated
using the `jmax` limitation scheme specified by the user and then `Ac` and `Aj` are
calculated as usual.

I think this is also a better user interface. The subdaily interface is very similar to
the existing P Model implementation and uses all the same settings, so a user just
needs to set the acclimation window and how strong a lag to apply (`alpha`). The new
code also provides a simple wrapper to automatically convert an existing standard
`PModel` instance to a `SubdailyPModel`.

Note that this _can_ also be used with the experimental optimal chi methods:
Aliénor's soil moisture approaches and Rodolfo's rootzone stress. This makes
intuitive sense - the plant acclimates to water stress too - but I don't know
if this is the right way to apply this!

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
pm_env = PModelEnvironment(tc=tc, patm=patm, vpd=vpd, co2=co2)
pm_env.summarize()
```

## Instantaneous C3 and C4 P Models

The standard implementation of the P Model used below assumes that plants can
instantaneously adopt optimal behaviour.

```{code-cell} ipython3
# Standard PModels
pmodC3 = PModel(env=pm_env, kphio=1 / 8, method_optchi="prentice14")
pmodC3.estimate_productivity(fapar=fapar, ppfd=ppfd)
pmodC3.summarize()
```

```{code-cell} ipython3
pmodC4 = PModel(env=pm_env, kphio=1 / 8, method_optchi="c4_no_gamma")
pmodC4.estimate_productivity(fapar=fapar, ppfd=ppfd)
pmodC4.summarize()
```

## Subdaily P Model

The code below then refits these models, with slow responses in $\xi$, $V_{cmax25}$ and
$J_{max25}$.

```{code-cell} ipython3
# Set the acclimation window to an hour either side of noon
fsscaler = FastSlowScaler(datetimes)
fsscaler.set_window(
    window_center=np.timedelta64(12, "h"),
    half_width=np.timedelta64(1, "h"),
)

# Fit C3 and C4 with the new implementation
subdailyC3 = SubdailyPModel(
    env=pm_env, 
    kphio=1 / 8, 
    method_optchi="prentice14",
    fapar=fapar,
    ppfd=ppfd,
    fs_scaler=fsscaler, 
    alpha=1 / 15, 
    handle_nan=True,
)
subdailyC4 = SubdailyPModel(
    env=pm_env, 
    kphio=1 / 8, 
    method_optchi="c4_no_gamma",
    fapar=fapar,
    ppfd=ppfd,
    fs_scaler=fsscaler, 
    alpha=1 / 15, 
    handle_nan=True,
)

# Fit C3 using the original implementation
fs_pmod = FastSlowPModel(
    env=pm_env,
    fs_scaler=fsscaler,
    handle_nan=True,
    fapar=fapar,
    ppfd=ppfd,
    alpha=1 / 15,
)
```

Reassuringly, the two subdaily C3 models predict equal GPP, except for trivial
precision differences.

```{code-cell} ipython3
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4))

ax1.scatter(fs_pmod.gpp.flatten(), subdailyC3.gpp.flatten())
ax1.set_xlabel("GPP from original implementation")
ax1.set_ylabel("GPP from new implementation")

ax2.hist((fs_pmod.gpp.flatten() - subdailyC3.gpp.flatten()) / subdailyC3.gpp.flatten())
ax2.set_xlabel("Percentage difference in GPP");
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
`convert_pmodel_to_subdaily` method:

```{code-cell} ipython3
# Convert standard C3 model
converted_C3 = convert_pmodel_to_subdaily(
    pmodel=pmodC3,
    fs_scaler=fsscaler, 
    alpha=1 / 15, 
    handle_nan=True,
)

# Convert standard C4 model
converted_C4 = convert_pmodel_to_subdaily(
    pmodel=pmodC4,
    fs_scaler=fsscaler, 
    alpha=1 / 15, 
    handle_nan=True,
)
```

This produces the same outputs as the `SubdailyPModel` class, but is convenient and more
compact when the two models are going to be compared.

```{code-cell} ipython3
# Models have identical GPP - maximum absolute difference is zero.
print(np.nanmax(abs(subdailyC3.gpp.flatten() - converted_C3.gpp.flatten())))
print(np.nanmax(abs(subdailyC4.gpp.flatten() - converted_C4.gpp.flatten())))
```

```{code-cell} ipython3
# Save the new predictions alongside the input data.
ds.to_netcdf('new_subdaily_C3_and_C4_GPP_values.nc')
```
