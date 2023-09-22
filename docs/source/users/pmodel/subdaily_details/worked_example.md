---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: pyrealm_python3
  language: python
  name: pyrealm_python3
---

# Worked example

```{code-cell}
from importlib import resources

import xarray
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import matplotlib.dates as mdates

from pyrealm.pmodel import PModel, FastSlowPModel, PModelEnvironment, FastSlowScaler
from pyrealm.hygro import convert_sh_to_vpd
```

This notebook shows an example analysis using the P Model including fast and slow
photosynthetic responses. The dataset is taken from WFDE5 v2 and provides 1 hourly
resolution data on a 0.5° spatial grid. The fAPAR data is interpolated to the same
spatial and temporal resolution from MODIS data.

* time: 2018-06-01 00:00 to 2018-07-31 23:00 at 1 hour resolution.
* longitude: 10°W to 4°E at 0.5° resolution.
* latitude: 49°N to 60°N at 0.5° resolution

```{code-cell}
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

The plot below shows the spatial extent of the dataset using noon temperature for a
focal day. It also shows the locations of three sites used below to show time series of
predictions.

```{code-cell}
focal_datetime = np.where(datetimes == np.datetime64("2018-06-12 12:00:00"))[0]

# Plot the temperature data for an example timepoint and show the sites
focal_temp = ds["Tair"][focal_datetime] - 273.15
focal_temp.plot()
plt.plot(sites["lon"], sites["lat"], "xr")
```

The WFDE data need some conversion for use in the PModel, along with the definition of
the atmospheric CO2 concentration.

```{code-cell}
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

```{code-cell}
# Generate and check the PModelEnvironment
pm_env = PModelEnvironment(tc=tc, patm=patm, vpd=vpd, co2=co2)
pm_env.summarize()
```

## Instantaneous P Model

The standard implementation of the P Model used below assumes that plants can
instantaneously adopt optimal behaviour.

```{code-cell}
# Standard PModel
pmod = PModel(env=pm_env, kphio=1 / 8)
pmod.estimate_productivity(fapar=fapar, ppfd=ppfd)
pmod.summarize()
```

## P Model with slow responses

The code below then refits the model, with slow responses in $\xi$, $V_{cmax25}$ and
$J_{max25}$.

```{code-cell}
# FastSlowPModel with 1 hour noon acclimation window
fsscaler = FastSlowScaler(datetimes)
fsscaler.set_window(
    window_center=np.timedelta64(12, "h"),
    half_width=np.timedelta64(1, "h"),
)
fs_pmod = FastSlowPModel(
    env=pm_env,
    fs_scaler=fsscaler,
    handle_nan=True,
    fapar=fapar,
    ppfd=ppfd,
    alpha=1 / 15,
)
```

## Spatial predictions

The plots below show the spatial variation in the predicted GPP from the two models for
the focal datetime, along with a scatterplot comparing the two predictions.

```{code-cell}
# Extract the spatial grid for the focal datetime
pmod_gpp_focal = pmod.gpp[focal_datetime].squeeze()
fs_pmod_gpp_focal = fs_pmod.gpp[focal_datetime].squeeze()

# Set up subfigures
fig = plt.figure(layout="constrained", figsize=(10, 3))
(subfig1, subfig2) = fig.subfigures(1, 2, width_ratios=[0.8, 2])

# Plot the GPP predictions of the two models
ax = subfig1.subplots(1, 1)
ax.scatter(pmod_gpp_focal, fs_pmod_gpp_focal)
ax.plot([100, 450], [100, 450], "r-", linewidth=0.5)
ax.set_xlabel("Instantaneous GPP")
ax.set_ylabel("FastSlow GPP")
ax.set_title(" ")

(ax1, ax2) = subfig2.subplots(1, 2, sharey=True, sharex=True)

# Get a shared colour scale using the maximum across the two approaches
cmap = cm.get_cmap("viridis")
normalizer = Normalize(0, np.nanmax([pmod_gpp_focal, fs_pmod_gpp_focal]))
im = cm.ScalarMappable(norm=normalizer)

# Plot the spatial grids
ax1.imshow(pmod_gpp_focal, aspect=1, origin="lower", cmap=cmap, norm=normalizer)
ax1.set_title("Instantaneous")
ax2.imshow(fs_pmod_gpp_focal, aspect=1, origin="lower", cmap=cmap, norm=normalizer)
ax2.set_title("Fast Slow")

# Add a colour bar
subfig2.colorbar(
    im, ax=[ax1, ax2], shrink=0.55, label=r"GPP ($\mu g C\,m^{-2}\,s^{-1}$)"
)
```

## Time series predictions

The code below then extracts the time series for the two months from the three sites
shown above and plots the instantaneous predictions against predictions including slow
photosynthetic responses.

```{code-cell}
# Store the predictions in the xarray Dataset to use indexing
ds["GPP_pmod"] = (ds["Tair"].dims, pmod.gpp)
ds["GPP_fs_pmod"] = (ds["Tair"].dims, fs_pmod.gpp)

# Get three sites to show time series for locations
site_ds = ds.sel(sites, method="nearest")

# Set up subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharey=True)

# Plot the time series for the two approaches for each site
for ax, st in zip(axes, sites["stid"].values):

    ax.plot(
        datetimes, site_ds["GPP_pmod"].sel(stid=st), label="Instantaneous", color="0.4"
    )
    ax.plot(
        datetimes,
        site_ds["GPP_fs_pmod"].sel(stid=st),
        label="Fast Slow",
        color="red",
        alpha=0.7,
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.text(0.02, 0.90, st, transform=ax.transAxes)

axes[0].legend(loc="lower center", bbox_to_anchor=[0.5, 1], ncols=2, frameon=False)
plt.tight_layout()
```
