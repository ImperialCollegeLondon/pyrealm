"""Run the original benchmark time series.

This script makes a test input data set to run through splash_run_time_series_parallel
from the original one dimensional test input data used in __main__ in main.py in the
SPLASH V1 code.

This data was originally provided in the SPLASH v1.0 implementation and is data from San
Francisco in 2000, with precipitation and temperature taken from WFDEI and sunshine
fraction interpolated from CRU TS.
"""

import numpy as np
import xarray

data = np.genfromtxt(
    "data/splash_sf_example_data.csv",
    dtype=None,
    delimiter=",",
    names=True,
    encoding="UTF-8",
)


# Set coordinatse
lat = np.array([37.7])
lon = np.array([122.4])
dates = np.arange(
    np.datetime64("2000-01-01"), np.datetime64("2001-01-01"), np.timedelta64(1, "D")
).astype("datetime64[ns]")

coords = {"time": dates, "lat": lat, "lon": lon}

# Convert data onto coordinate dimensions
pn = xarray.DataArray(data["pn"][:, np.newaxis, np.newaxis], coords=coords)
tair = xarray.DataArray(data["tair"][:, np.newaxis, np.newaxis], coords=coords)
sf = xarray.DataArray(data["sf"][:, np.newaxis, np.newaxis], coords=coords)
elev = xarray.DataArray(np.array([[142.0]]), coords={"lat": lat, "lon": lon})


# Assemble and save
splash_grid = xarray.Dataset(
    {
        "elev": elev,
        "tmp": tair,
        "pre": pn,
        "sf": sf,
    }
)

splash_grid.to_netcdf("data/splash_sf_example_data.nc")
