"""Script to create a small UK time series from WFDE5 data.

Requires:
module load anaconda3/personal
module load gsl
module load nco/4.7.3
module load tools/dev
module load CDO/2.0.5-gompi-2021b
"""

import os
import subprocess

import numpy as np
import xarray

# identify subset
years = [2018]
months = [6, 7]
lat_min = 49.0
lat_max = 60.0
lon_min = -10.0
lon_max = 4.0


# Source dir root
src_root = "/rds/general/project/lemontree/live/source"

# WFDE v5 - 0.5°, hourly

wfde_var = ["PSurf", "Qair", "Tair", "SWdown"]
lon_sel = None
lat_sel = None


output_datasets = []

for var in wfde_var:
    var_datasets = []

    for yr in years:
        for mn in months:
            # Get the path and load the dataset
            path = f"wfde5/wfde5_v2/{var}/{yr}/{var}_WFDE5_CRU_{yr}{mn:02d}_v2.0.nc"
            path = os.path.join(src_root, path)
            ds = xarray.load_dataset(path)

            if lon_sel is None:
                lon_sel = np.where(
                    np.logical_and(ds["lon"] >= lon_min, ds["lon"] <= lon_max)
                )[0]
            if lat_sel is None:
                lat_sel = np.where(
                    np.logical_and(ds["lat"] >= lat_min, ds["lat"] <= lat_max)
                )[0]

            uk = ds.isel(lat=lat_sel, lon=lon_sel)
            var_datasets.append(uk)

    output_datasets.append(xarray.concat(var_datasets, dim="time"))

uk_wfde5 = xarray.merge(output_datasets)

# FAPAR - 0.05°, daily
# Use 'ncea' to subset - much faster than loading the dataset into xarray first

for yr in years:
    path = os.path.join(src_root, f"SNU_Ryu_FPAR_LAI/FPAR_netcdf_cf/FPAR_{yr}.nc")

    ncea_cmd = [
        "ncea",
        "-d",
        f"latitude,{lat_min},{lat_max}",
        "-d",
        f"longitude,{lon_min},{lon_max}",
        path,
        "fpar_sub.nc",
    ]

    ncea_run = subprocess.run(ncea_cmd)

    # Load the file
    fpar = xarray.load_dataset("fpar_sub.nc")

    # Get a boolean index on the requested months
    fpar_month = [
        x.month for x in fpar["time"].to_numpy().astype("datetime64[D]").tolist()
    ]
    fpar_month = [m in months for m in fpar_month]
    fpar = fpar.isel(time=fpar_month)

    n_days = fpar.dims["time"]

    # Aggregate fAPAR to 0.5 degrees
    fpar_np = fpar["FPAR"].to_numpy()

    fpar_half_degree = np.nanmean(fpar_np.reshape(n_days, 22, 10, 28, 10), axis=(2, 4))

    # Repeat daily values for hourly obs
    fpar_half_degree_hourly = fpar_half_degree[np.repeat(np.arange(n_days), 24), ...]

    # Different direction of latitude axis
    uk_wfde5["fAPAR"] = ("time", "lat", "lon"), np.flip(fpar_half_degree_hourly, axis=1)


uk_wfde5.to_netcdf("UK_WFDE5_FAPAR_2018_JuneJuly.nc")
