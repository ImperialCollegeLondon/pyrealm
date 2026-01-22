"""Extract site soil moisture data for DE-GRI.

Simple script to extract a 20 year record of global SPLASH predictions based on CRU TS
4.07 data for the DE-Gri FluxNET site. The input files were run on the Imperial HPC
using pyrealm.
"""

from pathlib import Path

import xarray as xr

# Open a multi file dataset across years
paths = list(
    Path("/rds/general/project/lemontree/live/derived/splash_cru_ts4.07/data/").glob(
        "*.nc"
    )
)
ds = xr.open_mfdataset(paths)

# Select the cell closest to the site coordinates
site_data = ds.sel(lat=50.95, lon=13.5126, method="nearest")
site_data = site_data.sel(time=slice("2000-01-01", "2020-01-01"))

# Extract and save the dataset
site_data = site_data.compute()
site_data.to_netcdf("DE_gri_splash_cru_ts4.07.nc")
