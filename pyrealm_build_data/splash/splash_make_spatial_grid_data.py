"""Make a test input grid for SPLASH from the WFDE5 and CRU datasets.

This exports a gridded dataset for NW Coast USA that also covers the site of original
test data provided with the SPLASH v1 implementation. The result is a 20 x 20 spatial
grid with 731 daily observations.

The script needs to be run in a location where paths can be provided to the WFDE5 and
CRU TS datasets.
"""

import argparse
from pathlib import Path

import numpy as np
import xarray


def splash_make_grid(wfde5_base: Path, cruts_base: Path, output: Path) -> None:
    """Generate a gridded dataset from WFDE5 and CRU TS data."""

    # # Alternative setup for a UK focussed dataset
    # latslice = slice(49.5, 60)
    # lonslice = slice(-11, 2.5)
    # timeslice_wfde = slice(
    #     np.datetime64("2004-01-01 00:00"), np.datetime64("2005-12-31 23:59")
    # )
    # timeslice_cru = slice(np.datetime64("2004-01-01"), np.datetime64("2006-02-01"))

    # NW Coast USA
    # - time slice for cru needs to sample beyond the subset to allow forward fill to
    #   extrapolate to the total length
    latslice = slice(36.0, 46.0)
    lonslice = slice(-125.0, -115.0)
    timeslice_wfde = slice(
        np.datetime64("2000-01-01 00:00"), np.datetime64("2001-12-31 23:59")
    )
    timeslice_cru = slice(np.datetime64("2000-01-01"), np.datetime64("2002-02-01"))

    # Extract daily mean temperature from WDFE5
    temp_files = (wfde5_base / "Tair").rglob("*.nc")
    temp_source = xarray.open_mfdataset(list(temp_files))

    temp_subset = temp_source.sel(lat=latslice, lon=lonslice, time=timeslice_wfde)
    temp_daily = temp_subset.resample(time="1D").mean()

    # Extract daily average rainfall from WDFE5 in kg/m2/s
    prec_files = (wfde5_base / "Rainf").rglob("*.nc")
    prec_source = xarray.open_mfdataset(list(prec_files))

    prec_subset = prec_source.sel(lat=latslice, lon=lonslice, time=timeslice_wfde)
    prec_daily = prec_subset.resample(time="1D").mean()

    # Generate sunshine fraction by resampling from CRU monthly values to daily
    cld_files = (cruts_base / "data/cld").rglob("*.nc.gz")
    cld_source = xarray.open_mfdataset(list(cld_files))

    # Get subset - extending through to following month, reset dates to start of month
    # for ffill and fill time series and then reduce to length of the other two
    # variables
    cld_subset = cld_source.sel(lat=latslice, lon=lonslice, time=timeslice_cru)
    cldtimes = cld_subset.time.values.astype("datetime64[M]").astype("datetime64[ns]")
    cld_subset.coords["time"] = cldtimes
    cld_daily = cld_subset.resample(time="1D").ffill()
    cld_daily = cld_daily.sel(time=timeslice_wfde)

    # Extract elevation from WFDE5
    elev = xarray.open_dataset(wfde5_base / "Elev/ASurf_WFDE5_CRU_v2.0.nc")
    elev = elev.sel(lat=latslice, lon=lonslice)

    splash_grid = xarray.Dataset(
        {
            "elev": elev["ASurf"],
            "tmp": temp_daily["Tair"] - 273.15,  # convert from Kelvin to °C
            "pre": prec_daily["Rainf"] * 86400,  # convert from kg/m2/s1 to mm/day
            "sf": 1 - (cld_daily["cld"] / 100),  # 1 - cloud cover as proportion
        }
    )

    splash_grid.to_netcdf(output)


###############################################################################
# MAIN PROGRAM e.g.
# python splash_make_grid \
#    -w "/rds/general/project/lemontree/live/source/wfde5/wfde5_v2/" \
#    -c "/rds/general/project/lemontree/live/source/cru_ts/cru_ts_4.0.4/" \
#    -o "splash_test_grid_nw_us.nc"
###############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser("splash_make_grid")
    parser.add_argument("-w", "--wfde5_base", type=Path)
    parser.add_argument("-c", "--cruts_base", type=Path)
    parser.add_argument("-o", "--output", type=Path)

    args = parser.parse_args()
    splash_make_grid(
        wfde5_base=args.wfde5_base, cruts_base=args.cruts_base, output=args.output
    )
