"""Wrapper script to run the original SPLASH implementation across gridded inputs."""

import argparse
from dataclasses import dataclass, field
from datetime import datetime
from itertools import repeat
from pathlib import Path

import numpy as np
import xarray
from multiprocess.pool import Pool
from splash_py_version.splash import SPLASH  # type: ignore [import-not-found]


@dataclass
class SpinUpData:
    """Lightweight dropin replacement for complex SPLASH DATA class.

    This replicates the key attributes of the more complex class to allow data to be be
    passed to the spin up process.
    """

    pn_vec: np.ndarray
    tair_vec: np.ndarray
    sf_vec: np.ndarray
    year: int
    npoints: int = field(init=False)

    def __post_init__(self) -> None:
        self.npoints = len(self.pn_vec)
        self.num_lines = len(self.pn_vec)


def run_one_cell(
    coords: tuple[int, float, int, float, float], input_data: xarray.Dataset
) -> dict:
    """Run a time series extracted from one cell of the input data."""

    # Get the coords
    lat_idx, lat, lon_idx, lon, cell_elev = coords

    # date and time variables - repetitious, could move outside but meh.
    years = input_data.time.dt.year
    julian_day = input_data.time.dt.dayofyear
    year_one = years == years[0]

    # Create the SPLASH instance
    print(f"Running cell at {float(lon)} {float(lat)} {float(cell_elev)}")

    if np.isnan(float(cell_elev)):
        print(" - Water cell - skipping")
        return dict(coords=coords)

    splash = SPLASH(lat, cell_elev)

    # extract data
    pn_vec = input_data["pre"].sel(lat=lat, lon=lon).values
    tair_vec = input_data["tmp"].sel(lat=lat, lon=lon).values
    sf_vec = input_data["sf"].sel(lat=lat, lon=lon).values

    spin_up_data = SpinUpData(
        pn_vec=pn_vec[year_one],
        tair_vec=tair_vec[year_one],
        sf_vec=sf_vec[year_one],
        year=int(years[0].values),
    )

    # Spin up the data and store the final value
    splash.spin_up(spin_up_data)
    wn_spun_up = splash.wn
    wn_curr = splash.wn

    nt = len(years)
    ppfd_d = np.empty((nt))
    rn_d = np.empty((nt))
    rnn_d = np.empty((nt))
    cond = np.empty((nt))
    eet = np.empty((nt))
    aet = np.empty((nt))
    pet = np.empty((nt))
    wn = np.empty((nt))
    ro = np.empty((nt))

    for day_idx, this_day in enumerate(input_data.time):
        splash.run_one_day(
            n=julian_day[day_idx],
            y=int(years[day_idx]),
            wn=wn_curr,
            sf=sf_vec[day_idx],
            tc=tair_vec[day_idx],
            pn=pn_vec[day_idx],
        )
        ppfd_d[day_idx] = splash.evap.solar.ppfd_d
        rn_d[day_idx] = splash.evap.solar.rn_d
        rnn_d[day_idx] = splash.evap.solar.rnn_d
        cond[day_idx] = splash.cond
        eet[day_idx] = splash.eet
        pet[day_idx] = splash.pet
        aet[day_idx] = splash.aet
        wn[day_idx] = splash.wn
        ro[day_idx] = splash.ro

    return dict(
        coords=coords,
        wn_spun_up=wn_spun_up,
        ppfd_d=ppfd_d,
        rn_d=rn_d,
        rnn_d=rnn_d,
        cond=cond,
        eet=eet,
        pet=pet,
        aet=aet,
        wn=wn,
        ro=ro,
    )


def run_splash_time_series(
    input_file: Path, output_file: Path, n_cores: int = 8
) -> None:
    """Run a SPLASH time series for gridded data."""

    start = datetime.now()
    n_real = 0

    # Import the test data file
    input_data = xarray.load_dataset(input_file)

    # Create storage for the results: evap and core solar
    ppfd_d = xarray.DataArray(coords=input_data.coords)
    rn_d = xarray.DataArray(coords=input_data.coords)
    rnn_d = xarray.DataArray(coords=input_data.coords)
    cond = xarray.DataArray(coords=input_data.coords)
    eet = xarray.DataArray(coords=input_data.coords)
    pet = xarray.DataArray(coords=input_data.coords)
    aet = xarray.DataArray(coords=input_data.coords)
    wn = xarray.DataArray(coords=input_data.coords)
    ro = xarray.DataArray(coords=input_data.coords)
    wn_spun_up = xarray.DataArray(
        coords={"lat": input_data.lat, "lon": input_data.lon}, dims=["lat", "lon"]
    )

    # Create coords tuples to pass to run_one_cell
    coords = []
    for lat_idx, lat in enumerate(input_data.lat):
        for lon_idx, lon in enumerate(input_data.lon):
            # Create the SPLASH instance
            cell_elev = input_data["elev"][lat_idx, lon_idx]
            coords.append(
                (
                    lat_idx,
                    float(lat.data),
                    lon_idx,
                    float(lon.data),
                    float(cell_elev.data),
                )
            )

    # Use a pool of processes to run the set of cells
    with Pool(n_cores) as pool:
        result = pool.starmap_async(run_one_cell, zip(coords, repeat(input_data)))

        results = result.get()

    # Compile results
    for res in results:
        lat_idx, _, lon_idx, _, cell_elev = res["coords"]

        if not np.isnan(cell_elev):
            n_real += 1
        else:
            continue

        ppfd_d[lat_idx, lon_idx, :] = res["ppfd_d"]
        rn_d[lat_idx, lon_idx, :] = res["rn_d"]
        rnn_d[lat_idx, lon_idx, :] = res["rnn_d"]
        cond[lat_idx, lon_idx, :] = res["cond"]
        eet[lat_idx, lon_idx, :] = res["eet"]
        pet[lat_idx, lon_idx, :] = res["pet"]
        aet[lat_idx, lon_idx, :] = res["aet"]
        wn[lat_idx, lon_idx, :] = res["wn"]
        ro[lat_idx, lon_idx, :] = res["ro"]
        wn_spun_up[lat_idx, lon_idx] = res["wn_spun_up"]

    out = xarray.Dataset(
        {
            "ppfd": ppfd_d,
            "rn_d": rn_d,
            "rnn_d": rnn_d,
            "cond": cond,
            "eet": eet,
            "pet": pet,
            "aet": aet,
            "wn": wn,
            "ro": ro,
            "wn_spun_up": wn_spun_up,
        }
    )
    out.to_netcdf(output_file)

    end = datetime.now()
    print(f"{n_real} land cells calculated in {(end - start).seconds}")


###############################################################################
# MAIN PROGRAM
###############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser("splash_run_time_series_parallel")
    parser.add_argument("-i", "--input_file", type=Path)
    parser.add_argument("-o", "--output_file", type=Path)
    parser.add_argument("-n", "--n_cores", type=int, default=8)

    args = parser.parse_args()

    run_splash_time_series(
        input_file=args.input_file, output_file=args.output_file, n_cores=args.n_cores
    )
