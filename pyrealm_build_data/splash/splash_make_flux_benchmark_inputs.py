"""Make benchmark inputs for calculating daily fluxes.

This script generates 100 daily observations of dates, lat, elev, wn, tc, pn and sf
across a wide range of input values for each variable. This provides a robust benchmark
set of inputs for running through the original SPLASH and the pyrealm implementation.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas  # type: ignore [import-untyped]
from splash_py_version.evap import EVAP  # type: ignore [import-not-found]

from pyrealm.core.calendar import Calendar


def splash_make_flux_benchmark_inputs(output_file: str) -> None:
    """Generates a benchmark input file for SPLASH testing."""

    n_inputs = 100

    # generate random dates
    d_start = np.datetime64("1980-01-01")
    d_end = np.datetime64("2020-12-31")
    diff = d_end - d_start
    rand_day = np.random.choice(diff.astype(int), size=n_inputs)
    dates = d_start + rand_day
    calendar = Calendar(dates)

    inputs_dict = {
        "lat": np.random.uniform(low=-90, high=90, size=n_inputs),
        "elv": np.random.uniform(low=0, high=1000, size=n_inputs),
        "dates": dates,
        "year": calendar.year,
        "julian_day": calendar.julian_day,
        "days_in_year": calendar.days_in_year,
        "wn": np.random.uniform(low=0, high=150, size=n_inputs),
        "sf": np.random.uniform(low=0, high=1, size=n_inputs),
        "tc": np.random.uniform(low=0, high=35, size=n_inputs),
        "pn": np.random.uniform(low=0, high=500, size=n_inputs),
    }

    inputs = pandas.DataFrame(inputs_dict)

    # Calculate pressure - odd location of static function as class method. Note that
    # this is equivalent to pyrealm calc_patm but uses a different standard temperature.
    evap = EVAP(lat=0, elv=0)
    inputs["pa"] = evap.elv2pres(inputs["elv"])

    inputs.to_csv(output_file, index=False, float_format="%0.10e")

    return


###############################################################################
# MAIN PROGRAM
###############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser("run_splash_make_flux_benchmark_inputs")
    parser.add_argument("-o", "--output_file", type=Path)
    args = parser.parse_args()
    splash_make_flux_benchmark_inputs(args.output_file)
