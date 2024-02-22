"""Benchmark flux calculations.

This script is used to calculate the expected daily fluxes from the original SPLASH
implementation for use as a regression and unit test benchmark.
"""

import argparse
from pathlib import Path

import pandas  # type: ignore [import-untyped]
from splash_py_version.const import kCw, kWm  # type: ignore [import-not-found]
from splash_py_version.evap import EVAP  # type: ignore [import-not-found]


def splash_run_calc_daily_fluxes(input_file: str, output_file: str) -> None:
    """Calculate the daily fluxes for a set of locations.

    This function takes an input file where rows give locations, date, tc, sf and sw and
    then runs each row in turn through EVAP.calculate_daily_fluxes. That also runs the
    same data through the SOLAR.calculate_daily_fluxes method internally. The function
    then writes out a CSV files of all of the internal calculations of the two methods
    for use as a benchmark.
    """

    data = pandas.read_csv(input_file)
    results = []
    for _, row in data.iterrows():
        evap = EVAP(row.lat, row.elv)
        # Calculate evaporative supply rate
        sw = kCw * row.wn / kWm
        evap.calculate_daily_fluxes(
            n=row.julian_day, y=row.year, sf=row.sf, tc=row.tc, sw=sw
        )

        results.append(
            (
                evap.solar.my_nu,
                evap.solar.my_lambda,
                evap.solar.dr,
                evap.solar.delta,
                evap.solar.hs,
                evap.solar.ra_d,
                evap.solar.tau,
                evap.solar.ppfd_d,
                evap.solar.hn,
                evap.solar.rnl,
                evap.solar.rn_d,
                evap.solar.rnn_d,
                evap.sat,
                evap.lv,
                evap.pw,
                evap.psy,
                evap.econ,
                evap.cond,
                evap.eet_d,
                evap.pet_d,
                evap.rx,
                sw,
                evap.hi,
                evap.aet_d,
            )
        )

    out = pandas.DataFrame(
        results,
        columns=[
            "my_nu",
            "my_lambda",
            "dr",
            "delta",
            "hs",
            "ra_d",
            "tau",
            "ppfd_d",
            "hn",
            "rnl",
            "rn_d",
            "rnn_d",
            "sat",
            "lv",
            "pw",
            "psy",
            "econ",
            "cond",
            "eet_d",
            "pet_d",
            "rx",
            "sw",
            "hi",
            "aet_d",
        ],
    )
    out.to_csv(output_file, index=False, float_format="%0.8e")


###############################################################################
# MAIN PROGRAM
###############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser("splash_run_calc_daily_fluxes")
    parser.add_argument("-i", "--input_file", type=Path)
    parser.add_argument("-o", "--output_file", type=Path)

    args = parser.parse_args()

    splash_run_calc_daily_fluxes(args.input_file, args.output_file)
