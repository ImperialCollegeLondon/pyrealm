"""Benchmark flux calculations.

This script is used to calculate the expected daily fluxes from the original SPLASH
implementation for use as a regression and unit test benchmark.
"""

import argparse
from pathlib import Path

import pandas  # type: ignore [import-untyped]
from splash_py_version.splash import SPLASH  # type: ignore [import-not-found]


def splash_run_calc_daily_fluxes(input_file: Path, output_file: Path) -> None:
    """Calculate the daily fluxes for a set of locations.

    This function takes an input file where rows give locations, date, tc, sf and sw and
    then runs each row in turn through SPLASH.run_one_day. That also runs the same data
    through the EVAP.calculate_daily_fluxes and SOLAR.calculate_daily_fluxes method
    internally. The function then writes out a CSV files of all of the internal
    calculations for use as a benchmark.
    """

    data = pandas.read_csv(input_file)
    results = []
    for _, row in data.iterrows():
        print(row.dates)
        # evap = EVAP(row.lat, row.elv)
        # # Calculate evaporative supply rate
        # sw = kCw * row.wn / kWm
        # evap.calculate_daily_fluxes(
        #     n=row.julian_day, y=row.year, sf=row.sf, tc=row.tc, sw=sw
        # )

        # Calculate daily values
        splash = SPLASH(row.lat, row.elv)
        splash.run_one_day(
            n=row.julian_day, y=row.year, wn=row.wn, sf=row.sf, tc=row.tc, pn=row.pn
        )

        results.append(
            (
                splash.evap.solar.my_nu,
                splash.evap.solar.my_lambda,
                splash.evap.solar.dr,
                splash.evap.solar.delta,
                splash.evap.solar.hs,
                splash.evap.solar.ra_d,
                splash.evap.solar.tau,
                splash.evap.solar.ppfd_d,
                splash.evap.solar.hn,
                splash.evap.solar.rnl,
                splash.evap.solar.rn_d,
                splash.evap.solar.rnn_d,
                splash.evap.sat,
                splash.evap.lv,
                splash.evap.pw,
                splash.evap.psy,
                splash.evap.econ,
                splash.evap.rx,
                splash.evap.hi,
                splash.evap.cond,
                splash.evap.eet_d,
                splash.evap.pet_d,
                splash.evap.aet_d,
                splash.wn,
                splash.ro,
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
            "rx",
            "hi",
            "cond",
            "eet_d",
            "pet_d",
            "aet_d",
            "wn",
            "ro",
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
