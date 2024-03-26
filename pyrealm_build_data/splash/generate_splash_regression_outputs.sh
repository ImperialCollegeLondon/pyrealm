#! /bin/bash

# This shell script runs the commands to process the benchmark input data using the
# original SLASH version 1 data. It runs the key commands described in the accompanying
# README.md.

python splash_run_calc_daily_fluxes.py -i data/daily_flux_benchmark_inputs.csv -o data/daily_flux_benchmark_outputs.csv
python splash_run_time_series_parallel.py -i "data/splash_sf_example_data.nc" -o "data/splash_sf_example_data_details.nc"
python splash_run_time_series_parallel.py -i "data/splash_nw_us_grid_data.nc"  -o "data/splash_nw_us_grid_data_outputs.nc"