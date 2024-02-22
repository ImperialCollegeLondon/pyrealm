# SPLASH benchmark data

This directory contains the code and inputs used to generate the SPLASH benchmark
datasets used in unit testing `pyrealm.splash` and in regression tests against the
original SPLASH v1 implementation.

## Benchmark test data

The `splash_make_flux_benchmark_inputs.py` script is used to generate 100 random
locations around the globe with random dates, initial soil moisture, preciptiation,
cloud fraction and temperature (within reasonable bounds). This provides a robust test
of the calculations of various fluxes across a wide range of plausible value
combinations. The input data is created using:

```sh
python splash_make_flux_benchmark_inputs.py -o inputs.csv
```

The `splash_run_calc_daily_fluxes.py` script can then be used to run the inputs through
the original SPLASH implementation provided in the `splash_py_version` module.

```sh
python splash_run_calc_daily_fluxes.py -i inputs.csv -o benchmark_daily_fluxes.csv
```

## Gridded time series

This is a 20 x 20 cell spatial grid covering 2 years of daily data that is used to
validate the spin up of the initial moisture and the calculation of SPLASH water balance
over a time series. The dataset is generated using the `splash_make_grid.py` script,
which requires paths to local copies of the `WFDE5_v2` dataset and a version of the
`CRU TS` dataset.

```sh
python splash_make_grid.py \ 
   -w "/rds/general/project/lemontree/live/source/wfde5/wfde5_v2/" \ 
   -c "/rds/general/project/lemontree/live/source/cru_ts/cru_ts_4.0.4/" \ 
   -o "splash_test_grid_nw_us.nc"
```

The dataset can then be analysed using the original SPLASH implementation using the
script `splash_run_time_series_parallel.py`. This uses parallel processing to run
multiple cells simultaneously and will output the progress of the calculations.

```sh
python splash_run_time_series_parallel.py \ 
    -i "splash_test_grid_nw_us.nc" \ 
    -o "splash_test_grid_nw_us_out.nc"
```
