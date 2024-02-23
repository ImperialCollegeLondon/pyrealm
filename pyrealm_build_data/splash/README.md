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
python splash_run_calc_daily_fluxes.py -i inputs.csv -o benchmark_daily_fluxes_2.csv
```

## Original time series

The SPLASH v1.0 implementation provided a time series of inputs for a single location
around San Francisco in 2000, with precipitation and temperature taken from WFDEI and
sunshine fraction interpolated from CRU TS. The original source data is included as
`data/example_data.csv`.

The original SPLASH `main.py` provides a simple example to run this code and output
water balance, which can be used as a direct benchmark without any wrapper scripts. With
the alterations to make the SPLASH code importable, the command below can be used to run
the code and capture the output:

```sh
python -m splash_py_version.main > test_example_out_original.csv
```

Note that this command also generates `main.log`, which contains over 54K lines of
logging and takes up over 6 Mb. This is not included in the `pyrealm` repo.

Because the `test_example_out_original.csv` file only contains predicted water balance,
the same input data is also run through a wrapper script to allow daily calculations to
be benchmarked in more detail. The first step is to use the `splash_make_example.py`
script to convert the CSV data into a properly dimensioned NetCDF file:

```sh
python splash_make_example.py
```

This creates the file `splash_test_example.nc`, which can be run using the original
SPLASH components using script `splash_run_time_series_parallel.py`.

```sh
python splash_run_time_series_parallel.py \ 
    -i "splash_test_example.nc" \ 
    -o "splash_test_example_out_new.nc"
```

## Gridded time series

This is a 20 x 20 cell spatial grid covering 2 years of daily data that is used to
validate the spin up of the initial moisture and the calculation of SPLASH water balance
over a time series. The dataset is generated using the `splash_make_grid.py` script,
which requires paths to local copies of the `WFDE5_v2` dataset and a version of the
`CRU TS` dataset. Note that the file paths below are examples and these data are not
included in the `pyrealm` repo.

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
