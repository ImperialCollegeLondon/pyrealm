# SPLASH benchmark data

This directory contains the code and inputs used to generate the SPLASH benchmark
datasets used in unit testing `pyrealm.splash` and in regression tests against the
original SPLASH v1 implementation.

## Benchmark test data

The `splash_make_flux_benchmark_inputs.py` script is used to generate 100 random
locations around the globe with random dates, initial soil moisture, preciptiation,
cloud fraction and temperature (within reasonable bounds). This provides a robust test
of the calculations across a wide range of plausible value combinations. The input data
is created using:

```sh
python splash_make_flux_benchmark_inputs.py -o inputs.csv
```

The `splash_run_calc_daily_fluxes.py` script can then be used to run the inputs through
the original SPLASH implementation provided in the `splash_py_version` module.

```sh
python splash_run_calc_daily_fluxes.py -i inputs.csv -o benchmark_daily_fluxes.csv
```
