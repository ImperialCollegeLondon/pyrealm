---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3
  language: python
  name: pyrealm_python3
---

# The `pyrealm_build_data` package

The `pyrealm` repository includes both the `pyrealm` package and the
`pyrealm_build_data` package. The `pyrealm_build_data` package contains datasets that
are used in the `pyrealm` build and testing process. This includes:

* Example datasets that are used in the package documentation, such as simple spatial
  datasets for showing the use of the P Model.
* "Golden" datasets for regression testing `pyrealm` implementations against the outputs
  of other implementations. These datasets will include a set of input data and then
  output predictions from other implementations.
* Datasets for providing profiling of `pyrealm` code and for benchmarking new versions
  of the package code against earlier implementations to check for performance issues.

Note that `pyrealm_build_data` is a source distribution only (`sdist`) component of
`pyrealm`, so is not included in binary distributions (`wheel`) that are typically
installed by end users. This means that files in `pyrealm_build_data` are not available
if a user has simply used `pip install pyrealm`: please *do not* use
`pyrealm_build_data` within the main `pyrealm` code.

## Package contents

The package is organised into submodules that reflect the data use or previous
implementation.

### The `bigleaf` submodule

This submodule contains benchmark outputs from the `bigleaf` package in `R`, which has
been used as the basis for core hygrometry functions. The `bigleaf_conversions.R` R
script runs a set of test values through `bigleaf`. The first part of the file prints
out some simple test values that have been used in package doctests and then the second
part of the file generates more complex benchmarking inputs that are saved, along with
`bigleaf` outputs as `bigleaf_test_values.json`.

Running `bigleaf_conversions.R` requires an installation of R along with the `jsonlite`
and `bigleaf` packages, and the script can then be run from within the submodule folder
as:

```sh
Rscript bigleaf_conversions.R
```

### The `rpmodel` submodule

This submodule contains benchmark outputs from the `rpmodel` package in `R`, which has
been used as the basis for initial development of the standard P Model.

#### Test inputs

The `generate_test_inputs.py` file defines a set of constants for running P Model
calculations and then defines a set of scalar and array inputs for the forcing variables
required to run the P Model. The array inputs are set of 100 values sampled randomly
across the ranges of plausible forcing value inputs in order to benchmark the
calculations of the P Model implementation. All of these values are stored in the
`test_inputs.json` file.

It requires `python` and the `numpy` package and can be run as:

```sh
python generate_test_inputs.py
```

#### Simple `rpmodel` benchmarking

The `test_outputs_rpmodel.R` contains R code to run the test input data set, and store
the expected predictions from the `rpmodel` package as `test_outputs_rpmodel.json`. It
requires an installation of `R` and the `rpmodel` package and can be run as:

```sh
Rscript test_outputs_rpmodel.R
```

#### Global array test

The remaining files in the submodule are intended to provide a global test dataset for
benchmarking the use of `rpmodel` on a global time-series, so using 3 dimensional arrays
with latitude, longitude and time coordinates. It is currently not used in testing
because of issues with the `rpmodel` package in version 1.2.0. It may also be replaced
in testing with the `uk_data` submodule, which is used as an example dataset in the
documentation.

The files are:

* pmodel_global.nc: An input global NetCDF file containing forcing variables at 0.5°
  spatial resolution and for two time steps.
* test_global_array.R: An R script to run `rpmodel` using the dataset.
* rpmodel_global_gpp_do_ftkphio.nc: A NetCDF file containing `rpmodel` predictions using
 corrections for temperature effects on the `kphio` parameter.
* rpmodel_global_gpp_no_ftkphio.nc: A NetCDF file containing `rpmodel` predictions with
  fixed `kphio`.

To generate the predicted outputs again requires an R installation with the `rpmodel`
package:

```sh
Rscript test_global_array.R
```

### The `subdaily` submodule

At present, this submodule only contains a single file containing the predictions for
the `BE_Vie` fluxnet site from the original implementation of the `subdaily` module,
published in {cite}`mengoli:2022a`. Generating these predictions requires an
installation of R and then code from the following repository:

[https://github.com/GiuliaMengoli/P-model_subDaily](https://github.com/GiuliaMengoli/P-model_subDaily)

TODO - This submodule should be updated to include the required code along with the
settings files and a runner script to reproduce this code. Or possibly to checkout the
required code as part of a shell script.

### The `t_model` submodule

The `t_model.r` contains the original implementation of the T Model calculations in R
{cite:p}`Li:2014bc`. The `rtmodel_test_outputs.r` script sources this file and then
generates some simple bencmarking predictions, which are saved as `rtmodel_output.csv`.

To generate the predicted outputs again requires an R installation

```sh
Rscript rtmodel_test_outputs.r
```

### The `uk_data` submodule

This submodule contains the Python script `create_2D_uk_inputs.py`, which is used to
generate the NetCDF output file `UK_WFDE5_FAPAR_2018_JuneJuly.nc`. This contains P Model
forcings for the United Kingdom at 0.5° spatial resolution and hourly temporal
resolution over 2 months (1464 temporal observations). It is used for demonstrating the
use of the subdaily P Model.

The script is currently written with a hard-coded set of paths to key source data - the
WFDE5 v2 climate data and a separate source of interpolated hourly fAPAR. This should
probably be rewritten to generate reproducible content from publically available sources
of these datasets.
