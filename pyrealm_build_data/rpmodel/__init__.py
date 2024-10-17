"""This submodule contains benchmark outputs from the ``rpmodel`` package in ``R``,
which has been used as the basis for initial development of the standard P Model.

Test inputs
===========

The ``generate_test_inputs.py`` file defines a set of constants for running P Model
calculations and then defines a set of scalar and array inputs for the forcing variables
required to run the P Model. The array inputs are set of 100 values sampled randomly
across the ranges of plausible forcing value inputs in order to benchmark the
calculations of the P Model implementation. All of these values are stored in the
``test_inputs.json`` file.

It requires ``python`` and the ``numpy`` package and can be run as:

.. code:: sh

    python generate_test_inputs.py

Simple `rpmodel` benchmarking
=============================

The ``test_outputs_rpmodel.R`` contains R code to run the test input data set, and store
the expected predictions from the ``rpmodel`` package as ``test_outputs_rpmodel.json``.
It requires an installation of ``R`` and the ``rpmodel`` package and can be run as:

.. code:: sh

    Rscript test_outputs_rpmodel.R

Global array test
=================

The remaining files in the submodule are intended to provide a global test dataset for
benchmarking the use of ``rpmodel`` on a global time-series, so using 3 dimensional
arrays with latitude, longitude and time coordinates. It is currently not used in
testing because of issues with the ``rpmodel`` package in version 1.2.0. It may also be
replaced in testing with the ``uk_data`` submodule, which is used as an example dataset
in the documentation.

The files are:

* ``pmodel_global.nc``: An input global NetCDF file containing forcing variables at 0.5°
  spatial resolution and for two time steps.
* ``test_global_array.R``: An R script to run ``rpmodel`` using the dataset.
* ``rpmodel_global_gpp_do_ftkphio.nc``: A NetCDF file containing ``rpmodel`` predictions
  using corrections for temperature effects on the `kphio` parameter.
* ``rpmodel_global_gpp_no_ftkphio.nc``: A NetCDF file containing ``rpmodel`` predictions
  with fixed ``kphio``.

To generate the predicted outputs again requires an R installation with the ``rpmodel``
package:

.. code:: sh

    Rscript test_global_array.R

"""  # noqa: D205
