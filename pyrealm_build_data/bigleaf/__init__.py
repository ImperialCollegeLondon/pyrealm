"""This submodule contains benchmark outputs from the ``bigleaf`` package in ``R``,
which has been used as the basis for core hygrometry functions. The
``bigleaf_conversions.R`` R script runs a set of test values through `bigleaf`. The
first part of the file prints out some simple test values that have been used in package
doctests and then the second part of the file generates more complex benchmarking inputs
that are saved, along with `bigleaf` outputs as `bigleaf_test_values.json`.

Running ``bigleaf_conversions.R`` requires an installation of ``R`` along with the
``jsonlite`` and ``bigleaf`` packages, and the script can then be run from within the
submodule folder as:

.. code:: sh

    Rscript bigleaf_conversions.R

"""  # noqa: D205
