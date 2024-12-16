"""This submodule provides P Model forcings for the United Kingdom at 0.5° spatial
resolution and hourly temporal resolution over 2 months (1464 temporal observations). It
is used for demonstrating the use of the subdaily P Model.

The Python script ``create_2D_uk_inputs.py``  is used to generate the NetCDF output file
``UK_WFDE5_FAPAR_2018_JuneJuly.nc``. The script is currently written with a hard-coded
set of paths to key source data - the WFDE5 v2 climate data and a separate source of
interpolated hourly fAPAR. This should probably be rewritten to generate reproducible
content from publically available sources of these datasets.
"""  # noqa: D205
