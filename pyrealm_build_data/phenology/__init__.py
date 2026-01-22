"""This submodule provides regression test data for LAI phenology calculations

The `..._methods` directories contain the golden datasets of annual fapar max and daily
LAI from running a particular method. These are calculated using the data provided in
the `inputs directory`.

The `inputs/source` directory contains three original data files:

* **DE_GRI_hh_fluxnet_simple.csv**: This file is a subset of the original FluxNET
  dataset for the site (``FLX_DE-Gri_FLUXNET2015_FULLSET_HH_2004-2014_1-4.csv``). This
  original file contained the complete FluxNET data set for the 'DE-Gri' site at half
  hourly resolution, which includes 242 fields and is around 350 MB. The
  ``fluxnet_reducer.py`` script was used to remove fields not used in the calculations
  to reduce file size, creating the file ``DE_GRI_hh_fluxnet_simple.csv``.

* **DE_gri_splash_cru_ts4.07_2000_2019.nc**: This contains soil moisture data
  for the site, extracted from a global run of the pyrealm SPLASH model on the CRU TS
  4.07 data set (daily inputs, 0.5° resolution). The script ``splash_extractor.py`` was
  used to extract data from the global outputs for the single cell containing the site
  coordinates.

* **DE-GRI_site_data.json**: This contains required site data that is constant across
  all observations.

The `create_inputs.py` file then populates two directories of inputs for use in testing:

* `fortnightly`: this contains fortnightly summary data from the half hourly inputs for
  use with the standard PModel.

* `subdaily`: this contains outputs at the original 30 minute time scale for use with
  the subdaily PModel.

Each of those two directories contains identically structured files:

* `pmodel_inputs.csv`: processed and cleaned data in the correct units for fitting a P
  Model along with preciptation data and an indication of which observations are in the
  growing season.

* `pmodel_outputs.csv`:  GPP, ca, chi and ci values from fitting P Models using the
  data. For the fortnightly data this is a standard P Model, for the subdaily inputs
  this is a Subdaily model incorporating a soil moisture penalty.

* `annual_inputs.csv`: Annual summary data of the variables needed to calculate fapar
  max.

* `daily_assimilation.csv`: GPP and then molar assimilation at the daily timescale, by
  interpolation for fortnightly data and by aggregation for subdaily data.
"""  # noqa: D415
