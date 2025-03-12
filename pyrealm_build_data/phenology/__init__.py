"""This submodule provides regression test data from the initial implementation of LAI
phenology calculations, provided by Boya Zhou. The input data consists of three files:

* **DE_GRI_hh_fluxnet_simple.csv**: This file is a subset of the original FluxNET
  dataset for the site (``FLX_DE-Gri_FLUXNET2015_FULLSET_HH_2004-2014_1-4.csv``). This
  original file contained the complete FluxNET data set for the 'DE-Gri' site at half
  hourly resolution, which includes 242 fields and is around 350 MB. The
  ``fluxnet_reducer.py`` script was used to remove fields not used in the calculations
  to reduce file size, creating the file ``DE_GRI_hh_fluxnet_simple.csv``.

* **DE_gri_splash_cru_ts4.07_2000_2019.nc**: This contains soil moisture data
  for the site, extracted from a global run of the pyrealm SPLASH model on the CRU TS
  4.07 data set (daily inputs, 0.5° resolution). The script ``splash_extractor.py`` was
  used to extract data from the global outputs for the single  cell containing the site
  coordinates.

* **DE-GRI_site_data.json**:  This contains required site data that is constant across
  all observations.

The script file ``python_implementation.py`` contains a pure Python reimplementation of
Boya Zhou's original workflow, put together by David Orme and Boya Zhou to bring all of
the calculations into Python using agreed inputs to create a repeatable regression test
dataset.

The script creates three output files to allow regression testing at three time scales:

* **python_hh_outputs.csv**: The predictions from the P Model of GPP at the half hourly
  scale, along with optimal chi and ci values

* **python_daily_outputs.csv**: Daily total GPP along with soil moisture stress factors
  and resulting penalised daily GPP, growing season definition and resulting time series
  in LAI and lagged LAI.

* **python_annual_outputs.csv**:  Annual values used in calculations including total
  annual assimilation, precipitation, number of growing days, mean carbon chi and VPD
  within the growing season and then annual values for maximum FAPAR, LAI and the m
  parameter.
"""  # noqa: D205, D415
