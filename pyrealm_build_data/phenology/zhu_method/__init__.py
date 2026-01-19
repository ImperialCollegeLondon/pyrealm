"""Zhu method for calculating leaf area index phenology.

This directory contains code from Ziqi Zhu and  Mateusz Lisiewski, taken from Mateusz's
handover notes at the end of his postdoc. The reference source code has been brought
together into the `plmodel_timeseries.py` file, written by Ziqi Zhu.

The code is run against the annual summary stats using fortnightly inputs to generate
annual predictions of fAPAR max using the method. It then uses the interpolated daily A0
from the fortnightly example to generate predicted values of raw LAI, lagged LAI and
fAPAR.

* The `fapar_limitation.py` script generates annual maximum fAPAR values in
  `pyrealm_build_data/phenology/zhu_method/`zhu_annual_fapar_max_from_fortnightly_data.csv`

* The `daily_lai_predictions.py` script generates daily LAI predictions in
  `/zhu_daily_lai_from_fortnightly_example.csv` and saves scalar values from the
  calculations in `zhu_daily_lai_method_scalar_values.json`.
"""
