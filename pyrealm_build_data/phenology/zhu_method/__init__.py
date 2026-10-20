"""Zhu method for calculating leaf area index phenology.

This directory contains code from Ziqi Zhu and  Mateusz Lisiewski, taken from Mateusz's
handover notes at the end of his postdoc. The reference source code has been brought
together into the `plmodel_timeseries.py` file, written by Ziqi Zhu.

The code is run against the annual summary stats using fortnightly inputs to generate
annual predictions of fAPAR max using the method. It then uses the interpolated daily A0
from the fortnightly example to generate predicted values of raw LAI, lagged LAI and
fAPAR.

The `zhu_method.py` script generates annual maximum fAPAR values for the subdaily and
fortnightly example datasets (`fapar_max_predictions.csv`) and then generates daily LAI
predictions from the daily A0 values for those datasets (`daily_lai_predictions.csv`).
It saves scalar values from the calculations in
`zhu_daily_lai_method_scalar_values.json`.
"""
