"""Zhu method for calculating leaf area index phenology.

This directory contains code from Ziqi Zhu and  Mateusz Lisiewski, taken from Mateusz's
handover notes at the end of his postdoc.

The code is run against the annual summary stats using fortnightly inputs to generate
annual predictions of fAPAR max using the method. It then uses the interpolated daily A0
from the fortnightly example to generate predicted values of raw LAI, lagged LAI and
fAPAR.
"""
