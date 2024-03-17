---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: pyrealm_python3
  language: python
  name: pyrealm_python3
---

# Missing data in the subdaily model

The acclimation process in the subdaily model uses a weighted mean of the daily
acclimation conditions with preceeding daily conditions. When there are missing data,
this results in problems in the estimation of the realised daily values for $\xi$,
$V_{cmax25}$ and $J_{max25}$.

The key problem is if there are missing values in the observations set as the
acclimation window for the subdaily model. The daily optimal behaviour for the plant is
calculated using the average values of forcing variables within the window and missing
values within the window will lead to the average value also being missing. This
propagates through the weighted mean process - all days following the missing data will
also be missing.

The subdaily model provides two options to help deal with missing data.

1. The acclimation process can be set to **allow partial acclimation data** in the
   calculation of daily average values by ignoring missing data and taking the average
   of the available observations (`allow_partial_data`).

   However, allowing partial data cannot solve the issue where there is no data is
   present in the acclimation window for a day or where the P Model calculations are
   undefined for the optimal behaviour on a day. Both of these cases will lead to a
   missing value in the time series of daily optimal values, which would then be
   propagated through the calculation of realised values using weighted averages.

1. Hence the second option is to allow the iterated calculation of daily realised values
   to hold over the last valid value until the next valid daily optimal value is found
   (`allow_holdover`). If the first value is missing, this is held over until the first
   valid observation.

These do not fix all problems: the best way forward depends partly on the source of the
missing data and how common it is, as discused below.

## Sources of missing data

There are three ways that missing data can occur:

### Simple data gaps

Your data might simply be incomplete and have missing data through the time series. Note
that the main problems only arise if you have **missing data during the acclimation
window**, because this prevents the calculation of the realised values of $\xi$,
$V_{cmax25}$ and $J_{max25}$ at the subdaily time scale. Missing values at other points
in the daily cycle simply lead to missing predictions at individual observations.

You can fix this problem by interpolating your missing data, possibly using methods from
the :mod:`scipy.interpolate` module. With sparse missing data, you may also be able to
use the `allow_partial_data` option, but if any day has _no_ valid data during the
acclimation window for a variable, the model will still fail.

### Incomplete start and end days

It is common for observation data to start or end part way through a day, often as a
result of converting UTC times to local time. If the observations on the first and last
day only _partly cover the acclimation window_, then there is effectively missing
acclimation data.

You could simply truncate the data to complete days or extrapolate the data to fill in
the missing start and/or end observations. Using the subdaily model options:

* The `allow_holdover` option will skip over the first day - you will not get
  predictions until the following day, but then the rest of the calculations will
  continue as normal.
* The `allow_partial_data` will allow the estimation of optimal values on the first and
  last days and extend the predictions to the start of the data.

### Undefined behaviour in the P Model

The P Model itself can generate undefined values as part of the calculation of optimal
$\chi$, resulting in missing values in the daily model of optimal behaviour. This can
occur even when the data is complete and can only be fixed by using the `allow_holdover`
option.
