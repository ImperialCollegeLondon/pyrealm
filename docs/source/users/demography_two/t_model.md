---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
language_info:
  name: python
  version: 3.12.3
  mimetype: text/x-python
  codemirror_mode:
    name: ipython
    version: 3
  pygments_lexer: ipython3
  nbconvert_exporter: python
  file_extension: .py
---

# The T Model module

```{admonition} Run this notebook
:class: hint

* Read the guide on setting up your computer to [run Jupyter
  notebooks](../getting_started.md)
* Download {nb-download}`this notebook<./t_model.ipynb>` as a Jupyter notebook.

```

The T Model {cite}`Li:2014bc` provides a model of both:

* stem :term:`allometry`, given a diameter at breast height (DBH) for a stem and the
  [stem traits](./flora.md) from its plant functional type (PFT), and
* carbon allocation, given stem allometry and potential GPP.

```{code-cell} ipython3
import warnings

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from pyrealm.demography_two.flora import Flora
from pyrealm.demography_two.cohorts import Cohorts
from pyrealm.demography_two.tmodel import (
    StemAllocation,
    StemAllometry,
    calculate_whole_crown_gpp,
)

warnings.filterwarnings(
    "ignore",
    category=ExperimentalFeatureWarning,
)
```

To generate predictions under the T Model, we need a set of cohorts: this defines a sets
of stems with known traits from a Flora object at a known DBH.

```{code-cell} ipython3
# Create a flora with 3 PFTs with different maximum heights
flora = Flora(name=["short", "medium", "tall"], h_max=[10, 20, 30])

cohorts = Cohorts(
    flora=flora,
    pft_name=np.array(["short", "medium", "tall"]),
    dbh_value=np.array([0.1, 0.1, 0.1]),
    n_individuals=np.array([1, 1, 1]),
)
```

## Stem allometry

We can visualise how the stem size, canopy size and various masses of PFTs change with
stem diameter by using the {class}`~pyrealm.demography.tmodel.StemAllometry`
class. Creating a `StemAllometry` instance needs an existing `Flora` instance and an
array of values for diameter at breast height (DBH, metres). The returned class contains
the predictions of the T Model for:

* Stem height (`stem_height`, m),
* Crown area (`crown_area`, m2),
* Crown fraction (`crown_fraction`, -),
* Stem mass (`stem_mass`, kg),
* Foliage mass (`foliage_mass`, kg),
* Sapwood mass (`sapwood_mass`, kg),
* Crown radius scaling factor (`crown_r0`, -), and
* Height of maximum crown radius (`crown_z_max`, m).

Note that {attr}`~pyrealm.demography.tmodel.StemAllometry.stem_height` denotes the total
tree height, as used interchangeable in {cite:t}`Li:2014bc`, rather than just the height
of the trunk below the canopy.

The DBH input can be a scalar array or a one dimensional array providing a single value
for each PFT. This then calculates a single estimate at the given size for each stem.

```{code-cell} ipython3
# Calculate the allometry for the cohorts
cohort_allometry = StemAllometry(cohorts=cohorts)
cohort_allometry
```

The {meth}`~pyrealm.demography_two.tmodel.StemAllometry` class provides the
{meth}`~pyrealm.demography_two.StemAllometry.to_dataframe()` method to export the stem
data for data exploration. The `StemAllometry` data retains the unique cohort ids and
DBH from the original cohort data.

```{code-cell} ipython3
cohort_allometry.to_dataframe().transpose()
```

### Allometry profiles

The `at_dbh` argument to `StemAllometry` can be used to generate profiles of the
allometry predictions for PFTs at different sizes. The provided values are used to
calculate predictions instead of the cohort DBH values. The allometry attributes of
predictions are then 2 dimensional arraus arranged with each cohort as a column and each
DBH prediction as a row. This makes them convenient to plot using `matplotlib`.

```{code-cell} ipython3
# Column array of DBH values from 0.01 to 1.6 metres
dbh_profile = np.arange(0.01, 1.6, 0.01)
# Get the predictions at those DBH values.
allometry_profiles = StemAllometry(cohorts=cohorts, at_dbh=dbh_profile)
```

The code below shows how to use the returned allometries to generate a plot of the
scaling relationships across all of the PFTs in a `Flora` instance.

```{code-cell} ipython3
fig, axes = plt.subplots(ncols=2, nrows=4, sharex=True, figsize=(10, 10))

plot_details = [
    ("stem_height", "Stem height (m)"),
    ("crown_area", "Crown area (m2)"),
    ("crown_fraction", "Crown fraction (-)"),
    ("stem_mass", "Stem mass (kg)"),
    ("foliage_mass", "Foliage mass (kg)"),
    ("sapwood_mass", "Sapwood mass (kg)"),
    ("crown_r0", "Crown scaling factor (-)"),
    ("crown_z_max", "Height of maximum\ncrown radius (m)"),
]

for ax, (var, ylab) in zip(axes.flatten(), plot_details):
    ax.plot(dbh_profile, getattr(allometry_profiles, var), label=flora.name)
    ax.set_xlabel("Diameter at breast height (m)")
    ax.set_ylabel(ylab)

    if var == "crown_area":
        ax.legend(frameon=False)
```

The {meth}`~pyrealm.demography_two.StemAllometry.to_dataframe()` method can still be
used, but the values are stacked into columns identified by pairings of cohort ID and
DBH.

```{code-cell} ipython3
allometry_profiles.to_dataframe().head(6)[
    ["cohort_ids", "dbh", "stem_height", "crown_area", "crown_fraction"]
]
```

## Productivity allocation

The T Model also predicts how GPP will be allocated to respiration, turnover
and growth for stems with a given PFT and allometry using the
{meth}`~pyrealm.demography.tmodel.StemAllometry` class.

This requires an estimate of the GPP available to a stem. The original implementation of
the T Model implemented this (Equation 12, {cite:alp}`Li:2014bc`)using an estimate of
the potential GPP per square metre ($P_0$), scaled up to the crown area of the stem
($A_c$) and using the Beer-Lambert equation to estimate the proportion of potential GPP
captured by the crown as a function of the canopy light extinction coefficient ($k$) and
the canopy {term}`leaf area index<LAI>` ($L$):

$$
\textrm{GPP} =  P_0 A_c (1 - e^{-kL})
$$

This is implemented in the function `calculate_whole_crown_gpp`:

```{code-cell} ipython3
whole_crown_gpp = calculate_whole_crown_gpp(
    potential_gpp=np.array([55]),
    crown_area=cohort_allometry.crown_area,
    par_ext=cohorts.cohorts.par_ext,
    lai=cohorts.cohorts.lai,
)
print(whole_crown_gpp)
```

Those realised stem GPP values can then be provided to the `StemAllocation` class:

```{code-cell} ipython3
cohort_allocation = StemAllocation(
    cohorts=cohorts,
    allometry=cohort_allometry,
    whole_crown_gpp=whole_crown_gpp.to_numpy(),
)
cohort_allocation
```

The {meth}`~pyrealm.demography_two.tmodel.StemAllocation.to_dataframe` method can be
used to export data for exploration.

```{code-cell} ipython3
cohort_allocation.to_dataframe().transpose()
```

### Allocation profiles

As the `StemAllometry`, the `StemAllocation` class can be used to generate a profile of
the allocation predictions for different estimates of potential GPP. The `profile=True`
option is used to indicate that - instead of providing a single GPP estimate for each
cohort - you want predictions at each GPP estimate for each cohort.

```{code-cell} ipython3
# Calculate the stem GPP from potential GPP following the Li et al model
whole_crown_gpp_profile = np.arange(30, 100)

# Calculate the T Model allocation of those GPP values
allocation_profile = StemAllocation(
    cohorts=cohorts,
    allometry=cohort_allometry,
    whole_crown_gpp=whole_crown_gpp_profile,
    profile=True,
)
allocation_profile
```

```{code-cell} ipython3
fig, axes = plt.subplots(ncols=2, nrows=5, sharex=True, figsize=(10, 12))

plot_details = [
    ("sapwood_respiration", "sapwood_respiration"),
    ("foliage_respiration", "foliage_respiration"),
    ("fine_root_respiration", "fine_root_respiration"),
    ("npp", "npp"),
    ("foliage_turnover", "foliage_turnover"),
    ("fine_root_turnover", "fine_root_turnover"),
    ("delta_dbh", "delta_dbh"),
    ("delta_stem_mass", "delta_stem_mass"),
    ("delta_foliage_mass", "delta_foliage_mass"),
]

axes = axes.flatten()

for ax, (var, ylab) in zip(axes, plot_details):
    ax.plot(whole_crown_gpp_profile, getattr(allocation_profile, var), label=flora.name)
    ax.set_xlabel("GPP (m)")
    ax.set_ylabel(ylab)

    if var == "whole_crown_gpp":
        ax.legend(frameon=False)

# Delete unused panel in 5 x 2 grid
fig.delaxes(axes[-1])
```

The {meth}`~pyrealm.demography_two.StemAllocation.to_dataframe()` method can still be
used, but the values are stacked into columns identified by pairings of cohort ID and
DBH.

```{code-cell} ipython3
allocation_profile.to_dataframe().head(6)[
    ["cohort_ids", "whole_crown_gpp", "npp", "delta_dbh"]
]
```
