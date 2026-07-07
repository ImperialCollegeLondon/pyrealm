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
  version: 3.14.3
  mimetype: text/x-python
  codemirror_mode:
    name: ipython
    version: 3
  pygments_lexer: ipython3
  nbconvert_exporter: python
  file_extension: .py
---

# Allometry in the T Model

```{admonition} Run this notebook
:class: hint

* Read the guide on setting up your computer to [run Jupyter
  notebooks](../getting_started.md)
* Download {nb-download}`this notebook<./allometry.ipynb>` as a Jupyter notebook.

```

```{code-cell} ipython3
import warnings

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from pyrealm.core.experimental import ExperimentalFeatureWarning
from pyrealm.demography.flora import Flora
from pyrealm.demography.cohorts import create_cohorts, cohort_id_generator
from pyrealm.demography.tmodel import StemAllometry

warnings.filterwarnings(
    "ignore",
    category=ExperimentalFeatureWarning,
)
```

To generate allometric predictions under the T Model, we need to define a set of
[cohorts](./flora#plant-cohorts):

```{code-cell} ipython3
# Create a flora with 3 PFTs with different maximum heights
flora = Flora(name=["short", "medium", "tall"], h_max=[10, 20, 30])

# Create an id generator.
cid_generator = cohort_id_generator(mode="str")

# Create the cohorts
cohorts = create_cohorts(
    flora=flora,
    cid_generator=cid_generator,
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

The {meth}`~pyrealm.demography.tmodel.StemAllometry` class provides the
{meth}`~pyrealm.demography.StemAllometry.to_dataframe()` method to export the stem
data for data exploration. The `StemAllometry` data retains the unique cohort ids and
DBH from the original cohort data.

```{code-cell} ipython3
cohort_allometry.to_dataframe().transpose()
```

## Allometry profiles

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

The {meth}`~pyrealm.demography.StemAllometry.to_dataframe()` method can still be
used, but the values are stacked into columns identified by pairings of cohort ID and
DBH.

```{code-cell} ipython3
allometry_profiles.to_dataframe().head(6)[
    ["cohort_ids", "dbh", "stem_height", "crown_area", "crown_fraction"]
]
```
