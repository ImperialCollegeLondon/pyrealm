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
---

# Plant Functional Types and the Flora object

:::{admonition} Warning

This area of `pyrealm` is in active development and this notebook currently contains
notes and initial demonstration code.

:::

```{code-cell}
from matplotlib import pyplot as plt
import numpy as np

from pyrealm.demography.flora import PlantFunctionalType, Flora
```

The code below creates a simple `Flora` object containing 3 plant functional types with
different maximum stem heights.

```{code-cell}
flora = Flora(
    [
        PlantFunctionalType(name="short", h_max=10),
        PlantFunctionalType(name="medium", h_max=20),
        PlantFunctionalType(name="tall", h_max=30),
    ]
)

flora
```

We can visualise how the stem size, canopy size and various masses of a plant functional
type change with stem diameter by using the `Flora.get_allometries` method. This takes
an array of values for diameter at breast height (metres) and returns a dictionary
containing the predictions of the T Model for:

* Stem height ('stem_height', m)
* Crown area ('crown_area', m2)
* Crown fraction ('crown_fraction', -)
* Stem mass ('stem_mass', kg)
* Foliage mass ('foliage_mass', kg)
* Sapwood mass ('sapwood_mass', kg)

The returned values in the dictionary are 2 dimensional arrays with each DBH value as a
row and each PFT as a column. This makes them convenient to plot using `matplotlib`.

```{code-cell}
dbh = np.arange(0, 1.6, 0.01)[:, None]
allometries = flora.get_allometries(dbh=dbh)
```

The code below shows how to use the returned allometries to generate a plot of the
scaling relationships across all of the PFTs in a `Flora` instance.

```{code-cell}
fig, axes = plt.subplots(ncols=2, nrows=3, sharex=True, figsize=(10, 8))

plot_details = [
    ("stem_height", "Stem height (m)"),
    ("crown_area", "Crown area (m2)"),
    ("crown_fraction", "Crown fraction (-)"),
    ("stem_mass", "Stem mass (kg)"),
    ("foliage_mass", "Foliage mass (kg)"),
    ("sapwood_mass", "Sapwood mass (kg)"),
]

for ax, (var, ylab) in zip(axes.flatten(), plot_details):
    ax.plot(dbh, allometries[var], label=flora.keys())
    ax.set_xlabel("Diameter at breast height (m)")
    ax.set_ylabel(ylab)

    if var == "sapwood_mass":
        ax.legend(frameon=False)
```

```{code-cell}
potential_gpp = np.repeat(55, dbh.size)[:, None]
allocation = flora.get_allocation(dbh=dbh, potential_gpp=potential_gpp)
```

```{code-cell}
fig, axes = plt.subplots(ncols=2, nrows=5, sharex=True, figsize=(10, 12))

plot_details = [
    ("whole_crown_gpp", "whole_crown_gpp"),
    ("sapwood_respiration", "sapwood_respiration"),
    ("foliar_respiration", "foliar_respiration"),
    ("fine_root_respiration", "fine_root_respiration"),
    ("npp", "npp"),
    ("turnover", "turnover"),
    ("delta_dbh", "delta_dbh"),
    ("delta_stem_mass", "delta_stem_mass"),
    ("delta_foliage_mass", "delta_foliage_mass"),
]

axes = axes.flatten()

for ax, (var, ylab) in zip(axes, plot_details):
    ax.plot(dbh, allocation[var], label=flora.keys())
    ax.set_xlabel("Diameter at breast height (m)")
    ax.set_ylabel(ylab)

    if var == "whole_crown_gpp":
        ax.legend(frameon=False)

# Delete unused panel in 5 x 2 grid
fig.delaxes(axes[-1])
```
