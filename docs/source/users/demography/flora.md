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

# Plant Functional Types and Traits

:::{admonition} Warning

This area of `pyrealm` is in active development and this notebook currently contains
notes and initial demonstration code.

:::

This page introduces the main components of the {mod}`~pyrealm.demography` module that
describe plant functional types (PFTs) and their traits.

```{code-cell}
from matplotlib import pyplot as plt
import numpy as np

from pyrealm.demography.flora import PlantFunctionalType, Flora

from pyrealm.demography.t_model_functions import StemAllocation, StemAllometry
```

## Plant traits

The table below shows the traits used to define the behaviour of different PFTs in
demographic simulations. These traits mostly consist of the parameters defined in the T
Model {cite}`Li:2014bc` to govern the allometric scaling and carbon allocation of trees,
but also include parameters for crown shape that follow the implementation developed in
the PlantFATE model {cite}`joshi:2022a`.

:::{list-table}
:widths: 10 30
:header-rows: 1

* * Trait name
  * Description
* * `a_hd`
  * Initial slope of height-diameter relationship ($a$, -)
* * `ca_ratio`
  * Initial ratio of crown area to stem cross-sectional area ($c$, -)
* * `h_max`
  * Maximum tree height ($H_m$, m)
* * `rho_s`
  * Sapwood density ($\rho_s$, kg Cm-3)
* * `lai`
  * Leaf area index within the crown ($L$,  -)
* * `sla`
  * Specific leaf area ($\sigma$,  m2 kg-1 C)
* * `tau_f`
  * Foliage turnover time ($\tau_f$,years)
* * `tau_r`
  * Fine-root turnover time ($\tau_r$,  years)
* * `par_ext`
  * Extinction coefficient of photosynthetically active radiation (PAR) ($k$, -)
* * `yld`
  * Yield factor ($y$,  -)
* * `zeta`
  * Ratio of fine-root mass to foliage area ($\zeta$, kg C m-2)
* * `resp_r`
  * Fine-root specific respiration rate ($r_r$, year-1)
* * `resp_s`
  * Sapwood-specific respiration rate ($r_s$,  year-1)
* * `resp_f`
  * Foliage maintenance respiration fraction ($r_f$,  -)
* * `m`
  * Crown shape parameter ($m$, -)
* * `n`
  * Crown shape parameter ($n$, -)
* * `f_g`
  * Crown gap fraction ($f_g$, -)
* * `q_m`
  * Scaling factor to derive maximum crown radius from crown area.
* * `z_max_prop`
  * Proportion of stem height at which maximum crown radius is found.
:::

+++

## Plant Functional Types

The {class}`~pyrealm.demography.flora.PlantFunctionalType` class is used define a PFT
with a given name, along with the trait values associated with the PFT. By default,
values for each trait are taken from Table 1 of {cite}`Li:2014bc`, but these can be
adjusted for different PFTs. The code below contains three examples that just differ in
their maximum height.

Note that the `q_m` and `z_max_prop` traits are calculated from the `m` and `n` traits
and cannot be set directly.

```{code-cell}
short_pft = PlantFunctionalType(name="short", h_max=10)
medium_pft = PlantFunctionalType(name="medium", h_max=20)
tall_pft = PlantFunctionalType(name="tall", h_max=30)
```

The traits values set for a PFT instance can then be shown:

```{code-cell}
short_pft
```

## The Flora class

The {class}`~pyrealm.demography.flora.Flora` class is used to collect a list of PFTs
that will be used in a demographic simulation. It can be created directly by providing
the list of {class}`~pyrealm.demography.flora.PlantFunctionalType` instances. The only
requirement is that each PFT instance uses a different name.

```{code-cell}
flora = Flora([short_pft, medium_pft, tall_pft])

flora
```

You can also create `Flora` instances using PFT data stored TOML, JSON and CSV file formats.

## Stem allometry

We can visualise how the stem size, canopy size and various masses of PFTs change with
stem diameter by using the {class}`~pyrealm.demography.t_model_functions.StemAllometry`
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

The DBH input can be a scalar array or a one dimensional array providing a single value
for each PFT. This then calculates a single estimate for a specific stem size.

```{code-cell}
# Calculate a single prediction
single_allometry = StemAllometry(stem_traits=flora, at_dbh=np.array([0.1, 0.1, 0.1]))
single_allometry.stem_height
```

However, the DBH values can also be a column array (an `N` x 1 array). In this case, the
predictions are made at each DBH value for each PFT and the allometry attributes with
predictions arranged with each PFT as a column and each DBH prediction as a row. This
makes them convenient to plot using `matplotlib`.

```{code-cell}
# Column array of DBH values from 0 to 1.6 metres
dbh_col = np.arange(0, 1.6, 0.01)[:, None]
# Get the predictions
allometries = StemAllometry(stem_traits=flora, at_dbh=dbh_col)
```

The code below shows how to use the returned allometries to generate a plot of the
scaling relationships across all of the PFTs in a `Flora` instance.

```{code-cell}
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
    ax.plot(dbh_col, getattr(allometries, var), label=flora.name)
    ax.set_xlabel("Diameter at breast height (m)")
    ax.set_ylabel(ylab)

    if var == "sapwood_mass":
        ax.legend(frameon=False)
```

```{code-cell}
potential_gpp = np.repeat(55, dbh_col.size)[:, None]
print(potential_gpp)


allocation = StemAllocation(
    stem_traits=flora, stem_allometry=single_allometry, at_potential_gpp=potential_gpp
)
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
