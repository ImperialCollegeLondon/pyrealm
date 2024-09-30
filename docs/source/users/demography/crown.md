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

# The tree crown model

:::{admonition} Warning

This area of `pyrealm` is in active development and this notebook currently contains
notes and initial demonstration code.

:::

```{code-cell}
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from pyrealm.demography.flora import (
    calculate_crown_q_m,
    calculate_crown_z_max_proportion,
    PlantFunctionalType,
    Flora,
)

from pyrealm.demography.t_model_functions import (
    calculate_dbh_from_height,
    StemAllometry,
)

from pyrealm.demography.crown import (
    CrownProfile,
)
```

The {mod}`pyrealm.demography` module uses three-dimensional model of crown shape to
define the vertical distribution of leaf area. This is driven by four parameters within
the {class}`~pyrealm.demography.flora.PlantFunctionalType`:

* The `m` and `n` parameters that define the vertical shape of the crown profile.
* The `ca_ratio` parameters that defines the size of the crown relative to the stem size.
* The `f_g` parameter that define the crown gap fraction and sets the vertical
  distribution of leaves within the crown.

The code below provides a demonstration of the impacts of each parameter and shows the
use of `pyrealm` tools to visualise the crown model for individual stems.

## Crown shape parameters

The `m` and `n` parameters define the vertical profile of the crown but are also define
two  further parameters:

* `q_m`, which is used to scale the size of the crown radius to match the expected crown
  area, given the crown shape.
* `z_max_prop`, which sets the height on the stem at which the maximum crown radius is found.

The code below calculates `q_m` and `z_max_prop` for combinations of `m` and `n` and
then plots the resulting values.

```{code-cell}
m = n = np.arange(1.0, 5, 0.1)
q_m = calculate_crown_q_m(m=m, n=n[:, None])
z_max_prop = calculate_crown_z_max_proportion(m=m, n=n[:, None])
```

```{code-cell}
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10.9, 4))

# Plot q_m as a function of m and n
cntr_set1 = ax1.contourf(m, n, q_m, levels=10)
fig.colorbar(cntr_set1, ax=ax1, label="q_m")
ax1.set_xlabel("m")
ax1.set_ylabel("n")
ax1.set_aspect("equal")

# Plot z_max_prop as a function of m and n
cntr_set2 = ax2.contourf(m, n, z_max_prop, levels=10)
fig.colorbar(cntr_set2, ax=ax2, label="z_max_prop")
ax2.set_xlabel("m")
ax2.set_ylabel("n")
ax2.set_aspect("equal")
```

## Plotting crown profiles

The examples below show the calculation of crown profiles for a set of plant functional
types (PFTs) with differing crown trait values. We first need to create a
:class:`~pyrealm.demography.flora.Flora` object that defines those PFTs.

```{code-cell}
flora = Flora(
    [
        PlantFunctionalType(name="short", h_max=20, m=1.5, n=1.5, f_g=0, ca_ratio=20),
        PlantFunctionalType(
            name="medium", h_max=20, m=1.5, n=4, f_g=0.05, ca_ratio=500
        ),
        PlantFunctionalType(name="tall", h_max=20, m=4, n=1.5, f_g=0.2, ca_ratio=2000),
    ]
)

flora
```

```{code-cell}
pd.DataFrame({k: getattr(flora, k) for k in flora.trait_attrs})
```

We then also need to specify the size of a stem of each PFT, along with the resulting
allometric predictions from the T Model.

```{code-cell}
# Generate the expected stem allometries at a single height for each PFT
stem_height = np.array([18, 18, 18])
stem_dbh = calculate_dbh_from_height(
    a_hd=flora.a_hd, h_max=flora.h_max, stem_height=stem_height
)
stem_dbh
```

```{code-cell}
allometry = StemAllometry(stem_traits=flora, at_dbh=stem_dbh)
```

We can use {mod}`pandas` to visualise those allometric predictions.

```{code-cell}
pd.DataFrame({k: getattr(allometry, k) for k in allometry.allometry_attrs})
```

We can now use the {class}`~pyrealm.demography.crown.CrownProfile` class to
calculate the crown profile for each stem for a set of vertical heights. The heights
need to be defined as a column array, that is with a shape `(N, 1)`, in order to show
that we want the crown variables to be calculated at each height for each PFT.

```{code-cell}
# Create a set of vertical heights as a column array.
z = np.linspace(0, 18.0, num=181)[:, None]

# Calculate the crown profile across those heights for each PFT
crown_profiles = CrownProfile(stem_traits=flora, stem_allometry=allometry, z=z)
```

The `crown_profiles` object is a dictionary containing values for four crown profile
variables.

* The relative crown radius: this value is driven purely by `m`, `n` and the stem height
  and defines the vertical profile of the crown.
* The crown radius: this is simply a rescaling of the relative radius so that the
  maximum crown radius matches the expected crown area predicted by the allometry of the
  T Model.
* The projected crown area: this value shows how crown area accumulates from the top of
  the crown to the ground.
* The projected leaf area: this value shows how _leaf_ area accumulates from the top of
  the crown to the ground. The difference from the crown area is driven by the crown gap
  fraction for a given PFT.

```{code-cell}
crown_profiles
```

### Crown radius values

The relative crown radius values are arbitrary - they simply define the shape of the
vertical profile. For the example PFTs in the `Flora` object, the maximum relative
radius values on each stem are:

```{code-cell}
max_relative_crown_radius = crown_profiles.relative_crown_radius.max(axis=0)
print(max_relative_crown_radius)
```

However the scaled maximum radius values match the expected crown area in the allometry
table above

```{code-cell}
max_crown_radius = crown_profiles.crown_radius.max(axis=0)
print(max_crown_radius)
print(max_crown_radius**2 * np.pi)
```

The code below generates a plot of the vertical shape profiles of the crowns for each
stem. For each stem:

* the dashed line shows how the relative crown radius varies with height,
* the solid line shows the actual crown radius profile with height, and
* the dotted line shows the height at which the maximum crown radius is found.

```{code-cell}
fig, ax = plt.subplots(ncols=1)

# Find the maximum of the actual and relative maximum crown widths
stem_max_width = np.maximum(max_crown_radius, max_relative_crown_radius)

for pft_idx, offset, colour in zip((0, 1, 2), (0, 5, 12), ("r", "g", "b")):

    # Plot relative radius either side of offset
    stem_qz = crown_profiles.relative_crown_radius[:, pft_idx]
    ax.plot(stem_qz + offset, z, color=colour, linestyle="--", linewidth=1)
    ax.plot(-stem_qz + offset, z, color=colour, linestyle="--", linewidth=1)

    # Plot actual crown radius either side of offset
    stem_rz = crown_profiles.crown_radius[:, pft_idx]
    ax.plot(stem_rz + offset, z, color=colour)
    ax.plot(-stem_rz + offset, z, color=colour)

    # Add the height of maximum crown radius
    stem_rz_max = stem_max_width[pft_idx]

    ax.plot(
        [offset - stem_rz_max, offset + stem_rz_max],
        [allometry.crown_z_max[pft_idx]] * 2,
        color=colour,
        linewidth=1,
        linestyle=":",
    )

ax.set_aspect(aspect=1)
```

### Projected crown and leaf areas

We can use the crown profile to generate projected area plots, but it is much easier to
compare the effects when comparing stems with similar crown areas. The code below
generates these new predictions for a new set of PFTs.

```{code-cell}
flora2 = Flora(
    [
        PlantFunctionalType(name="short", h_max=20, m=1.5, n=1.5, f_g=0, ca_ratio=380),
        PlantFunctionalType(
            name="medium", h_max=20, m=1.5, n=4, f_g=0.05, ca_ratio=400
        ),
        PlantFunctionalType(name="tall", h_max=20, m=4, n=1.5, f_g=0.2, ca_ratio=420),
    ]
)

allometry2 = StemAllometry(stem_traits=flora2, at_dbh=stem_dbh)


# Calculate the crown profile across those heights for each PFT
crown_profiles2 = CrownProfile(stem_traits=flora2, stem_allometry=allometry2, z=z)
```

The plot below shows how projected crown area (solid lines) and leaf area (dashed lines)
change with height along the stem.

* The projected crown area increases from zero at the top of the crown until it reaches
  the maximum crown radius, at which point it remains constant down to ground level. The
  total crown areas differs for each stem because of the slightly different values used
  for the crown area ratio.

* The projected leaf area is very similar but leaf area is displaced towards the ground
  because of the crown gap fraction (`f_g`). Where `f_g = 0` (the red line), the two
  lines are identical, but as `f_g` increases, more of the leaf area is displaced down
  within the crown.

```{code-cell}
fig, ax = plt.subplots(ncols=1)

for pft_idx, offset, colour in zip((0, 1, 2), (0, 5, 10), ("r", "g", "b")):

    ax.plot(crown_profiles2.projected_crown_area[:, pft_idx], z, color=colour)
    ax.plot(
        crown_profiles2.projected_leaf_area[:, pft_idx],
        z,
        color=colour,
        linestyle="--",
    )
```

```{code-cell}

```
