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

# Canopy model

:::{admonition} Warning

This area of `pyrealm` is in active development and this notebook currently contains
notes and initial demonstration code.

:::

The canopy model uses the perfect plasticity approximation (PPA) {cite}`purves:2008a`,
which assumes that plants are always able to plastically arrange their crown within the
broader canopy of the community to maximise their crown size and fill the available
space $A$. When the area $A$ is filled, a new lower canopy layer is formed until all
of the individual crown area has been distributed across within the canopy.

The key variables in calculating the canopy model are the crown projected area $A_p$
and leaf projected projected area $\tilde{A}_{cp}(z)$, which are calculated for a stem
of a given size using the  [crown model](./crown.md).

```{code-cell}
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from pyrealm.demography.flora import PlantFunctionalType, Flora
from pyrealm.demography.community import Community
from pyrealm.demography.crown import CrownProfile
from pyrealm.demography.canopy import Canopy
```

## Canopy closure and canopy gap fraction

A simple method for finding the first canopy closure height is to find a height $z^*_1$
at which the sum of crown projected area across all stems $N_s$ in a community equals $A$:

$$
\sum_1^{N_s}{ A_p(z^*_1)} = A
$$

However, the canopy model is modified by a community-level
**canopy gap fraction** ($f_G$) that captures the overall proportion of the canopy area
that is left unfilled by canopy. This gap fraction, capturing processes such as crown
shyness, describes the proportion of open sky visible from the forest floor. This
gives the following definition of the height of canopy layer closure ($z^*_l$) for a
given canopy layer $l = 1, ..., l_m$:

$$
\sum_1^{N_s}{ A_p(z^*_l)} = l A(1 - f_G)
$$

The set of heights $z^*$  can be found numerically by using a root solver to find
values of $z^*_l$ for $l = 1, ..., l_m$ that satisfy:

$$
\sum_1^{N_s}{ A_p(z^*_l)} - l A(1 - f_G) = 0
$$

The total number of layers $l_m$ in a canopy, where the final layer may not be fully
closed, can be found given the total crown area across stems as:

$$
l_m = \left\lceil \frac{\sum_1^{N_s}{A_c}}{ A(1 - f_G)}\right\rceil
$$

+++

## Implementation in `pyrealm`

The {class}`~pyrealm.demography.canopy.Canopy` class automatically finds the canopy
closure heights, given a {class}`~pyrealm.demography.community.Community` instance
and the required canopy gap fraction.

The code below creates a simple community and then fits the canopy model:

```{code-cell}
# Two PFTs
# - a shorter understory tree with a columnar canopy and no crown gaps
# - a taller canopy tree with a top heavy canopy and more crown gaps

short_pft = PlantFunctionalType(
    name="short", h_max=15, m=1.5, n=1.5, f_g=0, ca_ratio=380
)
tall_pft = PlantFunctionalType(name="tall", h_max=30, m=1.5, n=2, f_g=0.2, ca_ratio=500)

# Create the flora
flora = Flora([short_pft, tall_pft])

# Create a simply community with three cohorts
# - 15 saplings of the short PFT
# - 5 larger stems of the short PFT
# - 2 large stems of tall PFT

community = Community(
    flora=flora,
    cell_area=32,
    cell_id=1,
    cohort_dbh_values=np.array([0.02, 0.20, 0.5]),
    cohort_n_individuals=np.array([15, 5, 2]),
    cohort_pft_names=np.array(["short", "short", "tall"]),
)
```

We can then look at the expected allometries for the stems in each cohort:

```{code-cell}
print("H = ", community.stem_allometry.stem_height)
print("Ac = ", community.stem_allometry.crown_area)
```

We can now calculate the canopy model for the community:

```{code-cell}
canopy = Canopy(community=community, canopy_gap_fraction=2 / 32)
```

We can then look at three key properties of the canopy model: the layer closure
heights ($z^*_l$) and the projected crown areas and leaf areas at each of those
heights for each stem in the three cohorts.

There are four canopy layers, with the top two very close together because of the
large crown area in the two stems in the cohort of `tall` trees.

```{code-cell}
canopy.layer_heights
```

The `stem_crown_area` attribute then provides the crown area of each stem found in each
layer.

```{code-cell}
canopy.stem_crown_area
```

Given the canopy gap fraction, the available crown area per layer is 30 m2, so
the first two layers are taken up entirely by the two stems in the cohort of large
trees. We can confirm that the calculation is correct by calculating the total crown area
across the cohorts at each height:

```{code-cell}
np.sum(canopy.stem_crown_area * community.cohort_data["n_individuals"], axis=1)
```

Those are equal to the layer closure areas of 30, 60 and 90 m2 and the last layer does
not quite close. The slight numerical differences result from the precision of the root
solver for finding $z^*_l$ and this can be adjusted by using the `layer_tolerance`
argument to the `Canopy` class

The projected leaf area per stem is reported in the `stem_leaf_area` attribute. This is
identical to the projected crown area for the first two cohorts because the crown gap
fraction $f_g$ is zero for this PFT. The projected leaf area is however displaced
towards the ground in the last cohort, because the `tall` PFT has a large gap fraction.

```{code-cell}
canopy.stem_leaf_area
```

### Visualizing layer closure heights and areas

We can use the {class}`~pyrealm.demography.crown.CrownProfile` class to calculate a
community crown and leaf area profile across a range of height values. For each height,
we calculate the sum of the product of stem projected area and the number of
individuals in each cohort.

```{code-cell}
# Set of vertical height to calculate crown profiles
at_z = np.linspace(0, 26, num=261)[:, None]

# Calculate the crown profile for the stem for each cohort
crown_profiles = CrownProfile(
    stem_traits=community.stem_traits, stem_allometry=community.stem_allometry, z=at_z
)

# Calculate the total projected crown area across the community at each height
community_crown_area = np.nansum(
    crown_profiles.projected_crown_area * community.cohort_data["n_individuals"], axis=1
)
# Do the same for the projected leaf area
community_leaf_area = np.nansum(
    crown_profiles.projected_leaf_area * community.cohort_data["n_individuals"], axis=1
)
```

We can now plot community-wide $A_p(z)$ and $\tilde{A}_{cp}(z)$ profiles, and
superimpose the calculated $z^*_l$ values and the cumulative canopy area for each layer
to confirm that the calculated values coincide with the profile. Note here that the
total area at each closed layer height is omitting the community gap fraction.

```{code-cell}
fig, ax = plt.subplots(ncols=1)

# Calculate the crown area at which each canopy layer closes.
closure_areas = np.arange(1, canopy.n_layers + 1) * canopy.crown_area_per_layer

# Add lines showing the canopy closure heights and closure areas.
for val in canopy.layer_heights:
    ax.axhline(val, color="red", linewidth=0.5, zorder=0)

for val in closure_areas:
    ax.axvline(val, color="red", linewidth=0.5, zorder=0)

# Show the community projected crown area profile
ax.plot(community_crown_area, at_z, zorder=1, label="Crown area")
ax.plot(
    community_leaf_area,
    at_z,
    zorder=1,
    linestyle="--",
    color="black",
    linewidth=1,
    label="Leaf area",
)


# Add z* values on the righthand axis
ax_rhs = ax.twinx()
ax_rhs.set_ylim(ax.get_ylim())
z_star_labels = [
    f"$z^*_{l + 1} = {val:.2f}$"
    for l, val in enumerate(np.nditer(canopy.layer_heights))
]
ax_rhs.set_yticks(canopy.layer_heights.flatten())
ax_rhs.set_yticklabels(z_star_labels)

# Add cumulative canopy area at top
ax_top = ax.twiny()
ax_top.set_xlim(ax.get_xlim())
area_labels = [f"$A_{l + 1}$ = {z:.1f}" for l, z in enumerate(np.nditer(closure_areas))]
ax_top.set_xticks(closure_areas)
ax_top.set_xticklabels(area_labels)

ax.set_ylabel("Vertical height ($z$, m)")
ax.set_xlabel("Community-wide projected area (m2)")
ax.legend(frameon=False)
```

The projected area from individual stems to each canopy layer can then be calculated at
$z^*_l$ and hence the projected area of canopy **within each layer**.

+++

### Light transmission through the canopy

Now we can use the leaf area by layer and the Beer-Lambert equation to calculate light
attenuation through the canopy layers.

$f_{abs} = 1 - e ^ {-kL}$,

where $k$ is the light extinction coefficient ($k$) and $L$ is the leaf area index
(LAI). The LAI can be calculated for each stem and layer:

```{code-cell}
# LAI = Acp_within_layer / canopy_area
# print(LAI)
```

This can be used to calculate the LAI of individual stems but also the LAI of each layer
in the canopy:

```{code-cell}
# LAI_stem = LAI.sum(axis=0)
# LAI_layer = LAI.sum(axis=1)

# print("LAI stem = ", LAI_stem)
# print("LAI layer = ", LAI_layer)
```

The layer LAI values can now be used to calculate the light transmission of each layer and
hence the cumulative light extinction profile through the canopy.

```{code-cell}
# f_abs = 1 - np.exp(-pft.traits.par_ext * LAI_layer)
# ext = np.cumprod(f_abs)

# print("f_abs = ", f_abs)
# print("extinction = ", ext)
```

One issue that needs to be resolved is that the T Model implementation in `pyrealm`
follows the original implementation of the T Model in having LAI as a fixed trait of
a given plant functional type, so is constant for all stems of that PFT.

```{code-cell}
# print("f_abs = ", (1 - np.exp(-pft.traits.par_ext * pft.traits.lai)))
```

## Things to worry about later

Herbivory - leaf fall (transmission increases, truncate at 0, replace from NSCs) vs leaf
turnover (transmission constant, increased GPP penalty)

Leaf area dynamics in PlantFATE - acclimation to herbivory and transitory decreases in
transimission, need non-structural carbohydrates  to recover from total defoliation.

Leaf economics.
