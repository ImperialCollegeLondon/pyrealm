---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
language_info:
  codemirror_mode:
    name: ipython
    version: 3
  file_extension: .py
  mimetype: text/x-python
  name: python
  nbconvert_exporter: python
  pygments_lexer: ipython3
  version: 3.11.9
settings:
  output_matplotlib_strings: remove
---

# The canopy model

:::{admonition} Warning

This area of `pyrealm` is in active development and this notebook currently contains
notes and initial demonstration code.

:::

```{code-cell} ipython3
:tags: [hide-input]

import warnings

from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd

from pyrealm.demography.flora import PlantFunctionalType, Flora
from pyrealm.demography.community import Cohorts, Community
from pyrealm.demography.crown import CrownProfile, get_crown_xy
from pyrealm.demography.canopy import Canopy
from pyrealm.demography.tmodel import StemAllometry
from pyrealm.core.experimental import ExperimentalFeatureWarning

warnings.filterwarnings(
    "ignore",
    category=ExperimentalFeatureWarning,
)

np.set_printoptions(precision=2)
```

The `canopy` module in `pyrealm` is used to calculate a vertically structured model of
leaf distribution for a [plant community](./community.md). The purpose of the `canopy`
module is two-fold:

1. to calculate the vertical distribution of crown and leaf area, including the
   partitioning of canopy into discrete vertical layers, and
2. to estimate light capture through canopy layers.

This page describes the model of the vertical canopy structure and the [light capture
model](./light_capture.md) is presented separately.

## Vertically structured canopies

The simplest "big leaf" model of a canopy, represents the entire crown crown ($A_c$) of
an individual tree as a single flat disc at the top of the stem. However, in `pyrealm`
the canopy model uses the crown shape described in the [crown model](./crown.md) to
model how leaf area accumulates vertically through a community.

### Community definition

To recap, a [plant community](./community.md) consists of a number of individual stems
that are growing together in a location. The community structure groups individuals
together into cohorts that are defined as:

* a number of individuals,
* with identical diameter at breast height (DBH, $D$),
* of the same [plant functional type (PFT)](./flora.md),
* and hence have the same [stem allometry](./t_model.md) and [crown model](./crown.md).

However, for the purposes of the canopy model, it is much simpler to consider the
community as a collection of individual stems, some of which just happen to share
identical properties.

### Community projected crown area

Each individual stem in a community has an projected crown area that describes how the
crown area of that stem accumulates with height ($z$) from the top of the tree to the
ground. The form of that curve varies with the size and PFT of the stem. If we have a
community with $N_s$ individuals, then each stem $i = 1, ..., N_s$ has a projected crown
area function $A_{p}(z)_i$.

If we take the sum across all individuals of this function at a height $z$, we get the
total community projected crown area across all individual stems at that same height:

$$
C_p(z) = \sum_{j=1}^{N_s}{A_{p}(z)_i}
$$

To demonstrate this, the code below creates a plant community: there are 12 individual
stems, grouped into 3 cohorts with different stem sizes and crown shapes.

```{code-cell} ipython3
# Define PFTs
short_pft = PlantFunctionalType(
    name="short", h_max=15, m=1.5, n=1.5, f_g=0, ca_ratio=380
)
tall_pft = PlantFunctionalType(
    name="tall", h_max=30, m=1.5, n=2, par_ext=0.6, f_g=0, ca_ratio=500
)

# Create the flora
flora = Flora([short_pft, tall_pft])

# Define a simply community with three cohorts
community = Community(
    flora=flora,
    cell_area=32,
    cell_id=1,
    cohorts=Cohorts(
        dbh_values=np.array([0.1, 0.20, 0.5]),
        n_individuals=np.array([7, 3, 2]),
        pft_names=np.array(["short", "short", "tall"]),
    ),
)
```

The total crown area of the individuals in each of the three cohorts is shown below:

```{code-cell} ipython3
community.stem_allometry.crown_area
```

The total crown area across the community is then the sum of those crown areas across
all individuals:

```{code-cell} ipython3
total_crown_area = np.sum(
    community.stem_allometry.crown_area * community.cohorts.n_individuals, keepdims=True
)
total_crown_area
```

So, the maximum projected community crown area is around 120 m2. The plot below shows
how $C_p(z)$ changes with height for this community from zero at the top of the canopy
to around 120 m2 at ground level. The horizontal dashed lines show the stem heights of
the individuals, where the crown area of each individual starts contributing to the
community wide projected crown area.

```{code-cell} ipython3
:tags: [hide-input]

# Calculate the crown profiles of the individuals across vertical heights
hghts = np.linspace(community.stem_allometry.stem_height.max(), 0, num=101)[:, None]
crown_profiles = CrownProfile(community.stem_traits, community.stem_allometry, z=hghts)

# Calculate the cumulative crown area across individuals at each height
crown_area = np.nansum(
    crown_profiles.projected_crown_area * community.cohorts.n_individuals,
    axis=1,
)

# Extract the crown profiles as XY arrays for plotting
profiles = get_crown_xy(
    crown_profile=crown_profiles,
    stem_allometry=community.stem_allometry,
    attr="crown_radius",
    as_xy=True,
)


# Helper function to add vertical or horizontal lines to plots
def add_hvlines(at, axes, vertical=True):

    # Line styling
    style = dict(color="black", linewidth=0.5, linestyle="--")

    # Add lines
    if vertical:
        axes.vlines(at, ymin=0, ymax=1, transform=axes.get_xaxis_transform(), **style)
    else:
        axes.hlines(at, xmin=0, xmax=1, transform=axes.get_yaxis_transform(), **style)


def add_second_axis(at, axes, fmt, xaxis=True):

    # Get labels
    labels = [fmt.format(val=val, idx=idx + 1) for idx, val in enumerate(at)]

    if xaxis:
        secax = axes.secondary_xaxis("top")
        secax.set_xticks(at, labels=labels)
    else:
        secax = axes.secondary_yaxis("right")
        secax.set_yticks(at, labels=labels)


# Create a side by side plot of the canopies and the cumulative community crown area
fig, (ax1, ax2) = plt.subplots(
    ncols=2, sharey=True, width_ratios=[1, 2], figsize=(8, 4)
)

for idx, crown in enumerate(profiles):

    # Get spaced but slightly randomized stem locations
    n_stems = community.cohorts.n_individuals[idx]
    stem_locations = np.linspace(0, 10, num=n_stems) + np.random.normal(size=n_stems)

    # Plot the crown model for each stem
    for stem_loc in stem_locations:
        ax1.add_patch(Polygon(crown + np.array([stem_loc, 0]), color="#00550055"))

ax1.autoscale_view()
add_hvlines(community.stem_allometry.stem_height, ax1, False)
ax1.set_ylabel("Height (m)")

ax2.plot(crown_area, hghts)
add_hvlines(community.stem_allometry.stem_height, ax2, False)
add_hvlines([0, total_crown_area], ax2, True)
_ = ax2.set_xlabel("Community projected crown area ($C_{p}(z), m^2$)")

plt.tight_layout()
```

If the community is growing in an area greater than 120 m2, then the individuals can
avoid overlapping any of their crown area. However, this is not possible if the
community is growing in a smaller area, and some crown area must overlie other parts
of the community, creating a vertical structure.

The canopy module implements the **perfect plasticity approximation (PPA)** model
{cite}`purves:2008a` to generate this structure. The PPA model assumes that all the
individuals within the community are able to plastically arrange their crown at each height
within the broader canopy of the community to fill available space. When the available
space ($A$) is filled, crown area lower in the canopy forms another layer until another
area $A$ is filled and this repeats down to the ground.

To fit this model, we need to find the heights $z^*_l$ at which the projected community
crown area occupies multiples of the available area $A$. This gives the heights at which
each successive canopy layer closes for canopy layers $l = 1, ..., l_m$. This is given as
the equality:

$$
C_p(z^*_l) = l A
$$

An extra detail is that the canopy model includes a community-level canopy gap
fraction ($f_G$) that captures the overall proportion of the canopy area that is left
unfilled by canopy. This gap fraction, capturing processes such as crown shyness,
describes the proportion of open sky visible from the forest floor. This term reduces
the amount of $A$ that can be occupied by crown area before a layer closes:

$$
C_p(z^*_l) = l A (1- f_G)
$$

The values of $z^*_l$ cannot be found analytically, so we use numerical methods to find
values of $z^*_l$ for $l = 1,..., l_m$ that satisfy:

$$
C_p(z^*_l) - l A(1 - f_G) = 0
$$

The total number of layers $l_m$ in a canopy, where the final layer may not be fully
closed, can be found given the total crown area across stems as:

$$
l_m = \left\lceil \frac{\sum_1^{N_s}{A_c}}{A(1 - f_G)}\right\rceil
$$

The community above has an available space $A=32\;m^2$. If we fit the PPA model to this
community we get the following heights:

```{code-cell} ipython3
# Fit the canopy model
canopy_ppa = Canopy(community=community, fit_ppa=True)
canopy_ppa.heights
```

We plot those layer heights below. The vertical dashed lines in the right hand plot show
the values $lA$ at which the cumulative space across layers forms closed layers: that is
$32 \times 1 = 32$, $32 \times 2 = 64$, etc. . The horizontal dashed lines then show
those calculated vertical heights. These are the canopy heights at which the vertical
lines intersect the cumulativecommunity projected crown area: where the total crown area
across the individuals in the community fills each available layer.

```{code-cell} ipython3
:tags: [hide-input]

# Create a plot of the canopy structure
fig, (ax1, ax2) = plt.subplots(
    ncols=2, sharey=True, width_ratios=[1, 2], figsize=(8, 4)
)

# Create a canopy plot
for idx, crown in enumerate(profiles):

    # Get spaced but slightly randomized stem locations
    n_stems = community.cohorts.n_individuals[idx]
    stem_locations = np.linspace(0, 10, num=n_stems) + np.random.normal(size=n_stems)

    # Plot the crown model for each stem
    for stem_loc in stem_locations:
        ax1.add_patch(Polygon(crown + np.array([stem_loc, 0]), color="#00550055"))

add_hvlines(canopy_ppa.heights, ax1, False)
ax1.set_ylabel("Height (m)")

# Plot community projected crown area
ax2.plot(crown_area, hghts)
ax2.set_xlabel("Cumulative crown area (m2)")

# Add lines to show PPA closure areas and heights
area_values = np.arange(1, 5) * 32
add_hvlines(canopy_ppa.heights, ax2, False)
add_hvlines(area_values, ax2, True)

# Add secondary axes to give values
add_second_axis(canopy_ppa.heights.squeeze(), ax2, "$z^*_{idx} = ${val:0.2f}", False)
add_second_axis(area_values, ax2, "$A_{idx} = ${val}", True)

plt.tight_layout()
```

We can look in detail at the how much crown area is in each layer from each individual.
The canopy model object contains a crown profile which records the projected crown
area at each height:

```{code-cell} ipython3
canopy_ppa.crown_profile.projected_crown_area
```

Those values are the projected *individual* crown areas: the accumulating crown area
from the top of the canopy down to the ground. We can take the differences between
vertical layers to give the actual amount of crown area in each layer for each
individual.

```{code-cell} ipython3
individual_crown_in_layer = np.diff(
    canopy_ppa.crown_profile.projected_crown_area, prepend=0, axis=0
)
individual_crown_in_layer
```

To bring those values back to the community model: each column represents a cohort, so
we can multiply those values by the number of individuals and then find the row sums
to show the amount of crown area in each layer. As expected, the crown area in each
layer matches the 32 m2 of available space, except for the last layer that is not
completely filled.

```{code-cell} ipython3
np.sum(individual_crown_in_layer * community.cohorts.n_individuals, axis=1)
```

### Crown and canopy gap fractions

The canopy and crown models are extended by providing two gap fractions:

* the crown gap fraction ($f_g$) is described in detail in the [crown
  model](./crown.md) and captures how an individual tree crown may contain holes that
  displace leaf area further down into the canopy.
* the canopy gap fraction ($f_G$) is described above and captures how the canopy across
  the whole community may leave space unfilled in the canopy leaving light gaps that
  reach down to the ground.

The code below alters the community and canopy model used above to include both crown
and canopy gap fractions.

```{code-cell} ipython3
# Define gappy PFTs
gappy_short_pft = PlantFunctionalType(
    name="short",
    h_max=15,
    m=1.5,
    n=1.5,
    f_g=0.1,
    ca_ratio=380,
)
gappy_tall_pft = PlantFunctionalType(
    name="tall", h_max=30, m=1.5, n=2, par_ext=0.6, f_g=0.1, ca_ratio=500
)

# Create the flora
gappy_flora = Flora([gappy_short_pft, gappy_tall_pft])

# Define community with three cohorts
gappy_community = Community(
    flora=gappy_flora,
    cell_area=32,
    cell_id=1,
    cohorts=Cohorts(
        dbh_values=np.array([0.1, 0.20, 0.5]),
        n_individuals=np.array([7, 3, 2]),
        pft_names=np.array(["short", "short", "tall"]),
    ),
)

# Calculate the canopy profile across vertical heights
gappy_canopy_ppa = Canopy(
    community=gappy_community, fit_ppa=True, canopy_gap_fraction=1 / 8
)
```

```{code-cell} ipython3
:tags: [hide-input]

# Calculate the crown profiles for the gappy community
gappy_crown_profiles = CrownProfile(
    gappy_community.stem_traits, gappy_community.stem_allometry, z=hghts
)

# Calculate the cumulative crown area across individuals for the gappy community
gappy_crown_area = np.nansum(
    gappy_crown_profiles.projected_crown_area * gappy_community.cohorts.n_individuals,
    axis=1,
)

# Calculate leaf areas for each community
leaf_area = np.nansum(
    crown_profiles.projected_leaf_area * community.cohorts.n_individuals,
    axis=1,
)

gappy_leaf_area = np.nansum(
    gappy_crown_profiles.projected_leaf_area * gappy_community.cohorts.n_individuals,
    axis=1,
)
```

The plots below show the cumulative projected leaf and crown areas across the
individuals in the two communities along with the PPA layer closure heights. The first
thing to note is that the projected crown area for the communities is identical: the
cohorts have the same crown shapes, stem heights and number of individuals. For the
non-gappy community, the projected leaf area and projected crown area are also
identical. However:

* The gappy community has a canopy gap fraction of 1/8, reducing the available total
  crown area within a single layer to 28 m2. The layer closure heights under the PPA
  model are displaced upwards and an extra closed canopy layer is needed to fit the
  total community crown area of ~120 m2 into the available area.

* The crown gap fractions in the gappy community are not zero and so the *projected leaf
  area* is displaced downwards within the canopy. Note that this does not affect the
  location of the canopy layer closure heights, but it *does* affect the location of
  leaf area for the purposes of [light capture](./light_capture.md).

```{code-cell} ipython3
:tags: [hide-input]

fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(10, 5))

# Plot projected community crown and leaf area for non gappy community
ax1.plot(crown_area, hghts)
ax1.plot(leaf_area, hghts, linestyle="--", color="red")

# Add PPA closure height and area lines and axes
add_hvlines(canopy_ppa.heights, ax1, False)
add_hvlines(area_values, ax1, True)
add_second_axis(canopy_ppa.heights.squeeze(), ax1, "$z^*_{idx} = ${val:0.2f}", False)
add_second_axis(area_values, ax1, "$A_{idx} = ${val}", True)

text_args = dict(x=0.95, y=0.95, ha="right", va="top", backgroundcolor="white")
ax1.text(s="Non-gappy community", transform=ax1.transAxes, **text_args)
ax1.set_xlabel("Projected community area (m2)")
ax1.set_ylabel("Height (m)")

# Gappy community
gappy_area_values = np.arange(1, 5) * 28

ax2.plot(gappy_crown_area, hghts, label="Crown area")
ax2.plot(gappy_leaf_area, hghts, linestyle="--", color="red", label="Leaf area")

# Add PPA closure height and area lines and axes
add_hvlines(gappy_canopy_ppa.heights.squeeze(), ax2, False)
add_hvlines(gappy_area_values, ax2, True)
add_second_axis(
    gappy_canopy_ppa.heights.squeeze(), ax2, "$z^*_{idx} = ${val:0.2f}", False
)
add_second_axis(gappy_area_values, ax2, "$A_{idx} = ${val}", True)

ax2.text(s="Gappy community", transform=ax2.transAxes, **text_args)
ax2.set_xlabel("Projected community area (m2)")
ax2.legend(framealpha=1.0)

plt.tight_layout()
```
