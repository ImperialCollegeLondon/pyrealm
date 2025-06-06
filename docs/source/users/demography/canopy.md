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
```

The `canopy` module in `pyrealm` is used to calculate a vertically structured model of
leaf distribution across all of cohorts within a [plant community](./community.md),
where a cohort is defined as:

* a number of individuals,
* with identical diameter at breast height (DBH, $D$),
* of the same [plant functional type (PFT)](./flora.md),
* that share the same [stem allometry](./t_model.md) and [crown model](./crown.md).

The purpose of the `canopy` module is two-fold:

1. to calculate the vertical distribution of crown and leaf area, including the
   partitioning of canopy into discrete vertical layers, and
2. to estimate light capture through canopy layers.

This page describes the first of the roles and [light capture](./light_capture.md)
is presented separately.

## Vertically structured canopies

The simplest "big leaf" model of a canopy, represents the entire area of the crown
($A_c$) as a single flat disc at the top of stem. However, the canopy model uses the
crown shape described in the [crown model](./crown.md) to model how leaf area
accumulates vertically through the cohorts of a community.

The canopy model can profile community crown area at given heights - useful to generate
detailed plots of vertical structure - but also implements a specific model that
partitions cohort crown area vertically into a series of closed canopy layers. The
partitioning follows the perfect plasticity approximation (PPA) model
{cite}`purves:2008a`, which assumes that plants are always able to plastically arrange
their crown within the broader canopy of the community to maximise their crown size and
fill the available space. The available space $A$ is simply the plot size that the
community is growing in - all crown area is constrained to occur within the plot.

If the total canopy area of all individuals within cohorts is less than $A$ then the
canopy model is complete - in this case, the predications are effectively the same as
the big-leaf model. However if the total crown area across the stems is greater than $A$
then the model finds heights at which successive layers fill up the available area,
giving closed canopy layers.

We can do this using the projected crown area ${A}_{p}(z)$ from the crown model, which
describes how crown area accumulates with height $z$ from the top of the stem to the
ground, given the crown shape of the tree. If we sum the projected crown area across all
stems in a community, we generate a prediction of the total community crown area at a
given height $z$:

$$
\sum_1^{N_s}{ A_p(z)}
$$

We then need to find the heights $z^*_l$ at which the total community crown area
occupies multiples of the available area $A$, giving the heights at which canopy
layers close for canopy layers $l = 1, ..., l_m$. This is given as the equality:

$$
\sum_1^{N_s}{ A_p(z^*_l)} = l A
$$

An extra detail is that the canopy model includes a community-level canopy gap
fraction ($f_G$) that captures the overall proportion of the canopy area that is left
unfilled by canopy. This gap fraction, capturing processes such as crown shyness,
describes the proportion of open sky visible from the forest floor. This term reduces
the amount of $A$ that can be occupied by crown area before a layer closes:

$$
\sum_1^{N_s}{ A_p(z^*_l)} = l A (1- f_G)
$$

The values of $z^*_l$ cannot be found analytically, so we use numerical methods to find
values of $z^*_l$ for $l = 1,..., l_m$ that satisfy:

$$
\sum_1^{N_s}{ A_p(z^*_l)} - l A(1 - f_G) = 0
$$

The total number of layers $l_m$ in a canopy, where the final layer may not be fully
closed, can be found given the total crown area across stems as:

$$
l_m = \left\lceil \frac{\sum_1^{N_s}{A_c}}{A(1 - f_G)}\right\rceil
$$

In this model, the crown area of a given stem is structured vertically across canopy
layers and the leaves in each layer are shaded by the layers above.

### A single tree example

The code below creates a community containing a single stem with a conveniently
straight sided profile and shows how the canopy model partitions this into layers
under the PPA.

```{code-cell} ipython3
# Define a community containing a single tree with a straight sided conifer profile
# also setting the crown gap fraction to zero so that crown area and leaf area are
# identical.
conifer = PlantFunctionalType(name="conifer", h_max=13, m=2, n=1, ca_ratio=2000, f_g=0)
flora = Flora([conifer])
community = Community(
    flora=flora,
    cell_area=32,
    cell_id=1,
    cohorts=Cohorts(
        dbh_values=np.array([0.5]),
        n_individuals=np.array([1]),
        pft_names=np.array(["conifer"]),
    ),
)

# Calculate a canopy model at fixed heights, to show the crown profile
# and a canopy model using the PPA to get closed layer heights
hghts = np.linspace(community.stem_allometry.stem_height.max(), 0, num=101)[:, None]
canopy = Canopy(community=community, layer_heights=hghts)
canopy_ppa = Canopy(community=community, fit_ppa=True)

# Calculate the cumulative crown area
cumulative_crown_area = np.nansum(
    canopy.crown_profile.projected_crown_area * community.cohorts.n_individuals,
    axis=1,
)
```

The `canopy_ppa.heights` attribute now contains the heights at which the PPA
layers close:

```{code-cell} ipython3
canopy_ppa.heights
```

Those three layer heights are shown on the plot below. The first two layers are
closed, with the stem leaf area within each layer equal to the community area of 32 m2.
The last layer is not completely filled.

```{code-cell} ipython3
:tags: [hide-input]

fig, (ax1, ax2) = plt.subplots(ncols=2, width_ratios=[1, 2], figsize=(8, 4))

# Extract the crown profiles as XY arrays for plotting
profiles = get_crown_xy(
    crown_profile=canopy.crown_profile,
    stem_allometry=community.stem_allometry,
    attr="crown_radius",
    as_xy=True,
)

ax1.add_patch(Polygon(profiles[0], color="#00550055"))
ax1.hlines(canopy_ppa.heights, -6, 6, color="black", linewidth=0.5)
ax1.set_ylabel("Height (m)")

area_values = 32 * np.arange(1, 4)
ax2.plot(cumulative_crown_area, hghts)
ax2.hlines(canopy_ppa.heights, 0, 32 * 3, color="black", linewidth=0.5)
ax2.vlines(area_values, 0, 13, color="black", linewidth=0.5)
ax2.set_xlabel("Cumulative crown area (m2)")

z_star_labels = [
    f"$z^*_{idx+1} = ${val:0.2f}"
    for idx, val in enumerate(canopy_ppa.heights.squeeze())
]
secaxy = ax2.secondary_yaxis("right")
secaxy.set_yticks(canopy_ppa.heights.squeeze(), labels=z_star_labels)

area_labels = [f"$A_{idx+1} = ${val}" for idx, val in enumerate(area_values)]
secaxx = ax2.secondary_xaxis("top")
secaxx.set_xticks(area_values, labels=area_labels)

plt.tight_layout()
```

### Multiple cohorts

The example below repeats the calculations above with a more complex community. There
are two plant functional types with different crown shapes and maximum height. The
community consists of:

* 7 saplings of the short PFT
* 3 larger stems of the short PFT
* 2 large stems of tall PFT

```{code-cell} ipython3
# Define PFTs
short_pft = PlantFunctionalType(
    name="short",
    h_max=15,
    m=1.5,
    n=1.5,
    f_g=0,
    ca_ratio=380,
)
tall_pft = PlantFunctionalType(
    name="tall", h_max=30, m=3, n=1.5, par_ext=0.6, f_g=0, ca_ratio=500
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

# Calculate the canopy profile across vertical heights
hghts = np.linspace(community.stem_allometry.stem_height.max(), 0, num=101)[:, None]
canopy = Canopy(community=community, layer_heights=hghts)
canopy_ppa = Canopy(community=community, fit_ppa=True)

# Calculate the cumulative crown area
cumulative_crown_area = np.nansum(
    canopy.crown_profile.projected_crown_area * community.cohorts.n_individuals,
    axis=1,
)
```

As with the simple example, we can show the crown shape of the individuals forming the
community and the accumulation of crown area with height.

```{code-cell} ipython3
:tags: [hide-input]

fig, (ax1, ax2) = plt.subplots(
    ncols=2, sharey=True, width_ratios=[1, 2], figsize=(8, 4)
)

# Extract the crown profiles as XY arrays for plotting
profiles = get_crown_xy(
    crown_profile=canopy.crown_profile,
    stem_allometry=community.stem_allometry,
    attr="crown_radius",
    as_xy=True,
)

for idx, crown in enumerate(profiles):

    # Get spaced but slightly randomized stem locations
    n_stems = community.cohorts.n_individuals[idx]
    stem_locations = np.linspace(0, 10, num=n_stems) + np.random.normal(size=n_stems)

    # Plot the crown model for each stem
    for stem_loc in stem_locations:
        ax1.add_patch(Polygon(crown + np.array([stem_loc, 0]), color="#00550055"))

ax1.autoscale_view()
ax1.hlines(canopy_ppa.heights, -5, 15, color="black", linewidth=0.5)
ax1.set_ylabel("Height (m)")

area_values = np.arange(1, 5) * 32
ax2.plot(cumulative_crown_area, hghts)
ax2.hlines(canopy_ppa.heights, 0, area_values.max(), color="black", linewidth=0.5)
ax2.vlines(area_values, 0, 25, color="black", linewidth=0.5)
_ = ax2.set_xlabel("Cumulative crown area (m2)")

# Add secondary ticks showing the layer heights and closure areas.
z_star_labels = [
    f"$z^*_{idx+1} = ${val:0.2f}"
    for idx, val in enumerate(canopy_ppa.heights.squeeze())
]
secaxy = ax2.secondary_yaxis("right")
secaxy.set_yticks(canopy_ppa.heights.squeeze(), labels=z_star_labels)

area_labels = [f"$A_{idx+1} = ${val}" for idx, val in enumerate(area_values)]
secaxx = ax2.secondary_xaxis("top")
secaxx.set_xticks(area_values, labels=area_labels)

plt.tight_layout()
```

### Crown and canopy gap fractions

The crown and canopy models include two gap fractions:

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
    name="tall", h_max=30, m=3, n=1.5, par_ext=0.6, f_g=0.1, ca_ratio=500
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
hghts = np.linspace(community.stem_allometry.stem_height.max(), 0, num=101)[:, None]
gappy_canopy = Canopy(community=gappy_community, layer_heights=hghts)
gappy_canopy_ppa = Canopy(
    community=gappy_community, fit_ppa=True, canopy_gap_fraction=1 / 8
)
```

We can now calculate the cumulative projected crown and leaf areas with height ($A_p(z)$
and $\tilde{A}_{cp}(z)$ in the crown model) across individuals within the two
communities.

```{code-cell} ipython3
# Calculate the cumulative projected crown and leaf areas for the non-gappy community
crown_area = np.nansum(
    canopy.crown_profile.projected_crown_area * community.cohorts.n_individuals,
    axis=1,
)

leaf_area = np.nansum(
    canopy.crown_profile.projected_leaf_area * community.cohorts.n_individuals,
    axis=1,
)

# Calculate the cumulative projected crown and leaf areas for the gappy community
gappy_crown_area = np.nansum(
    gappy_canopy.crown_profile.projected_crown_area
    * gappy_community.cohorts.n_individuals,
    axis=1,
)

gappy_leaf_area = np.nansum(
    gappy_canopy.crown_profile.projected_leaf_area
    * gappy_community.cohorts.n_individuals,
    axis=1,
)
```

The plots below show the cumulative leaf and crown areas across the communities along
with the PPA layer closure heights. The first thing to note is that the projected crown
area for the communities is identical: the cohorts have the same crown shapes, stem
heights and number of individuals. For the non-gappy community, the projected leaf area
is also identical. However:

* The gappy community has a canopy gap fraction of 1/8, reducing the available total
  crown area within a single layer to 28 m2. The layer closure heights under the PPA
  model are displaced upwards and an extra closed canopy layer is needed to fit the
  community crown area into the available area.

* The crown gap fractions in the gappy community are not zero and so the projected leaf
  area is displaced downwards within the canopy. Note that this does not affect the
  location of the canopy layer closure heights, but it _does_ affect the location of
  leaf area for the purposes of [light capture](./light_capture.md).

```{code-cell} ipython3
:tags: [hide-input]

fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(6, 8))

# Non gappy community
area_values = np.arange(1, 5) * 32

ax1.plot(crown_area, hghts)
ax1.plot(leaf_area, hghts, linestyle="--", color="red")
ax1.hlines(canopy_ppa.heights, 0, 128, color="black", linewidth=0.5)
ax1.vlines(area_values, 0, 25, color="black", linewidth=0.5)

# Add secondary ticks showing the layer heights and closure areas.
z_star_labels = [
    f"$z^*_{idx+1} = ${val:0.2f}"
    for idx, val in enumerate(canopy_ppa.heights.squeeze())
]
secaxy = ax1.secondary_yaxis("right")
secaxy.set_yticks(canopy_ppa.heights.squeeze(), labels=z_star_labels)

area_labels = [f"$A_{idx+1} = ${val}" for idx, val in enumerate(area_values)]
secaxx = ax1.secondary_xaxis("top")
secaxx.set_xticks(area_values, labels=area_labels)
ax1.text(128, 25, "Non-gappy community", ha="right", va="top", backgroundcolor="white")


# Gappy community
area_values = np.arange(1, 5) * 28

ax2.plot(gappy_crown_area, hghts, label="Crown area")
ax2.plot(gappy_leaf_area, hghts, linestyle="--", color="red", label="Leaf area")
ax2.hlines(gappy_canopy_ppa.heights, 0, 128, color="black", linewidth=0.5)
ax2.vlines(area_values, 0, 25, color="black", linewidth=0.5)
ax2.set_xlabel("Projected area (m2)")

# Add secondary ticks showing the layer heights and closure areas.
z_star_labels = [
    f"$z^*_{idx+1} = ${val:0.2f}"
    for idx, val in enumerate(gappy_canopy_ppa.heights.squeeze())
]
secaxy = ax2.secondary_yaxis("right")
secaxy.set_yticks(gappy_canopy_ppa.heights.squeeze(), labels=z_star_labels)

area_labels = [f"$A_{idx+1} = ${val}" for idx, val in enumerate(area_values)]
secaxx = ax2.secondary_xaxis("top")
secaxx.set_xticks(area_values, labels=area_labels)

ax2.text(128, 25, "Gappy community", ha="right", va="top", backgroundcolor="white")
ax2.legend()

plt.tight_layout()
```
