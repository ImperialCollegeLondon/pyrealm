---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
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
---

# The canopy model

:::{admonition} Warning

This area of `pyrealm` is in active development and this notebook currently contains
notes and initial demonstration code.

:::

```{code-cell} ipython3
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

from pyrealm.demography.flora import PlantFunctionalType, Flora
from pyrealm.demography.community import Community
from pyrealm.demography.crown import CrownProfile, get_crown_xy
from pyrealm.demography.canopy import Canopy
from pyrealm.demography.t_model_functions import StemAllometry
```

The `canopy` module in `pyrealm` is used to calculate a model of the light environment
across all of cohorts within a plant [community](./community.md). Each cohort consists
of a number of identically-sized individuals $n_{ch}$ from a given [plant functional
type (PFT)](./flora.md). Because the individuals are identically sized, they all share
the same [stem allometry](./t_model.md) and the same  [crown model](./crown.md).

## Light extinction for a single stem

The key variables in determining the light environment for a given
stem are as follows:

* The projected crown area ${A}_{p}(z)$ sets how crown area accumulates, given
  the crown shape, from the top of the stem to the ground.
* The projected leaf area $\tilde{A}_{cp}(z)$ modifies the crown area to allow
  for the vertical displacement of crown area by the crown gap fraction.
* The leaf area index $L$ for the PFT is a simple factor that sets the leaf density
  of the crown, allowing stems with identical crown area to vary in the density of
  actual leaf surface for light capture. Values of $L$ are always expressed as the area
  of leaf surface per square meter.
* The extinction coefficient $k$ for a PFT sets how much light is absorbed when passing
  through the leaf surface of that PFT.

For a single stem, the fraction of light absorbed through the entire crown is described
by the Beer-Lambert law:

$$
f_{abs} = 1 - e^{-kL}
$$

However, to calculate a vertical profile of light extinction through a crown with total
area $A_c$ and maximum stem height $H$, that equation needs to be expanded to calculate
the fraction of $L$ that falls between pairs of vertical heights $z_a > z_b$. The actual
area amount of leaf area $A_l$ for an individual stem falling between those two heights
is simply the diffence in projected leaf area between the two heights:

$$
A_{l[a,b]} = \tilde{A}_{cp}(z_a) - \tilde{A}_{cp}(z_b)
$$

Given that information, the calculation of $f_{abs}$ becomes:

$$
f_{abs[a,b]} = 1 - e^{\left(-k\dfrac{L A_{l[a,b]}}{A_c}\right)}
$$

When $z_a = H$ and $z_b=0$, then $A_{l[a,b]} = A_c$ and hence simplifies to
the original equation.

The code below creates a simple example community containing a single cohort containing
a single stem and then calculates the light extinction profile down through the canopy.

```{code-cell} ipython3
# Create a simple community with a single stem
simple_pft = PlantFunctionalType(name="defaults", m=2, n=2)
simple_flora = Flora([simple_pft])
stem_dbh = np.array([0.5])
simple_stem = StemAllometry(stem_traits=simple_flora, at_dbh=stem_dbh)

# The total area is exactly the crown area
total_area = simple_stem.crown_area[0]

# Define a simple community
simple_community = Community(
    flora=simple_flora,
    cell_area=total_area,
    cell_id=1,
    cohort_dbh_values=stem_dbh,
    cohort_n_individuals=np.array([1]),
    cohort_pft_names=np.array(["defaults"]),
)

# Get the canopy model for the simple case from the canopy top
# to the ground
hghts = np.linspace(simple_stem.stem_height[0], 0, num=101)[:, None]
simple_canopy = Canopy(community=simple_community, layer_heights=hghts)
```

As a simple check that the calculation across height layers is correct, the canopy
instance returns a vertical light extinction profile. The last value in this profile
should equal the whole canopy $f_{abs}$ calculated using the simple Beer-Lambert
equation and the PFT trait values.

```{code-cell} ipython3
:scrolled: true

print(simple_canopy.extinction_profile[-1])
```

```{code-cell} ipython3
:scrolled: true

print(1 - np.exp(-simple_pft.par_ext * simple_pft.lai))
```

The plot below shows:

1. The shape of crown radius for the stem (solid line) along with the projected leaf
   radius (dashed line). The leaf radius does not show the actual expected boundary of
   leaf area - which follows the crown - but is useful to visualise the displacement of
   leaf area on the same scale as the crown radius.
2. The vertical profile of the actual leaf area $A_{l[a,b]}$ between two height.
3. The resulting light absorption at each height.
4. The light extinction profile through the canopy.

```{code-cell} ipython3
fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, sharey=True, figsize=(12, 6))

# Generate plot structures for stem profiles
ch, crown_radius = get_crown_xy(
    crown_profile=simple_canopy.crown_profile,
    stem_allometry=simple_community.stem_allometry,
    attr="crown_radius",
    two_sided=False,
)[0]

ch, projected_leaf_radius = get_crown_xy(
    crown_profile=simple_canopy.crown_profile,
    stem_allometry=simple_community.stem_allometry,
    attr="projected_leaf_radius",
    two_sided=False,
)[0]

ax1.plot(crown_radius, ch, color="red")
ax1.plot(projected_leaf_radius, ch, linestyle="--", color="red")
ax1.set_xlabel("Profile radius (m)")
ax1.set_ylabel("Vertical height (m)")

# Plot the leaf area between heights for stems
ax2.plot(simple_canopy.layer_stem_leaf_area, hghts, color="red")
ax2.set_xlabel("Leaf area (m2)")

# Plot the fraction of light absorbed at different heights
ax3.plot(simple_canopy.layer_f_abs, hghts, color="red")
ax3.set_xlabel("Light absorption fraction (-)")

# Plot the light extinction profile through the canopy.
ax4.plot(simple_canopy.extinction_profile, hghts, color="red")
ax4.set_xlabel("Cumulative light\nabsorption fraction (-)")
```

## Light extinction within a community

Within a community, the calculations above need to be modified to account for:

* the number of cohorts $n$,
* the number of individuals $i_{h}$ within each cohort,
* differences in the LAI $L_{h}$ and light extinction coefficients $k_{h}$ between
  cohorts,
* scaling LAI to the total area available to the community $A_T$ rather than the cohort
  specific crown area $A_h$.

Within the community, each cohort now requires a whole cohort  LAI component $L_H$,
which consists of the total leaf area index across individuals divided by the total
community area to give an average leaf area index across the available space:

$$
L_H = \frac{i_h L_h A_h}{A_T}
$$

The Beer-Lambert equation across the cohorts is then:

$$
f_{abs} = 1 - e^{\left(\sum\limits_{m=1}^{n}-k_h{[m]} L_{H[m]}\right)}
$$

or equivalently

$$
f_{abs} = 1 - \prod\limits_{m=1}^{n}e^{-k_{[m]}  L_{H[m]}}
$$

This equation can be adjusted as before to partition light absorption within vertical
layers and the implementation is demonstrated below using a simple community containing
two plant functional types:

* a shorter understory tree with a columnar canopy and no crown gaps
* a taller canopy tree with a top heavy canopy and more crown gaps

and then three cohorts:

* 7 saplings of the short PFT
* 3 larger stems of the short PFT
* 1 large stems of tall PFT

```{code-cell} ipython3
# Define PFTs
short_pft = PlantFunctionalType(
    name="short",
    h_max=15,
    m=1.5,
    n=1.5,
    f_g=0,
    ca_ratio=380,
    lai=4,
)
tall_pft = PlantFunctionalType(name="tall", h_max=30, m=3, n=1.5, f_g=0.2, ca_ratio=500)

# Create the flora
flora = Flora([short_pft, tall_pft])

# Define a simply community with three cohorts
community = Community(
    flora=flora,
    cell_area=32,
    cell_id=1,
    cohort_dbh_values=np.array([0.05, 0.20, 0.5]),
    cohort_n_individuals=np.array([7, 3, 1]),
    cohort_pft_names=np.array(["short", "short", "tall"]),
)

# Calculate the canopy profile across vertical heights
hghts = np.linspace(community.stem_allometry.stem_height.max(), 0, num=101)[:, None]
canopy = Canopy(community=community, canopy_gap_fraction=0, layer_heights=hghts)
```

As before, we can verify that the cumulative light extinction at the bottom of the
vertical profile is equal to the expected value across the whole community.

```{code-cell} ipython3
# Calculate L_h for each cohort
cohort_lai = (
    community.cohort_data["n_individuals"]
    * community.stem_traits.lai
    * community.stem_allometry.crown_area
) / community.cell_area

# Calculate 1 - e ^ -k L
print(1 - np.exp(np.sum(-community.stem_traits.par_ext * cohort_lai)))
```

```{code-cell} ipython3
print(canopy.extinction_profile[-1])
```

```{code-cell} ipython3
cols = ["r", "b", "g"]

mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=cols)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, sharey=True, figsize=(12, 6))

# Generate plot structures for stem profiles
crown_profile = get_crown_xy(
    crown_profile=canopy.crown_profile,
    stem_allometry=community.stem_allometry,
    attr="crown_radius",
    two_sided=False,
)

leaf_profile = get_crown_xy(
    crown_profile=canopy.crown_profile,
    stem_allometry=community.stem_allometry,
    attr="projected_leaf_radius",
    two_sided=False,
)

for (stem_rh, stem_cr), (stem_lh, stem_plr), col in zip(
    crown_profile, leaf_profile, cols
):
    ax1.plot(stem_cr, stem_rh, color=col)
    ax1.plot(stem_plr, stem_lh, linestyle="--", color=col)

ax1.set_xlabel("Profile radius (m)")
ax1.set_ylabel("Vertical height (m)")

# Plot the leaf area between heights for stems
ax2.plot(canopy.layer_stem_leaf_area, hghts)
ax2.set_xlabel("Leaf area per stem (m2)")

# Plot the fraction of light absorbed at different heights
ax3.plot(canopy.layer_f_abs, hghts, color="grey")
ax3.plot(1 - canopy.layer_cohort_f_trans, hghts)
ax3.set_xlabel("Light absorption fraction (-)")

# Plot the light extinction profile through the canopy.
ax4.plot(canopy.extinction_profile, hghts, color="grey")
ax4.set_xlabel("Cumulative light\nabsorption fraction (-)")
```

## Canopy closure and canopy gap fraction

In addition to calculating profiles from a provided sequence of vertical heights, the
canopy model also implements the calculation of canopy layers, following the perfect
plasticity approximation (PPA) {cite}`purves:2008a`. This model divides the vertical
structure of the canopy into discrete closed layers. The model assumes that plants are
always able to plastically arrange their crown within the broader canopy of the
community to maximise their crown size and fill the available space $A$. When the area
$A$ is filled, a new lower canopy layer is formed until all of the individual crown area
has been distributed across within the canopy.

A simple method for finding the first canopy closure height is to find a height $z^*_1$
at which the sum of crown projected area across all stems $N_s$ in a community equals
$A$:

$$
\sum_1^{N_s}{ A_p(z^*_1)} = A
$$

However, the canopy model also allows for modification by a community-level **canopy gap
fraction** ($f_G$) that captures the overall proportion of the canopy area that is left
unfilled by canopy. This gap fraction, capturing processes such as crown shyness,
describes the proportion of open sky visible from the forest floor. This gives the
following definition of the height of canopy layer closure ($z^*_l$) for a given canopy
layer $l = 1, ..., l_m$:

$$
\sum_1^{N_s}{ A_p(z^*_l)} = l A(1 - f_G)
$$

The set of heights $z^*$  can be found numerically by using a root solver to find values
of $z^*_l$ for $l = 1, ..., l_m$ that satisfy:

$$
\sum_1^{N_s}{ A_p(z^*_l)} - l A(1 - f_G) = 0
$$

The total number of layers $l_m$ in a canopy, where the final layer may not be fully
closed, can be found given the total crown area across stems as:

$$
l_m = \left\lceil \frac{\sum_1^{N_s}{A_c}}{ A(1 - f_G)}\right\rceil
$$

```{code-cell} ipython3
canopy = Canopy(community=community, canopy_gap_fraction=2 / 32, fit_ppa=True)
```

```{code-cell} ipython3
canopy.layer_heights
```

## Implementation in `pyrealm`

The {class}`~pyrealm.demography.canopy.Canopy` class automatically finds the canopy
closure heights, given a {class}`~pyrealm.demography.community.Community` instance and
the required canopy gap fraction.

```{code-cell} ipython3
canopy.crown_profile.projected_leaf_area[:, 1]
```

```{code-cell} ipython3
canopy.layer_stem_leaf_area[:, 1]
```

```{code-cell} ipython3
plt.plot(canopy.layer_stem_leaf_area[:, 1], "ro")
```

We can then look at three key properties of the canopy model: the layer closure heights
($z^*_l$) and the projected crown areas and leaf areas at each of those heights for each
stem in the three cohorts.

There are four canopy layers, with the top two very close together because of the large
crown area in the two stems in the cohort of `tall` trees.

```{code-cell} ipython3
canopy.layer_heights
```

The `stem_crown_area` attribute then provides the crown area of each stem found in each
layer.

```{code-cell} ipython3
canopy.stem_crown_area
```

Given the canopy gap fraction, the available crown area per layer is 30 m2, so
the first two layers are taken up entirely by the two stems in the cohort of large
trees. We can confirm that the calculation is correct by calculating the total crown area
across the cohorts at each height:

```{code-cell} ipython3
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

```{code-cell} ipython3
canopy.stem_leaf_area
```

### Visualizing layer closure heights and areas

We can use the {class}`~pyrealm.demography.crown.CrownProfile` class to calculate a
community crown and leaf area profile across a range of height values. For each height,
we calculate the sum of the product of stem projected area and the number of
individuals in each cohort.

```{code-cell} ipython3
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

```{code-cell} ipython3
fig, ax = plt.subplots(ncols=1)

# Calculate the crown area at which each canopy layer closes.
closure_areas = (
    np.arange(1, len(canopy.layer_heights) + 1) * community.cell_area * (30 / 32)
)

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

```{code-cell} ipython3
ppfd = 1000

ppfd_layer = -np.diff(ppfd * np.cumprod(canopy.layer_f_trans), prepend=ppfd)
ppfd_layer
```

```{code-cell} ipython3
(1 - canopy.layer_cohort_f_trans) * ppfd_layer[:, None]
```

## Things to worry about later

Herbivory - leaf fall (transmission increases, truncate at 0, replace from NSCs) vs leaf
turnover (transmission constant, increased GPP penalty)

Leaf area dynamics in PlantFATE - acclimation to herbivory and transitory decreases in
transimission, need non-structural carbohydrates  to recover from total defoliation.

Leaf economics.
