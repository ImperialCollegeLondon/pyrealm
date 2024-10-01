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
tall_pft = PlantFunctionalType(name="tall", h_max=30, m=1.5, n=4, f_g=0.2, ca_ratio=500)

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

We can now plot the canopy stems alongside the community $A_p(z)$ profile, and
superimpose the calculated $z^*_l$ values and the cumulative canopy area for each layer
to confirm that the calculated values coincide with the profile. Note here that the
total area at each closed layer height is omitting the community gap fraction.

```{code-cell}
community_Ap_z = np.nansum(Ap_z, axis=1)

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 5))

zcol = "red"

# Plot the canopy parts
ax1.plot(stem_x + rz_below_zm, z, color="khaki")
ax1.plot(stem_x - rz_below_zm, z, color="khaki")
ax1.plot(stem_x + rz_above_zm, z, color="forestgreen")
ax1.plot(stem_x - rz_above_zm, z, color="forestgreen")

# Add the maximum radius
ax1.plot(np.vstack((stem_x - rm, stem_x + rm)), np.vstack((zm, zm)), color="firebrick")

# Plot the stem centre lines
ax1.vlines(stem_x, 0, pft.height, linestyles="-", color="grey")

ax1.set_ylabel("Height above ground ($z$, m)")
ax1.set_xlabel("Arbitrary stem position")
ax1.hlines(z_star, 0, (stem_x + rm).max(), color=zcol, linewidth=0.5)

# Plot the projected community crown area by height along with the heights
# at which different canopy layers close
ax2.hlines(z_star, 0, community_Ap_z.max(), color=zcol, linewidth=0.5)
ax2.vlines(
    canopy_area * np.arange(1, len(z_star)) * (1 - community_gap_fraction),
    0,
    pft.height.max(),
    color=zcol,
    linewidth=0.5,
)

ax2.plot(community_Ap_z, z)
ax2.set_xlabel("Projected crown area above height $z$ ($A_p(z)$, m2)")

# Add z* values on the righthand axis
ax3 = ax2.twinx()


def z_star_labels(X):
    return [f"$z^*_{l + 1}$ = {z:.2f}" for l, z in enumerate(X)]


ax3.set_ylim(ax2.get_ylim())
ax3.set_yticks(z_star)
ax3.set_yticklabels(z_star_labels(z_star))

ax4 = ax2.twiny()

# Add canopy layer closure areas on top axis
cum_area = np.arange(1, len(z_star)) * canopy_area * (1 - community_gap_fraction)


def cum_area_labels(X):
    return [f"$A_{l + 1}$ = {z:.1f}" for l, z in enumerate(X)]


ax4.set_xlim(ax2.get_xlim())
ax4.set_xticks(cum_area)
ax4.set_xticklabels(cum_area_labels(cum_area))

plt.tight_layout()
```

The projected area from individual stems to each canopy layer can then be calculated at
$z^*_l$ and hence the projected area of canopy **within each layer**.

```{code-cell}
# Calculate the canopy area above z_star for each stem
Ap_z_star = calculate_projected_area(z=z_star[:, None], pft=pft, m=m, n=n, qm=qm, zm=zm)

print(Ap_z_star)
```

```{code-cell}
:lines_to_next_cell: 2

# Calculate the contribution _within_ each layer per stem
Ap_within_layer = np.diff(Ap_z_star, axis=0, prepend=0)

print(Ap_within_layer)
```

+++ {"lines_to_next_cell": 2}

### Leaf area within canopy layers

The projected area occupied by leaves at a given height $\tilde{A}_{cp}(z)$ is
needed to calculate light transmission through those layers. This differs from the
projected area $A_p(z)$ because, although a tree occupies an area in the canopy
following the PPA, a **crown gap fraction** ($f_g$) reduces the actual leaf area
at a given height $z$.

The crown gap fraction does not affect the overall projected canopy area at ground
level or the community gap fraction: the amount of clear sky at ground level is
governed purely by $f_G$. Instead it models how leaf gaps in the upper canopy are
filled by leaf area at lower heights. It captures the vertical distribution of
leaf area within the canopy: a higher $f_g$ will give fewer leaves at the top of
the canopy and more leaves further down within the canopy.

The calculation of $\tilde{A}_{cp}(z)$ is defined as:

$$
\tilde{A}_{cp}(z)=
\begin{cases}
0, & z \gt H \\
A_c \left(\dfrac{q(z)}{q_m}\right)^2 \left(1 - f_g\right), & H \gt z \gt z_m \\
Ac - A_c \left(\dfrac{q(z)}{q_m}\right)^2 f_g, & zm \gt z
\end{cases}
$$

The function below calculates $\tilde{A}_{cp}(z)$.

```{code-cell}
def calculate_leaf_area(
    z: float,
    fg: float,
    pft,
    m: Stems,
    n: Stems,
    qm: Stems,
    zm: Stems,
) -> np.ndarray:
    """Calculate leaf area above a given height.

    This function takes PFT specific parameters (shape parameters) and stem specific
    sizes and estimates the projected crown area above a given height $z$. The inputs
    can either be scalars describing a single stem or arrays representing a community
    of stems. If only a single PFT is being modelled then `m`, `n`, `qm` and `fg` can
    be scalars with arrays `H`, `Ac` and `zm` giving the sizes of stems within that
    PFT.

    Args:
        z: Canopy height
        fg: crown gap fraction
        m, n, qm : PFT specific shape parameters
        pft, qm, zm: stem data
    """

    # Calculate q(z)
    qz = calculate_relative_canopy_radius_at_z(z, pft.height, m, n)

    # Calculate Ac term
    Ac_term = pft.crown_area * (qz / qm) ** 2
    # Set Acp either side of zm
    Acp = np.where(z <= zm, pft.crown_area - Ac_term * fg, Ac_term * (1 - fg))
    # Set Ap = 0 where z > H
    Acp = np.where(z > pft.height, 0, Acp)

    return Acp
```

The plot below shows how the vertical leaf area profile for the community changes for
different values of $f_g$. When $f_g = 0$, then $A_cp(z) = A_p(z)$ (red line) because
there are no crown gaps and hence all of the leaf area is within the crown surface. As
$f_g \to 1$, more of the leaf area is displaced deeper into the canopy, leaves in the
lower crown intercepting light coming through holes in the upper canopy.

```{code-cell}
fig, ax1 = plt.subplots(1, 1, figsize=(6, 5))

for fg in np.arange(0, 1.01, 0.05):

    if fg == 0:
        color = "red"
        label = "$f_g = 0$"
        lwd = 0.5
    elif fg == 1:
        color = "blue"
        label = "$f_g = 1$"
        lwd = 0.5
    else:
        color = "black"
        label = None
        lwd = 0.25

    Acp_z = calculate_leaf_area(z=z[:, None], fg=fg, pft=pft, m=m, n=n, qm=qm, zm=zm)
    ax1.plot(np.nansum(Acp_z, axis=1), z, color=color, linewidth=lwd, label=label)

ax1.set_xlabel(r"Projected leaf area above height $z$ ($\tilde{A}_{cp}(z)$, m2)")
ax1.legend(frameon=False)
```

We can now calculate the crown area occupied by leaves above the height of each closed
layer $z^*_l$:

```{code-cell}
# Calculate the leaf area above z_star for each stem
crown_gap_fraction = 0.05
Acp_z_star = calculate_leaf_area(
    z=z_star[:, None], fg=crown_gap_fraction, pft=pft, m=m, n=n, qm=qm, zm=zm
)

print(Acp_z_star)
```

And from that, the area occupied by leaves **within each layer**. These values are
similar to the projected crown area within layers (`Ap_within_layer`, above) but
leaf area is displaced into lower layers because $f_g > 0$.

```{code-cell}
# Calculate the contribution _within_ each layer per stem
Acp_within_layer = np.diff(Acp_z_star, axis=0, prepend=0)

print(Acp_within_layer)
```

### Light transmission through the canopy

Now we can use the leaf area by layer and the Beer-Lambert equation to calculate light
attenuation through the canopy layers.

$f_{abs} = 1 - e ^ {-kL}$,

where $k$ is the light extinction coefficient ($k$) and $L$ is the leaf area index
(LAI). The LAI can be calculated for each stem and layer:

```{code-cell}
LAI = Acp_within_layer / canopy_area
print(LAI)
```

This can be used to calculate the LAI of individual stems but also the LAI of each layer
in the canopy:

```{code-cell}
LAI_stem = LAI.sum(axis=0)
LAI_layer = LAI.sum(axis=1)

print("LAI stem = ", LAI_stem)
print("LAI layer = ", LAI_layer)
```

The layer LAI values can now be used to calculate the light transmission of each layer and
hence the cumulative light extinction profile through the canopy.

```{code-cell}
f_abs = 1 - np.exp(-pft.traits.par_ext * LAI_layer)
ext = np.cumprod(f_abs)

print("f_abs = ", f_abs)
print("extinction = ", ext)
```

One issue that needs to be resolved is that the T Model implementation in `pyrealm`
follows the original implementation of the T Model in having LAI as a fixed trait of
a given plant functional type, so is constant for all stems of that PFT.

```{code-cell}
print("f_abs = ", (1 - np.exp(-pft.traits.par_ext * pft.traits.lai)))
```

## Things to worry about later

Herbivory - leaf fall (transmission increases, truncate at 0, replace from NSCs) vs leaf
turnover (transmission constant, increased GPP penalty)

Leaf area dynamics in PlantFATE - acclimation to herbivory and transitory decreases in
transimission, need non-structural carbohydrates  to recover from total defoliation.

Leaf economics.
