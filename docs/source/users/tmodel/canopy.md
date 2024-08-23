---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: pyrealm_python3
  language: python
  name: pyrealm_python3
---

# Canopy model

This notebook walks through the steps in generating the canopy model as (hopefully) used
in Plant-FATE.

## The T Model

The T Model provides a numerical description of how tree geometry scales with stem
diameter, and an allocation model of how GPP predicts changes in stem diameter.

The implementation in `pyrealm` provides a class representing a particular plant
functional type, using a set of traits. The code below creates a PFT with the default
set of trait values.

```{warning}
This sketch:

* Assumes a single individual of each stem diameter, but in practice
  we are going to want to include a number of individuals to capture cohorts.
* Assumes a single PFT, where we will need to provide a mixed community.
* Consequently handles forest inventory properties in a muddled way: we will
  likely package all of the stem data into a single class, probably a community
  object.
```

```{code-cell}
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

from pyrealm.tmodel import TTree

np.set_printoptions(precision=3)

# Plant functional type with default parameterization
pft = TTree(diameters=np.array([0.1, 0.15, 0.2, 0.25, 0.38, 0.4, 1.0]))
pft.traits
```

The scaling of a set of trees is automatically calculated using the initial diameters to
the `TTree` instance. This automatically calculates the other dimensions, such as
height, using the underlying scaling equations of the T Model.

```{code-cell}
pft.height
```

```{code-cell}
pft.crown_area
```

### Crown shape

Jaideep's extension of the T Model adds a crown shape model, driven by two parameters
($m$ and $n$) that provide a very flexible description of the vertical crown profile.
A third constant parameter ($q_m$) can be calculated from $m$ and $n$ as:

$$
q_m = m n \left(\dfrac{n-1}{m n -1}\right)^ {1 - \tfrac{1}{n}}
    \left(\dfrac{\left(m-1\right) n}{m n -1}\right)^ {m-1}
$$

For individual stems, with expected height $H$ and crown area $A_c$, we can estimate:

* the height $z_m$ at which the maximum crown radius $r_m$ is found, and
* a slope $r_0$ that scales the relative canopy radius so that the $r_m$ matches
  the allometric prediction of $A_c$ from the T Model.

$$
\begin{align}
z_m &= H \left(\dfrac{n-1}{m n -1}\right)^ {\tfrac{1}{n}} \\[8pt]
r_0 &= \frac{1}{q_m}\sqrt{\frac{A_c}{\pi}}
\end{align}
$$

```{code-cell}
def calculate_qm(m, n):

    # Constant q_m
    return (
        m
        * n
        * ((n - 1) / (m * n - 1)) ** (1 - 1 / n)
        * (((m - 1) * n) / (m * n - 1)) ** (m - 1)
    )


def calculate_stem_canopy_factors(pft, m, n):

    # Height of maximum crown radius
    zm = pft.height * ((n - 1) / (m * n - 1)) ** (1 / n)

    # Slope to give Ac at zm
    r0 = 1 / qm * np.sqrt(pft.crown_area / np.pi)

    return zm, r0


# Shape parameters for a fairly top heavy crown profile
m = 2
n = 5
qm = calculate_qm(m=m, n=n)
zm, r0 = calculate_stem_canopy_factors(pft=pft, m=m, n=n)

print("qm = ", np.round(qm, 4))
print("zm = ", zm)
print("r0 = ", r0)
```

The following functions then provide the value at height $z$ of relative $q(z)$ and
actual $r(z)$ canopy radius:

$$
\begin{align}
q(z) &= m n \left(\dfrac{z}{H}\right) ^ {n -1}
    \left( 1 - \left(\dfrac{z}{H}\right) ^ n \right)^{m-1}\\[8pt]
r(z) &= r_0 \; q(z)
\end{align}
$$

```{code-cell}
def calculate_relative_canopy_radius_at_z(z, H, m, n):
    """Calculate q(z)"""

    z_over_H = z / H

    return m * n * z_over_H ** (n - 1) * (1 - z_over_H**n) ** (m - 1)
```

```{code-cell}
# Validate that zm and r0 generate the predicted maximum crown area
q_zm = calculate_relative_canopy_radius_at_z(zm, pft.height, m, n)
rm = r0 * q_zm
print("rm = ", rm)
```

```{code-cell}
np.allclose(rm**2 * np.pi, pft.crown_area)
```

Vertical crown radius profiles can now be calculated for each stem:

```{code-cell}
# Create an interpolation from ground to maximum stem height, with 5 cm resolution.
# Also append a set of values _fractionally_ less than the exact height  of stems
# so that the height at the top of each stem is included but to avoid floating
# point issues with exact heights.

zres = 0.05
z = np.arange(0, pft.height.max() + 1, zres)
z = np.sort(np.concatenate([z, pft.height - 0.00001]))

# Convert the heights into a column matrix to broadcast against the stems
# and then calculate r(z) = r0 * q(z)
rz = r0 * calculate_relative_canopy_radius_at_z(z[:, None], pft.height, m, n)

# When z > H, rz < 0, so set radius to 0 where rz < 0
rz[np.where(rz < 0)] = 0

np.cumsum(np.convolve(rm, np.ones(2), "valid") + 0.1)
```

Those can be plotted out to show the vertical crown radius profiles

```{code-cell}
# Separate the stems along the x axis for plotting
stem_x = np.concatenate(
    [np.array([0]), np.cumsum(np.convolve(rm, np.ones(2), "valid") + 0.4)]
)

# Get the canopy sections above and below zm
rz_above_zm = np.where(np.logical_and(rz > 0, np.greater.outer(z, zm)), rz, np.nan)
rz_below_zm = np.where(np.logical_and(rz > 0, np.less_equal.outer(z, zm)), rz, np.nan)

# Plot the canopy parts
plt.plot(stem_x + rz_below_zm, z, color="khaki")
plt.plot(stem_x - rz_below_zm, z, color="khaki")
plt.plot(stem_x + rz_above_zm, z, color="forestgreen")
plt.plot(stem_x - rz_above_zm, z, color="forestgreen")

# Add the maximum radius
plt.plot(np.vstack((stem_x - rm, stem_x + rm)), np.vstack((zm, zm)), color="firebrick")

# Plot the stem centre lines
plt.vlines(stem_x, 0, pft.height, linestyles="-", color="grey")

plt.gca().set_aspect("equal")
```

## Canopy structure

The canopy structure model uses the perfect plasticity approximation (PPA), which
assumes that plants can arrange their canopies to fill the available space $A$.
It takes the **projected area of stems** $Ap(z)$ within the canopy and finds the heights
at which each canopy layer closes ($z^*_l$ for $l = 1, 2, 3 ...$) where the total projected
area of the canopy equals $lA$.

### Canopy projected area

The projected area $A_p$ for a stem with height $H$, a maximum crown area $A_c$ at a
height $z_m$ and $m$, $n$ and $q_m$ for the associated plant functional type is

$$
A_p(z)=
\begin{cases}
A_c, & z \le z_m \\
A_c \left(\dfrac{q(z)}{q_m}\right)^2, & H > z > z_m \\
0, & z > H
\end{cases}
$$

```{code-cell}
Stems = float | np.ndarray


def calculate_projected_area(
    z: float,
    pft,
    m: Stems,
    n: Stems,
    qm: Stems,
    zm: Stems,
) -> np.ndarray:
    """Calculate projected crown area above a given height.

    This function takes PFT specific parameters (shape parameters) and stem specific
    sizes and estimates the projected crown area above a given height $z$. The inputs
    can either be scalars describing a single stem or arrays representing a community
    of stems. If only a single PFT is being modelled then `m`, `n`, `qm` and `fg` can
    be scalars with arrays `H`, `Ac` and `zm` giving the sizes of stems within that
    PFT.

    Args:
        z: Canopy height
        m, n, qm : PFT specific shape parameters
        pft, qm, zm: stem data

    """

    # Calculate q(z)
    qz = calculate_relative_canopy_radius_at_z(z, pft.height, m, n)

    # Calculate Ap given z > zm
    Ap = pft.crown_area * (qz / qm) ** 2
    # Set Ap = Ac where z <= zm
    Ap = np.where(z <= zm, pft.crown_area, Ap)
    # Set Ap = 0 where z > H
    Ap = np.where(z > pft.height, 0, Ap)

    return Ap
```

The code below calculates the projected crown area for each stem and then plots the
vertical profile for individual stems and across the community.

```{code-cell}
# Calculate the projected area for each stem
Ap_z = calculate_projected_area(z=z[:, None], pft=pft, m=m, n=n, qm=qm, zm=zm)

# Plot the calculated values for each stem and across the community
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 5))

# Plot the individual stem projected area
ax1.set_ylabel("Height above ground ($z$, m)")
ax1.plot(Ap_z, z)
ax1.set_xlabel("Stem $A_p(z)$ (m2)")

# Plot the individual stem projected area
ax2.plot(np.nansum(Ap_z, axis=1), z)
ax2.set_xlabel("Total community $A_p(z)$ (m2)")

plt.tight_layout()
```

### Canopy closure and canopy gap fraction

The total cumulative projected area shown above is modified by a community-level
**canopy gap fraction** ($f_G$) that captures the overall proportion of the canopy area
that is left unfilled by canopy. This gap fraction, capturing processes such as crown
shyness, describes the proportion of open sky visible from the forest floor.

The definition of the height of canopy layer closure ($z^*_l$) for a given canopy
layer $l = 1, ..., l_m$ is then:

$$
\sum_1^{N_s}{ A_p(z^*_l)} = l A(1 - f_G)
$$

This can be found numerically using a root solver as:

$$
\sum_1^{N_s}{ A_p(z^*_l)} - l A(1 - f_G) = 0
$$

The total number of layers $l_m$ in a canopy, where the final layer may not be fully closed,
can be found given the total crown area across stems as:

$$
l_m = \left\lceil \frac{\sum_1^{N_s}{ A_c}}{ A(1 - f_G)}\right\rceil
$$

```{code-cell}
def solve_canopy_closure_height(
    z: float,
    l: int,
    A: float,
    fG: float,
    m: Stems,
    n: Stems,
    qm: Stems,
    pft: Stems,
    zm: Stems,
) -> np.ndarray:
    """Solver function for canopy closure height.

    This function returns the difference between the total community projected area
    at a height $z$ and the total available canopy space for canopy layer $l$, given
    the community gap fraction for a given height. It is used with a root solver to
    find canopy layer closure heights $z^*_l* for a community.

    Args:
        m, n, qm : PFT specific shape parameters
        H, Ac, zm: stem specific sizes
        A, l: cell area and layer index
        fG: community gap fraction
    """

    # Calculate Ap(z)
    Ap_z = calculate_projected_area(z=z, pft=pft, m=m, n=n, qm=qm, zm=zm)

    # Return the difference between the projected area and the available space
    return Ap_z.sum() - (A * l) * (1 - fG)


def calculate_canopy_heights(
    A: float,
    fG: float,
    m: Stems,
    n: Stems,
    qm: Stems,
    pft,
    zm: Stems,
):

    # Calculate the number of layers
    total_community_ca = pft.crown_area.sum()
    n_layers = int(np.ceil(total_community_ca / (A * (1 - fG))))

    # Data store for z*
    z_star = np.zeros(n_layers)

    # Loop over the layers TODO - edge case of completely filled final layer
    for lyr in np.arange(n_layers - 1):
        z_star[lyr] = root_scalar(
            solve_canopy_closure_height,
            args=(lyr + 1, A, fG, m, n, qm, pft, zm),
            bracket=(0, pft.height.max()),
        ).root

    return z_star
```

The example below calculates the projected crown area above ground level for the example
stems. These should be identical to the crown area of the stems.

```{code-cell}
# Set the total available canopy space and community gap fraction
canopy_area = 32
community_gap_fraction = 2 / 32

z_star = calculate_canopy_heights(
    A=canopy_area, fG=community_gap_fraction, m=m, n=n, qm=qm, pft=pft, zm=zm
)

print("z_star = ", z_star)
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
# Calculate the contribution _within_ each layer per stem
Ap_within_layer = np.diff(Ap_z_star, axis=0, prepend=0)

print(Ap_within_layer)
```

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
ext = np.cumproduct(f_abs)

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
