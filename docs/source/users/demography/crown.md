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
define the vertical distribution of leaf area, following the implementation of crown
shape in the Plant-FATE model {cite}`joshi:2022a`.

## Crown traits

The crown model for a plant functional type (PFT) is driven by four traits within
the {class}`~pyrealm.demography.flora.PlantFunctionalType` class:

* The `m` and `n` ($m, n$) traits set the vertical shape of the crown profile.
* The `ca_ratio` trait sets the area of the crown ($A_c$) relative to the stem size.
* The `f_g` ($f_g$) trait sets the crown gap fraction and sets the vertical
  distribution of leaves within the crown.

### Canopy shape

For a stem of height $H$, the $m$ and $n$ traits are used to calculate the *relative*
crown radius $q(z)$ at a height $z$ of as:

$$
q(z)= m n \left(\dfrac{z}{H}\right) ^ {n -1}
    \left( 1 - \left(\dfrac{z}{H}\right) ^ n \right)^{m-1}
$$

In order to align the arbitrary relative radius values with the predictions of the
T Model for the stem, we then need to find the height at which the relative radius
is at its maximum and a scaling factor that will convert the relative area at this
height to the expected crown area under the T Model. These can be calculated using
the following two additional traits, which are invariant for a PFT.

* `z_max_prop` ($p_{zm}$) sets the proportion of the height of the stem at which
   the maximum relative crown radius is found.
* `q_m` ($q_m$) is used to scale the size of the crown radius at $z_{max}$ to match
   the expected crown area.

$$
\begin{align}
p_{zm} &= \left(\dfrac{n-1}{m n -1}\right)^ {\tfrac{1}{n}}\\[8pt]
q_m &= m n \left(\dfrac{n-1}{m n -1}\right)^ {1 - \tfrac{1}{n}}
    \left(\dfrac{\left(m-1\right) n}{m n -1}\right)^ {m-1}
\end{align}
$$

For individual stems, with expected height $H$ and crown area $A_c$, we can then
calculate:

* the height $z_m$ at which the maximum crown radius $r_m$ is found,
* a height-specific scaling factor $r_0$ such that $\pi q(z_m)^2 = A_c$,
* the actual crown radius $r(z)$ at a given height $z$.

$$
\begin{align}
z_m &= H p_{zm}\\[8pt]
r_0 &= \frac{1}{q_m}\sqrt{\frac{A_c}{\pi}}\\[8pt]
r(z) &= r_0 \; q(z)
\end{align}
$$

### Projected crown and leaf area

From the crown radius profile, the model can then be used to calculate how crown area
and leaf area accumulates from the top of the crown towards the ground, giving
functions given height $z$ for:

* the projected crown area $A_p(z)$, and
* the projected leaf area $\tilde{A}_{cp}(z)$.

The projected crown area is calculated for a stem of known height and crown area is:

$$
A_p(z)=
\begin{cases}
A_c, & z \le z_m \\
A_c \left(\dfrac{q(z)}{q_m}\right)^2, & H > z > z_m \\
0, & z > H
\end{cases}
$$

That is, the projected crown area is zero above the top of the stem, increases to the
expected crown area at $z_max$ and is then constant to ground level.

The projected leaf area $\tilde{A}_{cp}(z)$ models how the vertical distribution of
leaf area within the crown is modified by the crown gap fraction $f_g$. This trait
models how leaf gaps higher in the crown are filled by leaf area at lower heights:
it does not change the profile of the crown but allows leaf area in the crown to be
displaced downwards. When $f_g = 0$, the projected crown and leaf areas are identical,
but as $f_g \to 1$ the projected leaf area is pushed downwards.
The calculation of $\tilde{A}_{cp}(z)$ is defined as:

$$
\tilde{A}_{cp}(z)=
\begin{cases}
0, & z \gt H \\
A_c \left(\dfrac{q(z)}{q_m}\right)^2 \left(1 - f_g\right), & H \gt z \gt z_m \\
Ac - A_c \left(\dfrac{q(z)}{q_m}\right)^2 f_g, & zm \gt z
\end{cases}
$$

## Calculating crown model traits in `pyrealm`

The {class}`~pyrealm.demography.flora.PlantFunctionalType` class is typically
used to set specific PFTs, but the functions to calculate $q_m$ and $p_{zm}$
are used directly below to provides a demonstration of the impacts of each trait.

```{code-cell}
# Set a range of values for m and n traits
m = n = np.arange(1.0, 5, 0.1)

# Calculate the invariant q_m and z_max_prop traits
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

## Crown calculations in `pyrealm`

The examples below show the calculation of crown variables in `pyrealm`.

### Calculting crown profiles

The  {class}`~pyrealm.demography.crown.CrownProfile` class is used to calculate crown
profiles for PFTs. It requires:

* a {class}`~pyrealm.demography.flora.Flora` instance providing a set of PFTs to be
  compared,
* a {class}`~pyrealm.demography.t_model_functions.StemAllometry` instance setting the
  specific stem sizes for the profile, and
* a set of heights at which to estimate the profile variables

The code below creates a set of PFTS with differing crown trait values and then creates
a `Flora` object using the PFTs.

```{code-cell}
# A PFT with a small crown area and equal m and n values
narrow_pft = PlantFunctionalType(name="narrow", h_max=20, m=1.5, n=1.5, ca_ratio=20)
# A PFT with an intermediate crown area  and m < n
medium_pft = PlantFunctionalType(name="medium", h_max=20, m=1.5, n=4, ca_ratio=500)
# A PFT with a wide crown area and m > n
wide_pft = PlantFunctionalType(name="wide", h_max=20, m=4, n=1.5, ca_ratio=2000)

# Generate a Flora instance using those PFTs
flora = Flora([narrow_pft, medium_pft, wide_pft])
flora
```

The Flora object can also be used to show a table of canopy variables:

```{code-cell}
# TODO - add a Flora.to_pandas() method
flora_data = pd.DataFrame({k: getattr(flora, k) for k in flora.trait_attrs})
flora_data[["name", "ca_ratio", "m", "n", "f_g", "q_m", "z_max_prop"]]
```

The next section of code generates the `StemAllometry` to use for the profiles.
The T Model requires DBH to define stem size - here the
{meth}`~pyrealm.demography.t_model_functions.calculate_dbh_from_height` function
is used to back-calculate the required DBH values to give three stems with similar
heights that are near the maximum height for each PFT.

```{code-cell}
# Generate the expected stem allometries at similar heights for each PFT
stem_height = np.array([19, 17, 15])
stem_dbh = calculate_dbh_from_height(
    a_hd=flora.a_hd, h_max=flora.h_max, stem_height=stem_height
)
stem_dbh
```

```{code-cell}
# Calculate the stem allometries
allometry = StemAllometry(stem_traits=flora, at_dbh=stem_dbh)
```

We can again use {mod}`pandas` to get a table of those allometric predictions:

```{code-cell}
pd.DataFrame({k: getattr(allometry, k) for k in allometry.allometry_attrs})
```

Finally, we can define a set of vertical heights. In order to calculate the
variables for each PFT at each height, this needs to be provided as a column array,
that is with a shape `(N, 1)`.

We can then calculate the crown profiles.

```{code-cell}
# Create a set of vertical heights as a column array.
z = np.linspace(-1, 20.0, num=211)[:, None]

# Calculate the crown profile across those heights for each PFT
crown_profiles = CrownProfile(stem_traits=flora, stem_allometry=allometry, z=z)
```

The `crown_profiles` object then provides the four crown profile attributes describe
above calculated at each height $z$:

* The relative crown radius $q(z)$
* The crown radius $r(z)$
* The projected crown area
* The projected leaf area

```{code-cell}
crown_profiles
```

### Visualising crown profiles

The code below generates a plot of the vertical shape profiles of the crowns for each
stem. For each stem:

* the dashed line shows how the relative crown radius $q(z)$ varies with height $z$,
* the solid line shows the actual crown radius $r(z)$ varies with height, and
* the dotted horizontal line shows the height at which the maximum crown radius is
  found ($z_{max}$).

Note that the equation for the relative radius $q(z)$ that defines canopy shape can
make predictions with actual values outside of the range of the actual stem and not
only where $0 \leq z \leq H$.

```{code-cell}
fig, ax = plt.subplots(ncols=1)

# Find the maximum of the actual and relative maximum crown widths
max_relative_crown_radius = np.nanmax(crown_profiles.relative_crown_radius, axis=0)
max_crown_radius = np.nanmax(crown_profiles.crown_radius, axis=0)
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

    ax.set_xlabel("Crown profile")
    ax.set_ylabel("Height above ground (m)")

ax.set_aspect(aspect=1)
```

We can also use the `CanopyProfile` class with a single row of heights to calculate
the crown profile at the expected $z_{max}$ and show that this matches the expected
crown area from the T Model allometry.

```{code-cell}
# Calculate the crown profile across those heights for each PFT
z_max = flora.z_max_prop * stem_height
profile_at_zmax = CrownProfile(stem_traits=flora, stem_allometry=allometry, z=z_max)

print(profile_at_zmax.crown_radius**2 * np.pi)
print(allometry.crown_area)
```

### Visualising crown and leaf projected areas

We can also use the crown profile to generate projected area plots. This is hard to see
using the PFTs defined above because they have very different crown areas, so the code
below generates new profiles for a new set of PFTs that have similar crown area ratios
but different shapes and gap fractions.

```{code-cell}
no_gaps_pft = PlantFunctionalType(
    name="no_gaps", h_max=20, m=1.5, n=1.5, f_g=0, ca_ratio=380
)
few_gaps_pft = PlantFunctionalType(
    name="few_gaps", h_max=20, m=1.5, n=4, f_g=0.05, ca_ratio=400
)
many_gaps_pft = PlantFunctionalType(
    name="many_gaps", h_max=20, m=4, n=1.5, f_g=0.2, ca_ratio=420
)

# Calculate allometries for each PFT at the same stem DBH
area_stem_dbh = np.array([0.4, 0.4, 0.4])
area_flora = Flora([no_gaps_pft, few_gaps_pft, many_gaps_pft])
area_allometry = StemAllometry(stem_traits=area_flora, at_dbh=area_stem_dbh)

# Calculate the crown profiles across those heights for each PFT
area_z = np.linspace(0, area_allometry.stem_height.max(), 201)[:, None]
area_crown_profiles = CrownProfile(
    stem_traits=area_flora, stem_allometry=area_allometry, z=area_z
)
```

The plot below then shows how projected crown area (solid lines) and leaf area (dashed
lines) change with height along the stem.

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

    ax.plot(area_crown_profiles.projected_crown_area[:, pft_idx], area_z, color=colour)
    ax.plot(
        area_crown_profiles.projected_leaf_area[:, pft_idx],
        area_z,
        color=colour,
        linestyle="--",
    )
    ax.set_xlabel("Projected area (m2)")
    ax.set_ylabel("Height above ground (m)")
```

We can also generate predictions for a single PFT with varying crown gap fraction. In
the plot below, note that all leaf area is above $z_{max}$ when $f_g=1$ and all leaf
area is *below*

```{code-cell}
fig, ax = plt.subplots(ncols=1)

# Loop over f_g values
for f_g in np.linspace(0, 1, num=11):

    label = None
    colour = "gray"

    if f_g == 0:
        label = "$f_g=0$"
        colour = "red"
    elif f_g == 1:
        label = "$f_g=1$"
        colour = "blue"

    # Create a flora with a single PFT with current f_g and then generate a
    # stem allometry and crown profile
    flora_f_g = Flora(
        [PlantFunctionalType(name="example", h_max=20, m=2, n=2, f_g=f_g)]
    )
    allometry_f_g = StemAllometry(stem_traits=flora_f_g, at_dbh=np.array([0.4]))
    profile = CrownProfile(
        stem_traits=flora_f_g, stem_allometry=allometry_f_g, z=area_z
    )

    # Plot the projected leaf area with height
    ax.plot(profile.projected_leaf_area, area_z, color=colour, label=label, linewidth=1)

# Add a horizontal line for z_max
ax.plot(
    [-1, allometry_f_g.crown_area[0] + 1],
    [allometry_f_g.crown_z_max, allometry_f_g.crown_z_max],
    linestyle="--",
    color="black",
    label="$z_{max}$",
    linewidth=1,
)

ax.set_ylabel(r"Vertical height ($z$, m)")
ax.set_xlabel(r"Projected leaf area ($\tilde{A}_{cp}(z)$, m2)")
ax.legend(frameon=False)
```
