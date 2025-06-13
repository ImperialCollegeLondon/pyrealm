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

# Light capture in the canopy

:::{admonition} Warning

This area of `pyrealm` is in active development and this notebook currently contains
notes and initial demonstration code.

:::

```{code-cell} ipython3
import numpy as np


from pyrealm.demography.flora import PlantFunctionalType, Flora
from pyrealm.demography.community import Cohorts, Community

from pyrealm.demography.canopy import Canopy, CohortCanopyData

import matplotlib.pyplot as plt
```

## Light capture in a simple model

The big leaf model provides a simple model of the light absorption by a tree. The key
variables in determining the light absorption are:

* The tree has a crown area $A_c$.
* The leaf area index $L$ for the plant functional type is a simple factor that sets the
  leaf density of the crown, allowing stems with identical crown area to vary in the
  density of actual leaf surface for light capture. Values of $L$ are always expressed
  as the area of leaf surface per square meter.
* The extinction coefficient $k$ for a PFT sets how much light is absorbed when passing
  through the leaf surface of that PFT.

The values of $k$ and $L$ set the fraction of light absorbed $f_{abs}$ through the
single crown layer following the Beer-Lambert law:

$$
f_{abs} = 1 - e^{-kL}
$$

In the big leaf model of a tree canopy, the entire crown area $A_c$ is exposed to the
downwelling photosynthetic photon flux density (PPFD) at the top of the canopy and hence
the absorbed radiation ($I_{abs}$) is:

$$
I_{abs} = \textrm{PPFD} \cdot A_c  (1 - e^{-kL})
$$

Effectively the crown is treated as a single flat area - hence the "big leaf".

## Light capture in a vertically structured canopy

Instead of the big leaf model, the [canopy model](./canopy.md) uses the vertical
distribution of leaf area for individual stems, as described in the [crown
model](./crown.md), to partition all of crown area of individuals in a community into
layers. The light capture model needs to account for light absorption through the canopy
layers to calculate how much light is absorbed by different individuals within the
different layers.

The canopy model defines a set of canopy closure heights ($z^*_l$ for layers $l = 1,
..., m$) that define the  canopy layers, but what are these layers exactly? Each layer:

* has a depth that extends from the closure height of the layer above (or the top of the
  canopy for the first layer), down to the layer's closure height.
* contains projected crown and leaf area from at least one individual in the community.
* absorbs light, following the projected leaf area of the individuals in the layer and
  the leaf area index (LAI, $L$).

Each successive layer therefore has access to less and less light as it is filtered out
by the layers above.

```{admonition} Projected leaf area, leaf area index and layer depth

The projected leaf area and leaf area index sound similar but represent very different
things:

* The projected leaf area is a simple two dimensional measure of how much of the total
  crown area is found at different heights.
* The actual amount of leaf area within the layer is the projected leaf area multiplied
  by the LAI. Two individuals can have the same crown area and vertical distribution of
  projected leaf area, but still differ in their LAI. A stem with a larger LAI will
  have denser canopy that can capture more light.
* The LAI is effectively the "depth" of leaf area within a layer, because as $L$
  increases, more light is captured within those leaves because $L$ is included in the
  Beer-Lambert law.
* The actual vertical depth of a layer - the distance between two successive canopy
  closure heights - is not actually used in the light capture model.

```

For a single individual within the community, the light absorbed in a particular layer
$l$ of the canopy is described as:

$$
I_{abs} = \textrm{PPFD} \cdot T_l \cdot A_{fl} \cdot  (1 - e^{-kL})
$$

The first two terms are the PPFD at the top of the canopy and a term $T_l$ that captures
the fraction of the canopy top flux that is transmitted down to layer $l$.

The last two terms then describe the ability of the individual to capture that
downwelling light flux. The Beer-Lambert term is assumed not to change with height - the
fraction of light absorbed per metre squared of leaf area does not vary through the
canopy. The other term $A_{fl}$ is the area of foliage of that individual in the layer
and is the difference between the projected leaf area at the top and bottom of the
layers. That is, given the function from the crown model for the projected leaf area at
height $z$ ($\tilde{A}_{cp}(z)$) and a sequence of layer heights $z_l$:

$$
A_{fl} =  \begin{cases}
  \tilde{A}_{cp}(z_l), & \textrm{if } l=1 \\
  \tilde{A}_{cp}(z_l) - \tilde{A}_{cp}(z_{l-1}) & \textrm{otherwise}
\end{cases}
$$

```{attention}
The calculation of $A_{fl}$ uses the **projected leaf area** and not the projected crown
area.

* The projected crown area dictates how each stem occupies space in canopy layers:
  competition for space in the canopy layers use crown area.
* When the canopy gap fraction ($f_g$) is one, then the projected crown area and
  projected leaf area are equal. However then $f_g < 1$, the individual tree crown has
  holes within their crown and projected leaf area is vertically displaced down into the
  canopy. Because projected leaf area dictates where the actual leaves are located in
  the canopy, this is what is used to calculate absorbance in the light capture model.
```

### Calculation of light transmission terms

For a given layer $l$, the community-wide average fraction of light absorbed by the
layer ($a_l$) can be calculated as the average absorption of each individual, weighted
by the total area occupied by the community $A$. Given a community of $N$ stems:

$$
a_{l} = \frac{\sum^{i=1}_{N} {A_{fl}}_{i} \cdot  (1 - e^{-k_i L_i})}{A}
$$

```{attention}
Note that the calculation of average transmission requires the intermediate calculation
of average absorbance in order to account for unoccupied area in the layer. This can
arise in incompletely closed layers at the bottom of the canopy or where a canopy or
crown gap fraction is set. Because the absorbance of empty space is zero, the
calculation of average absorbance implicitly accounts for unfilled space.

The same is not true for transmission, where unfilled area would need to be included
in the summation with a transmittance value of one.
```

The average transmission through the layer can then be calculated as $t_l = 1 - a_l$ and
the cumulative transmission $T_l$ can be calculated as the cumulative product of layer
transmission values after setting $t_1 = 1$:

$$
T_{l} =  \begin{cases}
  1 & \textrm{if } l=1 \\
  T_{l-1} \cdot t_l & \textrm{otherwise}
\end{cases}
$$

The last value of $T_l$ is the fraction of canopy top radiation passing through the
lowest canopy layer and reaching the ground.

## Numerical examples of light capture calculation

The code implements a layer by layer simulation of light capture through a simple canopy
and then shows that the array calculation used in `pyrealm` gives the same results as
the simulation.

### Input data

The input data sets up the stem leaf area within five canopy layers for three cohorts of
trees occupying an 8 m2 plot. The cohorts vary in their leaf area index ($L$),
extinction coefficient ($k$) and number of individuals. The crown area of each stem is
arranged into vertical layers, representing a set of closed canopy layers as in the
perfect plasticity approximation.

```{code-cell} ipython3
# Vertical distribution of projected leaf area for each cohort.
projected_leaf_area = np.array(
    [[6, 2, 0], [10, 6, 0], [13, 9, 1], [13, 11, 4], [13, 11, 5]]
)

# Calculate the leaf area per layer
layer_leaf_area = np.diff(projected_leaf_area, axis=0, prepend=0)
layer_leaf_area
```

```{code-cell} ipython3
# Light extinction parameters for cohorts.
pft_lai = np.array([2, 1, 2])
pft_par_ext = np.array([0.5, 0.5, 0.6])

n_individuals = np.array([1, 1, 2])
cell_area = 8

# Set an initial light flux in µmol m-2 s-1
initial_ppfd = 1000  # µmol m2 s1
```

Just to show the layer structure in more detail, if we multiply the stem leaf area by
the number of individuals, we get the leaf area from each cohort in each layer, and then
the sum of values within layers shows the leaf area occupying the available cell area
within each layer except for the last.

```{code-cell} ipython3
cohort_leaf_area = layer_leaf_area * n_individuals
cohort_leaf_area
```

```{code-cell} ipython3
cohort_leaf_area.sum(axis=1)
```

### Layer-wise simulation

The loop below iterates over the canopy layers from the top layer to the ground. At each
layer:

* The absorption fraction of each layer for each cohort is calculated following the
  Beer-Lambert law as $1- e^{-kL}$ and used to calculate:
  * the actual PPFD absorbed per metre squared
  * the fraction of the initial PPFD absorbed in that layer by the leaf area of each
    cohort.
* The stem per m2 values are scaled up to the community level using the stem leaf area
  in each cohort and the number of individuals and used to calculate:
  * the total flux captured across the cell in that layer
  * the average captured flux per m2 and hence
  * the remaining PPFD reaching the layer below and
  * the average fraction of initial PPFD transmitted through the layer

```{code-cell} ipython3
# Set the initial PPFD
ppfd = initial_ppfd

# Create stores for variable created and updated in the loop
simulated_transmission = np.empty((6, 1))
simulated_transmission[0, 0] = 1
simulated_per_stem_f_abs = np.empty((5, 3))
simulated_total_capture = 0

# Loop over the layers
for layer in np.arange(5):

    # Calculate the per stem light capture per m2
    per_stem_light_abs = ppfd * (1 - np.exp(-pft_par_ext * pft_lai))
    simulated_per_stem_f_abs[layer, :] = per_stem_light_abs / initial_ppfd

    # Calculate the total light captured across the layer
    cohort_capture = per_stem_light_abs * layer_leaf_area[layer,] * n_individuals

    # Scale that back down to give the average captured flux per m2
    average_captured_ppfd = cohort_capture.sum() / cell_area
    simulated_total_capture += cohort_capture.sum()

    # Calculate the remaining flux transmitted to the next layer
    ppfd -= average_captured_ppfd
    simulated_transmission[layer + 1, 0] = ppfd / initial_ppfd
```

The resulting transmission start with the fraction of light reaching the first layer
(always 1) and ends with the fraction of light passing through the last layer to the
ground.

```{code-cell} ipython3
simulated_transmission
```

The per stem fraction absorbed gives the fraction of the canopy top incident light
absorbed by stem leaf area for each cohort in each layer.

Critically these values **do not** assume any steam leaf area is actually present in the
layer. They are what the leaves **could achieve** if present and so need to be
multiplied by stem leaf area to give actual absorbance.

```{code-cell} ipython3
simulated_per_stem_f_abs
```

Lastly - as a check sum - the total capture shows how much of the incident light across
the whole cell (PPFD x cell area) is captured by leaves.

```{code-cell} ipython3
simulated_total_capture
```

### Matrix approach

The code below calculates the same quantities more efficiently using array based
calculation.

First, we can calculate an array of Beer-Lambert absorption coefficients for the stems
in each cohort: this will be constant across array layers.

```{code-cell} ipython3
stem_absorption = 1 - np.exp(-pft_par_ext * pft_lai)
stem_absorption
```

Next, we calculate the average absorption within each layer. We find the sum of the
product of the per stem absorptions and the total cohort leaf areas and then divide by
the cell area to give the average fraction absorbed ($a_l$) for a given square meter.

```{code-cell} ipython3
# Calculate the layer average absorption
average_layer_absorption = (stem_absorption * cohort_leaf_area).sum(axis=1) / cell_area
average_layer_absorption
```

We can also calculate the average leaf area index within the layer - this is useful for
easy calculation of the total leaf surface within the layer across cohorts as the
product of cell area and average leaf area index.

```{code-cell} ipython3
cohort_leaf_area_index = np.broadcast_to(pft_lai, layer_leaf_area.shape)

average_layer_lai = (cohort_leaf_area_index * cohort_leaf_area).sum(axis=1) / cell_area

average_layer_lai
```

We can now calculate average transmission through each layers ($t_l = 1 - a_l$):

```{code-cell} ipython3
average_layer_transmission = 1 - average_layer_absorption
average_layer_transmission
```

And hence the cumulative product of the average per layer transmissions to give the
fraction of downwelling light flux transmitted to each layer ($T_l$):

```{code-cell} ipython3
transmission_profile = np.cumprod(np.concat([[1], average_layer_transmission]))[:, None]
transmission_profile
```

We can then check that the simulated and matrix based calculations are the same:

```{code-cell} ipython3
if np.allclose(simulated_transmission, transmission_profile):
    print("\U00002705 Transmission matches")
```

Next, calculate a matrix of stem fractional absorption as the product of the per
layer stem absorptions and the per layer average transmission

```{code-cell} ipython3
per_stem_f_abs = stem_absorption * transmission_profile[:-1]
per_stem_f_abs
```

And again check that we get the same solution:

```{code-cell} ipython3
if np.allclose(simulated_per_stem_f_abs, per_stem_f_abs):
    print("\U00002705 Per stem fraction absorbed matches")
```

We can now calculate the total absorption in each layer for each stem. This is the key
result for use in most cases: **how much light flux is absorbed by each stem in the leaf
area of each layer**.

```{code-cell} ipython3
per_stem_absorption = per_stem_f_abs * layer_leaf_area * initial_ppfd
per_stem_absorption
```

Lastly, we can verify that the total absorption across all individuals within the
community is the same as the simulation:

```{code-cell} ipython3
total_capture = (per_stem_absorption * n_individuals).sum()
total_capture
```

```{code-cell} ipython3
if np.allclose(simulated_total_capture, total_capture):
    print("\U00002705 Total capture matches")
```

## Implementation in pyrealm

The calculation above can be directly repeated using the `CohortCanopyData` class: this
data class is not typically used directly but is designed for use with the input data
defined above.

```{code-cell} ipython3
canopy_light_model = CohortCanopyData(
    projected_leaf_area=projected_leaf_area,
    lai=pft_lai,
    par_ext=pft_par_ext,
    n_individuals=n_individuals,
    cell_area=cell_area,
)
```

The light transmission profile and the final light transmission to the ground match the
calculations above.

```{code-cell} ipython3
canopy_light_model.community_data.transmission_profile
```

```{code-cell} ipython3
canopy_light_model.community_data.transmission_to_ground
```

Similarly, the per stem light absorption matches:

```{code-cell} ipython3
initial_ppfd * canopy_light_model.stem_leaf_area * canopy_light_model.fapar
```

### Calculation of light capture for a community

The `Canopy` model automatically calculates the light capture model for a community. The
code below uses the final example community from the documentation of the [vertical
canopy model](./canopy.md) calculations.

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
    cell_area=150,
    cell_id=1,
    cohorts=Cohorts(
        dbh_values=np.array([0.1, 0.20, 0.5]),
        n_individuals=np.array([7, 3, 2]),
        pft_names=np.array(["short", "short", "tall"]),
    ),
)
```

We can use the community object directly to calculate the big leaf approximation of
light capture.

```{code-cell} ipython3
(
    initial_ppfd
    * gappy_community.stem_allometry.crown_area
    * (
        1
        - np.exp(-gappy_community.stem_traits.par_ext * gappy_community.stem_traits.lai)
    )
).round(2)
```

If the cell area of the community is set to be greater than the total crown
area of the community, a canopy model fitted to the community gives the same estimates:
there is a single canopy layer with no shading of any of the stem canopies.

```{code-cell} ipython3
# Calculate the canopy profile across vertical heights
gappy_canopy_ppa = Canopy(community=gappy_community, fit_ppa=True)

# Layer closure heights
gappy_canopy_ppa.heights
```

The canopy model then contains the data required to calculate the absorbed irradiance
within each layer for a stem in each cohort.

```{code-cell} ipython3
i_abs = (
    initial_ppfd
    * gappy_canopy_ppa.cohort_data.fapar
    * gappy_canopy_ppa.cohort_data.stem_leaf_area
)
i_abs.sum(axis=0).round(2)
```

The example code below places the same community of cohorts into a much smaller area:
the tree crowns cannot all fit into a single layer and so are arranged under the PPA
model into layers.

```{code-cell} ipython3
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

# Recalculate the canopy profile across vertical heights
gappy_canopy_ppa = Canopy(community=gappy_community, fit_ppa=True)
```

The PPA solution for this community finds four layers and we can show the canopy closure
heights of those layers and the amount of leaf area present in each layer for each of
the three cohorts.

```{code-cell} ipython3
# Layer closure heights
gappy_canopy_ppa.heights
```

```{code-cell} ipython3
gappy_canopy_ppa.cohort_data.stem_leaf_area.round(2)
```

All of the first layer is occupied by the crowns of the two individuals in the tallest
cohort and the rest of the leaf area is shaded, reducing the overall light absorbed by
each stem below the big leaf prediction. This is most marked in the smallest stems,
which only have leaf area in the very lowest layer.

```{code-cell} ipython3
i_abs = (
    initial_ppfd
    * gappy_canopy_ppa.cohort_data.fapar
    * gappy_canopy_ppa.cohort_data.stem_leaf_area
)
i_abs.sum(axis=0).round(2)
```
