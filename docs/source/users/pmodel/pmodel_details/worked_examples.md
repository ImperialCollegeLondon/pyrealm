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

# Worked examples

This page shows two worked examples of how to use `pyrealm` to make predictions using
the P Model.

The first example uses a single point but the second shows how the package can be used
with array data. The `pyrealm` package uses the `numpy` package and expects arrays of
data to be be passed to all inputs. Input arrays can be a single scalar value, but all
non-scalar inputs must be **arrays with the same shape**: the `pyrealm` packages does
not attempt to resolve the broadcasting of array dimensions.

```{code-cell} ipython3
from importlib import resources

from matplotlib import pyplot as plt
import numpy as np
import xarray

from pyrealm.pmodel.pmodel import PModel
from pyrealm.pmodel import PModelEnvironment
from pyrealm.core.pressure import calc_patm
```

:::{warning}

The `pyrealm` package uses a modular approach to define many of the [shared
components](../shared_components/overview.md) of the standard and subdaily P Model.
Some combinations of the methods implemented may modify the same aspect of the model
using different approaches.

As an example, there are multiple approaches to incorporating effects of
[soil moisture stress](../shared_components/soil_moisture.md) on productivity, via
modulation of $\phi_0$, $m_j$ and the calculation of GPP penalty factors.

At present, `pyrealm` does not automatically check the compatibility of method
selection, so take care when setting methods options for fitting a P Model.

:::

## Simple point estimate

This example calculates a single point estimate of GPP. The first step is to use
estimates of environmental variables to calculate the
photosynthetic environment for the model
({class}`~pyrealm.pmodel.pmodel_environment.PModelEnvironment`).

The example shows the steps required using a single site with:

* a temperature of 20°C,
* standard atmospheric at sea level (101325 Pa),
* a vapour pressure deficit of 0.82 kPa (~ 65% relative humidity), and
* an atmospheric $\ce{CO2}$ concentration of 400 ppm.

### Estimating productivity

The {class}`~pyrealm.pmodel.pmodel_environment.PModelEnvironment` also accepts estimates
of the fraction of absorbed photosynthetically active radiation ($f_{APAR}$, `fapar`,
unitless) and the photosynthetic photon flux density (PPFD,`ppfd`, µmol m-2 s-1).
Together these are used to calculate the asorbed irradiance, which is used to scale up
the estimated light use efficiency to estimate the actual productivity of the model.
Here we are using:

* An absorption fraction of 0.91 (-), and
* a PPFD of 834 µmol m-2 s-1.

```{warning}

In the {meth}`~pyrealm.pmodel.pmodel_environment.PModelEnvironment`, the estimated PPFD
must be expressed as **µmol m-2 s-1**.

Estimates of PPFD sometimes use different temporal or spatial scales - for
example daily moles of photons per hectare. Although GPP can also be expressed
with different units, many other predictions of the P Model ($J_{max}$,
$V_{cmax}$, $g_s$ and $r_d$) _must_ be expressed as µmol m-2 s-1 and so this
standard unit must also be used for PPFD.
```

```{code-cell} ipython3
# Calculate the PModelEnvironment
env = PModelEnvironment(tc=20.0, patm=101325.0, vpd=820, co2=400, fapar=0.91, ppfd=834)
env
```

The `env` object now holds the photosynthetic environment, which can be re-used with
different P Model settings. The representation of a
{class}`~pyrealm.pmodel.pmodel_environment.PModelEnvironment` object (`env`) is
deliberately terse - just the shape of the data - but the
{class}`PModelEnvironment.summarize<pyrealm.pmodel.pmodel_environment.PModelEnvironment.summarize>`
method provides a more detailed summary of the attributes.

```{code-cell} ipython3
env.summarize()
```

### Fitting the P Model

Next, the P Model can be fitted to the photosynthetic environment using the
({class}`~pyrealm.pmodel.pmodel.PModel`) class:

```{code-cell} ipython3
model = PModel(env)
```

The returned model object holds a lot of information. The representation of the
model object shows a terse display of the settings used to run the model:

```{code-cell} ipython3
model
```

A {class}`~pyrealm.pmodel.pmodel.PModel` instance also has a
{meth}`~pyrealm.pmodel.pmodel.PModel.summarize` method that summarizes settings and
displays a summary of calculated predictions. Initially, this shows two measures of
photosynthetic efficiency: the intrinsic water use efficiency (``iwue``) and the light
use efficiency (``lue``).

```{code-cell} ipython3
model.summarize()
```

### $\chi$ estimates and $\ce{CO2}$ limitation

The instance also contains a {class}`~pyrealm.pmodel.optimal_chi.OptimalChiPrentice14`
object,
recording key parameters from the [calculation of
$\chi$](../shared_components/optimal_chi).
This object also has a {meth}`~pyrealm.pmodel.optimal_chi.OptimalChiABC.summarize`
method:

```{code-cell} ipython3
model.optchi.summarize()
```

## 3D grid example

This example shows how the {mod}`~pyrealm.pmodel` module can be used with array inputs
to calculate a global map of gross primary productivity (GPP).

First, we load some
example data from a NetCDF format file using the excellent {mod}`xarray` package.
These data are 0.5° global grids containing data for 2 months and so the loaded
data are three dimensional arrays and shape `(2, 360, 720)` . Note that the arrays have
to be the same size so some of the variables have repeated data across dimensions:

* The CO2 data is globally constant for each month, but the values are repeated for each
  cell.
* Elevation is constant across months, so the data for each month is repeated.

```{code-cell} ipython3
# Load an example dataset containing the forcing variables.
data_path = resources.files("pyrealm_build_data.rpmodel") / "pmodel_global.nc"
ds = xarray.load_dataset(data_path)

# Extract the six variables for the two months and convert from
# xarray DataArray objects to numpy arrays
temp = ds["temp"].to_numpy()
co2 = ds["CO2"].to_numpy()
elev = ds["elevation"].to_numpy()
vpd = ds["VPD"].to_numpy()
fapar = ds["fAPAR"].to_numpy()
ppfd = ds["ppfd"].to_numpy()
```

The model can now be run using that data. The first step is to convert the elevation
data to atmospheric pressure, and then this is used to set the photosynthetic
environment for the model:

```{code-cell} ipython3
# Convert elevation to atmospheric pressure
patm = calc_patm(elev)

# Mask out temperature values below -25°C
temp[temp < -25] = np.nan

# Clip VPD to force negative VPD to be zero
vpd = np.clip(vpd, 0, np.inf)

# Calculate the photosynthetic environment
env = PModelEnvironment(tc=temp, co2=co2, patm=patm, vpd=vpd, fapar=fapar, ppfd=ppfd)
env.summarize()
```

That environment can then be run to calculate the P model predictions for GPP:

```{code-cell} ipython3
# Run the P model
model = PModel(env)

fig, (ax1, ax2) = plt.subplots(2, 1)

# Plot LUE for first month
im = ax1.imshow(model.lue[0, :, :], origin="lower", extent=[-180, 180, -90, 90])
plt.colorbar(im, fraction=0.022, pad=0.03, ax=ax1)
ax1.set_title("LUE")

im = ax2.imshow(model.gpp[0, :, :], origin="lower", extent=[-180, 180, -90, 90])
plt.colorbar(im, fraction=0.022, pad=0.03, ax=ax2)
ax2.set_title("GPP")

plt.tight_layout()
```
