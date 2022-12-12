---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.6.0
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Worked example

This page shows the use of the {mod}`~pyrealm.pmodel` module to go from raw data to a
global map of gross primary productivity (GPP). The first section of code loads some
example data from a NetCDF format file.

```{code-cell} ipython3
from matplotlib import pyplot as plt
import numpy as np
import netCDF4
from pyrealm import pmodel

%matplotlib inline

# Load an example dataset containing the main variables.
ds = netCDF4.Dataset('../../../data/pmodel_inputs.nc')
ds.set_auto_mask(False)

# Extract the six variables for all months
temp = ds['temp'][:]
co2 = ds['CO2'][:]         # Note - spatially constant but mapped.
elev = ds['elevation'][:]  # Note - temporally constant but repeated
vpd = ds['VPD'][:]
fapar = ds['fAPAR'][:]
ppfd = ds['ppfd'][:]

ds.close()
```

The model can now be run using that data. The first step is to convert the elevation
data to atmospheric pressure, and then this is used to set the photosynthetic
environment for the model:

```{code-cell} ipython3
# Convert elevation to atmospheric pressure
patm = pmodel.calc_patm(elev)

# Mask out temperature values below -25°C
temp[temp < -25] = np.nan

# Clip VPD to force negative VPD to be zero
vpd = np.clip(vpd, 0, np.inf)

# Calculate the photosynthetic environment
env = pmodel.PModelEnvironment(tc=temp, co2=co2, patm=patm, vpd=vpd)
env.summarize()
```

That environment can then be run to calculate the P model predictions for light use
efficiency:

```{code-cell} ipython3
# Run the P model
model = pmodel.PModel(env)

ax = plt.imshow(model.lue[0, :, :], origin='lower', extent=[-180,180,-90,90])
plt.colorbar()
plt.show()
```

Finally, the light use efficiency can be used to calculate GPP given the
photosynthetic photon flux density and fAPAR.

```{code-cell} ipython3
# Scale the outputs from values per unit iabs to realised values
model.estimate_productivity(fapar, ppfd)

ax = plt.imshow(model.gpp[0, :, :], origin='lower', extent=[-180,180,-90,90])
plt.colorbar()
plt.show()
```
