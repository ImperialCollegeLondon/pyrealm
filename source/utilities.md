---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Utilities

The {mod}`~pyrealm.utilities` module contains a set of utility functions that
are shared between modules, including:

* conversion functions for common alternative inputs to models.
* input checking   

```{code-cell} python
:tags: [hide-input]
# This code loads required packages and then creates a representative range of
# values of the core variables to use in function plots.
#
# Note that the ranges are created (`_1d`) but are also cast to two dimensional
# arrays of repeating values (`_2d`) to generate response surfaces for functions
# with multuple inputs.

from matplotlib import pyplot
import numpy as np
from pyrealm import utilities
%matplotlib inline

# Set the resolution of examples
n_pts = 101

# Create a range of representative values for key inputs.
ta_1d = np.linspace(0, 60, n_pts)
vp_1d = np.linspace(0, 20, n_pts)
rh_1d = np.linspace(0, 1, n_pts)
sh_1d = np.linspace(0, 0.02, n_pts)

# Broadcast the range into arrays with repeated values.
ta_2d = np.broadcast_to(ta_1d, (n_pts, n_pts))
vp_2d = np.broadcast_to(vp_1d, (n_pts, n_pts))
rh_2d = np.broadcast_to(rh_1d, (n_pts, n_pts))
```

## Conversion functions

### Hygrometric conversions


```{code-cell} ipython3
:tags: [hide-input]
# Create a sequence of air temperatures and calculate the saturated vapour pressure   
vp_sat = utilities.calc_vp_sat(ta_1d)

# Plot ta against vp_sat
pyplot.plot(ta_1d, vp_sat)
pyplot.xlabel('Temperature °C')
pyplot.ylabel('Saturated vapour pressure (kPa)')
pyplot.show()
```



```{code-cell} python
:tags: [hide-input]
vpd = utilities.convert_vp_to_vpd(vp_2d, ta_2d.transpose())

# Plot vpd
fig, ax = pyplot.subplots()
CS = ax.contour(vp_1d, ta_1d, vpd, colors='black')
ax.clabel(CS, inline=1, fontsize=10)
ax.set_title('Converting VP to VPD')
ax.set_xlabel('Vapour Pressure (kPa)')
ax.set_ylabel('Temperature (°C)')
pyplot.show()
```

```{code-cell} python
:tags: [hide-input]
vpd = utilities.convert_rh_to_vpd(rh_2d, ta_2d.transpose())

# Plot vpd
fig, ax = pyplot.subplots()
CS = ax.contour(rh_1d, ta_1d, vpd, colors='black')
ax.clabel(CS, inline=1, fontsize=10)
ax.set_title('Converting RH to VPD')
ax.set_xlabel('Relative humidity (-)')
ax.set_ylabel('Temperature (°C)')
pyplot.show()
```

```{code-cell} python
:tags: [hide-input]
# Create a sequence of air temperatures and calculate the saturated vapour pressure   
vpd1 = utilities.convert_sh_to_vpd(sh_1d, ta=20, patm=101.325)
vpd2 = utilities.convert_sh_to_vpd(sh_1d, ta=30, patm=101.325)
vpd3 = utilities.convert_sh_to_vpd(sh_1d, ta=20, patm=90)
vpd4 = utilities.convert_sh_to_vpd(sh_1d, ta=30, patm=90)


# Plot vpd against sh
pyplot.plot(sh_1d, vpd1, sh_1d, vpd2, sh_1d, vpd3, sh_1d, vpd4)
pyplot.xlabel('Specific humidity (kg kg-1)')
pyplot.ylabel('Vapour pressure deficit (kPa)')
pyplot.show()
```



## Module documentation

```{eval-rst}
.. automodule:: pyrealm.utilities

```
