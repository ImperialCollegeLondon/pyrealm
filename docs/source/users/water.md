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
  name: python
  version: 3.12.3
  mimetype: text/x-python
  codemirror_mode:
    name: ipython
    version: 3
  pygments_lexer: ipython3
  nbconvert_exporter: python
  file_extension: .py
---

# Water density and viscosity

The density ($\rho$, kg m-3) and viscosity ($\mu$, Pa s) of water are required inputs to
the calculation of productivity and other calculations within `pyrealm`. Both
quantities vary with temperature and atmospheric pressure, although the variation with
pressure is very small. However, the behaviour of water viscosity and density with
temperature is highly complex and algorithms for calculating them differ very widely in
the level of precision that they aim to achieve.

The `pyrealm` package provides a number of implementations for both quantities. This
includes very high precision implementations but these are expensive to calculate and
may be far more precise than is really warranted, given uncertainty in forcing variables
or the accuracy required for day to day usage. This page reviews the available methods
and shows the relative precision and computational complexity of alternative methods.

```{note}
The methods implemented in `pyrealm` are a sample of a wide range of different
implementations of varying complexity. They are not intended to be an exhaustive
survey, just to provide reasonable approaches of varying complexity.
```

## Water density

The {mod}`pyrealm.core.water` module provides the following alternative methods for
calculating water density:

* `kell` ({meth}`~pyrealm.core.water.calculate_density_h2o_kell`, {cite}`kell:1975a`),
* `jones_harris_eq6`
  ({meth}`~pyrealm.core.water.calculate_density_h2o_jones_harris_eq6`,
  {cite}`jones:1992a`),
* `jones_harris_eq8`
  ({meth}`~pyrealm.core.water.calculate_density_h2o_jones_harris_eq8`,
  {cite}`jones:1992a`),
* `chen` ({meth}`~pyrealm.core.water.calculate_density_h2o_chen`, {cite:t}`chen:2008a`),
* `fisher` ({meth}`~pyrealm.core.water.calculate_density_h2o_fisher`,
  {cite:t}`Fisher:1975tm`)

The first two of these implementations (`kell`, `jones_harris_eq6`) do not correct for
the effects of atmospheric pressure - although the functions require `patm` as inputs,
these values are not then used in the calculation.

```{code-cell} ipython3
:tags: [remove-cell]

import timeit

from importlib import resources
import matplotlib.pyplot as plt
import numpy as np
import xarray

from pyrealm.core.water import DENSITY_METHODS, VISCOSITY_METHODS
from pyrealm.core.pressure import calc_patm
from pyrealm.constants import CoreConst
from pyrealm.pmodel import PModel, PModelEnvironment

# Create some inputs
tc = np.arange(-10, 60, 0.1)[:, None]
tc = np.broadcast_to(tc, (700, 3))
tk = tc + 273.15
patm = np.array([[87000, 101325, 108570]])
N = 2 ** np.arange(10, 20)

# Calculate the values for each of the density and methods
values_density = {key: func(tc=tc, patm=patm) for key, func in DENSITY_METHODS.items()}
```

```{code-cell} ipython3
:tags: [remove-cell, skip-execution]

# This cell is slowish (~ 20 seconds, so is not executed). The results array is then
# declared in the next cell.

# Calculate the scaling of the function runtime with number of values
results_density = np.zeros((2, len(N), len(DENSITY_METHODS)))

for i, n in enumerate(N):
    for j, (_, func) in enumerate(DENSITY_METHODS.items()):
        setup = f"""
import pyrealm.core.water as mod
from pyrealm.constants import CoreConst
import numpy as np
function=getattr(mod, '{func.__name__}')
tc=np.linspace(0,60, {n})
const=CoreConst()
            """
        t = timeit.Timer(
            stmt="function(tc=tc, patm=101325, core_const=const)", setup=setup
        )
        res = t.autorange()
        results_density[:, i, j] = res
```

```{code-cell} ipython3
:tags: [remove-cell]

results_density = np.array(
    [
        [
            [5.00e04, 5.00e04, 2.00e04, 1.00e04, 1.00e04],
            [2.00e04, 5.00e04, 2.00e04, 5.00e03, 1.00e04],
            [2.00e04, 2.00e04, 1.00e04, 5.00e03, 5.00e03],
            [1.00e04, 1.00e04, 5.00e03, 2.00e03, 5.00e03],
            [5.00e03, 5.00e03, 5.00e03, 1.00e03, 1.00e03],
            [5.00e03, 5.00e03, 2.00e03, 5.00e02, 1.00e03],
            [2.00e03, 2.00e03, 1.00e03, 5.00e02, 5.00e02],
            [1.00e03, 1.00e03, 5.00e02, 2.00e02, 2.00e02],
            [2.00e02, 2.00e02, 1.00e02, 5.00e01, 5.00e01],
            [1.00e02, 1.00e02, 5.00e01, 2.00e01, 2.00e01],
        ],
        [
            [4.73e-01, 3.25e-01, 3.00e-01, 3.85e-01, 2.93e-01],
            [2.26e-01, 3.94e-01, 3.66e-01, 2.39e-01, 3.45e-01],
            [3.53e-01, 2.43e-01, 2.91e-01, 3.86e-01, 2.83e-01],
            [3.35e-01, 2.31e-01, 2.59e-01, 2.78e-01, 5.01e-01],
            [2.98e-01, 2.02e-01, 4.60e-01, 2.47e-01, 2.13e-01],
            [4.90e-01, 3.35e-01, 3.04e-01, 2.22e-01, 2.99e-01],
            [3.73e-01, 2.55e-01, 2.99e-01, 4.70e-01, 3.01e-01],
            [4.26e-01, 2.85e-01, 3.47e-01, 3.96e-01, 2.65e-01],
            [3.10e-01, 2.17e-01, 2.43e-01, 3.30e-01, 2.46e-01],
            [3.30e-01, 2.31e-01, 2.69e-01, 2.88e-01, 2.14e-01],
        ],
    ]
)
```

```{code-cell} ipython3
:tags: [remove-input]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
ax1, ax2, ax3, ax4 = axes.flatten()

for method_name, data in values_density.items():
    # Plot actual rho
    ax1.plot(tc[:, 1], data[:, 1], label=method_name)

    # Difference from chen for low pressure
    ax3.plot(
        tc[:, 1],
        (values_density["chen"] - data)[:, 0],
        label=method_name,
    )

    # Difference from chen for high pressure
    ax4.plot(
        tc[:, 1],
        (values_density["chen"] - data)[:, 2],
        label=method_name,
    )

# Performance
lines = ax2.plot(N, results_density[1, :, :] / results_density[0, :, :])
ax2.set_xscale("log")
ax2.set_yscale("log")

# Annotation
ax1.legend(frameon=False)
ax1.set_xlabel("Temperature (°C)")
ax2.set_xlabel("Data size")
ax3.set_xlabel("Temperature (°C)")
ax4.set_xlabel("Temperature (°C)")

ax1.set_ylabel(r"Density ($\rho$, kg/m3)")
ax2.set_ylabel("Processing time (seconds)")
ax3.set_ylabel(r"Density difference  ($\Delta \rho$, kg/m3)")
ax4.set_ylabel(r"Density difference  ($\Delta \rho$, kg/m3)")

ax1.text(0.90, 0.90, "A", transform=ax1.transAxes, fontsize=16)
ax1.text(0.95, 0.80, "P = 101325 Pa", transform=ax1.transAxes, ha="right")
ax2.text(0.10, 0.90, "B", transform=ax2.transAxes, fontsize=16)
ax3.text(0.10, 0.90, "C", transform=ax3.transAxes, fontsize=16)
ax3.text(0.90, 0.10, "P = 87000 Pa", transform=ax3.transAxes, ha="right")
ax4.text(0.90, 0.10, "P = 108570 Pa", transform=ax4.transAxes, ha="right")
ax4.text(0.10, 0.90, "D", transform=ax4.transAxes, fontsize=16)

fig.suptitle("Figure 1 - predictions and performance of water density methods", y=-0.02)
plt.tight_layout()
plt.show()
```

Figure 1A shows the predicted variation in density for each of the methods. Figure 1B
shows the computational performance of each method: the most complex method (`chen`) is
an order of magnitude slower than the fastest method (`jones_harris_eq6`). Figure 1C
and 1D show the differences between predicted $\rho$, relative to the `chen` method:
the differences in predicted $\rho$ are small, particularly in the temperature
range of 0-40°C. Comparing Figure 1C and 1D, differences in predicted $\rho$ due to
atmospheric pressure are extremely small and accounting for them is unlikely to be
useful.

## Water viscosity

The {mod}`pyrealm.core.water` module provides the following methods for
calculating water viscosity.

* `vogel` ({meth}`~pyrealm.core.water.calculate_viscosity_h2o_vogel`)
* `viswanath_natarajan`
  ({meth}`~pyrealm.core.water.calculate_viscosity_h2o_viswanath_natarajan`)
* `girifalco` ({meth}`~pyrealm.core.water.calculate_viscosity_h2o_girifalco`)
* `reid` ({meth}`~pyrealm.core.water.calculate_viscosity_h2o_reid`)
* `daubert_danner` ({meth}`~pyrealm.core.water.calculate_viscosity_h2o_daubert_danner`)
* `huber` ({meth}`~pyrealm.core.water.calculate_viscosity_h2o_huber`)

Only the last of these (`huber`) corrects for the effect of atmospheric pressure. Again,
all the functions require `patm` as inputs, but these values are only used by the
`huber` method.

```{code-cell} ipython3
:tags: [remove-cell]

# Calculate the viscosity predictions for each method
values_viscosity = {
    key: func(tk=tk, patm=patm) for key, func in VISCOSITY_METHODS.items()
}
```

```{code-cell} ipython3
:tags: [remove-cell, skip-execution]

# This cell is slowish (~ 20 seconds, so is not executed). The results array is then
# declared in the next cell.

# Calculate the scaling of the function runtime with number of values
results_viscosity = np.zeros((2, len(N), len(VISCOSITY_METHODS)))


for i, n in enumerate(N):
    for j, (_, func) in enumerate(VISCOSITY_METHODS.items()):
        setup = f"""
import pyrealm.core.water as mod
from pyrealm.constants import CoreConst
import numpy as np
function=getattr(mod, '{func.__name__}')
tk=np.linspace(0,60, {n}) +273.15
const=CoreConst()
            """
        t = timeit.Timer(
            stmt="function(tk=tk, patm=101325, core_const=const)", setup=setup
        )
        res = t.autorange()
        results_viscosity[:, i, j] = res
```

```{code-cell} ipython3
:tags: [remove-cell]

# Calculate the scaling of the function runtime with number of values
results_viscosity = np.array(
    [
        [
            [5.00e04, 2.00e04, 2.00e04, 5.00e04, 2.00e04, 5.00e02],
            [5.00e04, 1.00e04, 1.00e04, 2.00e04, 1.00e04, 5.00e02],
            [2.00e04, 1.00e04, 5.00e03, 2.00e04, 5.00e03, 2.00e02],
            [1.00e04, 5.00e03, 5.00e03, 5.00e03, 2.00e03, 1.00e02],
            [5.00e03, 2.00e03, 2.00e03, 5.00e03, 1.00e03, 5.00e01],
            [2.00e03, 1.00e03, 1.00e03, 2.00e03, 5.00e02, 2.00e01],
            [1.00e03, 5.00e02, 5.00e02, 1.00e03, 5.00e02, 1.00e01],
            [5.00e02, 2.00e02, 2.00e02, 5.00e02, 1.00e02, 5.00e00],
            [2.00e02, 1.00e02, 1.00e02, 2.00e02, 5.00e01, 2.00e00],
            [1.00e02, 5.00e01, 5.00e01, 1.00e02, 2.00e01, 1.00e00],
        ],
        [
            [2.87e-01, 2.32e-01, 2.65e-01, 4.00e-01, 3.75e-01, 2.17e-01],
            [4.40e-01, 2.08e-01, 2.28e-01, 2.32e-01, 3.31e-01, 3.56e-01],
            [2.97e-01, 3.95e-01, 2.10e-01, 3.93e-01, 3.16e-01, 2.61e-01],
            [3.39e-01, 4.37e-01, 4.32e-01, 2.45e-01, 2.78e-01, 3.67e-01],
            [3.15e-01, 3.25e-01, 3.48e-01, 4.77e-01, 2.65e-01, 3.12e-01],
            [2.43e-01, 3.17e-01, 3.38e-01, 3.11e-01, 2.57e-01, 2.31e-01],
            [2.25e-01, 3.22e-01, 3.42e-01, 3.04e-01, 5.09e-01, 2.24e-01],
            [2.27e-01, 2.61e-01, 2.88e-01, 2.99e-01, 2.02e-01, 2.13e-01],
            [2.70e-01, 3.06e-01, 3.31e-01, 3.64e-01, 2.30e-01, 2.14e-01],
            [2.78e-01, 3.01e-01, 3.37e-01, 3.71e-01, 2.03e-01, 2.26e-01],
        ],
    ]
)
```

```{code-cell} ipython3
:tags: [remove-input]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
ax1, ax2, ax3, ax4 = axes.flatten()

for method_name, data in values_viscosity.items():
    # Plot actual mu
    ax1.plot(tc[:, 1], data[:, 1], label=method_name)

    # Difference from huber for low pressure
    ax3.plot(
        tc[:, 1],
        (values_viscosity["huber"] - data)[:, 0],
        label=method_name,
    )

    # Difference from chen for high pressure
    ax4.plot(
        tc[:, 1],
        (values_viscosity["huber"] - data)[:, 2],
        label=method_name,
    )

# Performance
lines = ax2.plot(N, results_viscosity[1, :, :] / results_viscosity[0, :, :])
ax2.set_xscale("log")
ax2.set_yscale("log")

# Annotation
ax1.legend(frameon=False)
ax1.set_xlabel("Temperature (°C)")
ax2.set_xlabel("Data size")
ax3.set_xlabel("Temperature (°C)")
ax4.set_xlabel("Temperature (°C)")

ax1.set_ylabel(r"Viscosity ($\mu$, Pa s)")
ax2.set_ylabel("Processing time (seconds)")
ax3.set_ylabel(r"Viscosity difference  ($\Delta \mu$, Pa s)")
ax4.set_ylabel(r"Viscosity difference  ($\Delta \mu$, Pa s)")

ax1.text(0.10, 0.20, "A", transform=ax1.transAxes, fontsize=16)
ax1.text(0.05, 0.10, "P = 101325 Pa", transform=ax1.transAxes, ha="left")
ax2.text(0.10, 0.90, "B", transform=ax2.transAxes, fontsize=16)
ax3.text(0.90, 0.90, "C", transform=ax3.transAxes, fontsize=16)
ax3.text(0.95, 0.80, "P = 87000 Pa", transform=ax3.transAxes, ha="right")
ax4.text(0.90, 0.90, "D", transform=ax4.transAxes, fontsize=16)
ax4.text(0.95, 0.80, "P = 108570 Pa", transform=ax4.transAxes, ha="right")

fig.suptitle(
    "Figure 2 - predictions and performance of water viscosity methods", y=-0.02
)
plt.tight_layout()
plt.show()
```

Figure 2A shows the predicted variation in viscosity with temperature. Figure 2B shows
that the most complex method (`huber`) is roughly two orders of magnitude slower than
the fastest method (`vogel`). The differences in predicted viscosity from the most
complex method (Figure 2C,D) are again very small, particularly within reasonable
temperatures (~0-60°C), and the effects of atmospheric pressure are tiny (compare Figure
2C,D).

## Effect on GPP predictions

The code below fits the standard PModel to [example
data](./pmodel/pmodel_details/worked_examples.md#3d-grid-example) using the most complex
and then most efficient implementations. The following plot then shows the difference in
estimated light use efficiency between the two models. The conditions in the example
data cover a wide range of possible environmental conditions and the absolute difference
in light use efficiency arising from using the different implementations is extremely
small.

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
fapar = np.array([1.0])
ppfd = np.array([1.0])

# Convert elevation to atmospheric pressure
patm = calc_patm(elev)

# Mask out temperature values below -25°C
temp[temp < -25] = np.nan

# Clip VPD to force negative VPD to be zero
vpd = np.clip(vpd, 0, np.inf)

# Calculate the photosynthetic environment using different viscosity calculations.
env_high_precision = PModelEnvironment(
    tc=temp,
    co2=co2,
    patm=patm,
    vpd=vpd,
    fapar=fapar,
    ppfd=ppfd,
    core_const=CoreConst(water_density_method="chen", water_viscosity_method="huber"),
)
env_lower_precision = PModelEnvironment(
    tc=temp,
    co2=co2,
    patm=patm,
    vpd=vpd,
    fapar=fapar,
    ppfd=ppfd,
    core_const=CoreConst(
        water_density_method="jones_harris_eq6", water_viscosity_method="vogel"
    ),
)

# Run the P model
model_higher_precision = PModel(env_high_precision)
model_lower_precision = PModel(env_lower_precision)

# Calculate and plot the difference
diff = model_higher_precision.gpp - model_lower_precision.gpp
plt.hist(diff.flatten())
plt.ylabel("Number of values")
_ = plt.xlabel("Difference in light use efficiency (g C mol-1)")
```
