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


```{code-cell} python
:tags: [hide-input]
from itertools import product
from pyrealm import pmodel
import numpy as np
from matplotlib import pyplot
from matplotlib.lines import Line2D

# Create inputs for a temperature curve at:
# - two atmospheric pressures
# - two CO2 concentrations
# - two VPD values

n_pts = 1001

tc_1d = np.linspace(-10, 60, n_pts)
patm_1d = np.array([101325, 80000])
vpd_1d = np.array([500, 2000])
co2_1d = np.array([280, 410]) 

tc_4d, patm_4d, vpd_4d, co2_4d = np.meshgrid(tc_1d, patm_1d, vpd_1d, co2_1d)

# Calculate the photosynthetic environment 
pmodel_env = pmodel.PModelEnvironment(tc=tc_4d, patm=patm_4d,vpd=vpd_4d, co2=co2_4d)  

# Run the P Models
pmodel_c3 = pmodel.PModel(pmodel_env)
pmodel_c4 = pmodel.PModel(pmodel_env, c4=True)

# Estimate productivity for tropical forest conditions (monthly, m2)
pmodel_c3.estimate_productivity(fapar=0.91, ppfd=834)
pmodel_c4.estimate_productivity(fapar=0.91, ppfd=834)

# Create line plots of optimal chi

# Create a list of combinations and line formats 
# (line col: PATM, style: CO2, marker used for VPD)

idx_vals = {'vpd': zip([0, 1], vpd_1d), 
            'patm': zip([0, 1], patm_1d), 
            'co2': zip([0, 1], co2_1d)}

idx_combos = list(product(*idx_vals.values())) 
line_formats = ['r-','r--','b-', 'b--'] * 2


def plot_fun(estvar, estvarlab):
    """Helper function to plot an estimated variable

    Args:
        estvar: String naming variable to be plotted
        estvarlab: String to be used in axis labels
    """

    # Create side by side subplots
    fig, (ax1, ax2) = pyplot.subplots(1, 2, figsize=(12, 5), sharey=True, sharex=True)

    # Loop over the envnt combinations for c3 and c4 models
    for this_mod, this_axs in zip((pmodel_c3, pmodel_c4), (ax1, ax2)):

        for ((vdx, vvl), (pdx,pvl), (cdx, cvl)), lfmt in zip(idx_combos, line_formats):

            mrkr = 'o' if vvl == 500 else '^'
            plotvar = getattr(this_mod, estvar)

            this_axs.plot(tc_1d, plotvar[pdx, :, vdx, cdx], lfmt)
            max_idx = plotvar[pdx, :, vdx, cdx].argmax()
            this_axs.scatter(tc_1d[max_idx], 
                              plotvar[pdx, :, vdx, cdx][max_idx],
                              marker=mrkr,  s=60, c='none', edgecolor='black')
        
        # Set axis labels
        this_axs.set_xlabel('Temperature °C')
        this_axs.set_ylabel(f'Estimated {estvarlab}')

    ax1.set_title(f'C3 variation in estimated {estvarlab}')
    ax2.set_title(f'C4 variation in {estvarlab}')

    # create a legend showing the combinations
    blnk = Line2D([], [], color='none')
    rd = Line2D([], [], linestyle='-', color='r')
    bl = Line2D([], [], linestyle='-', color='b')
    sld = Line2D([], [], linestyle='-', color='k')
    dsh = Line2D([], [], linestyle='--', color='k')
    circ = Line2D([], [], marker='o', linestyle='', markersize=10,
                markeredgecolor='k', markerfacecolor='none')
    trng = Line2D([], [], marker='^', linestyle='', markersize=10, 
                markeredgecolor='k', markerfacecolor='none')

    ax1.legend([blnk, blnk, blnk, rd, sld, circ, bl, dsh, trng ], 
            ['patm', 'co2', 'vpd', 
                f"{patm_1d[0]} Pa", f"{co2_1d[0]} ppm", f"{vpd_1d[0]} Pa",
                f"{patm_1d[1]} Pa", f"{co2_1d[1]} ppm", f"{vpd_1d[1]} Pa"
                ], 
            ncol=3, loc='upper left', frameon=False)
    pyplot.show()

```

# Estimating productivity

Measures of photosynthetic productivity, such as GPP, are calculated by
providing the P Model with estimates of the fraction of absorbed
photosynthetically active radiation (`fapar`) and the photosynthetic photon flux
density (`ppfd`). The product of these two variables is an estimate of absorbed
irradiance ($I_{abs}$).

The {meth}`~pyrealm.pmodel.PModelEnvironment.estimate_productivity` method is 
used to provide these estimates to the P Model instance. Once this has been run,
the following additional variables are populated:

* Gross primary productivity (``gpp``)
* Dark respiration (``rd``)
* Maximum rate of carboxylation (``vcmax``)
* Maximum rate of carboxylation at standard temperature (``vcmax25``)
* Maximum rate of electron transport. (``jmax``)
* Stomatal conductance (``gs``)

These variables are now also shown by the {meth}`~pyrealm.pmodel.PModel.summarize` 
method. 

```{code-cell} ipython3
from pyrealm import pmodel
env  = pmodel.PModelEnvironment(tc=20.0, patm=101325.0, vpd=820, co2=400)
model = pmodel.PModel(env)
model.estimate_productivity(fapar=0.91, ppfd=834)
model.summarize()
```

## Units of PPFD

Note that the units of PPFD determine the units of these additional predictions. 
The example above uses representative values for tropical rainforest, with PPFD 
expressed as $\text{mol}\,m^{-2}\,\text{month}^{-1}$: GPP is 
therefore $g\,C\,m^{-2}\text{month}^{-1}$ . 

If required, productivity estimates per unit absorbed irradiance can be simply
calculated using ``fapar=1, ppfd=1``, which are the default values to
{meth}`~pyrealm.pmodel.PModelEnvironment.estimate_productivity`.

```{code-cell} ipython3
model.estimate_productivity() # Per unit I_abs
model.summarize()
```

## Environmental variation in estimated variables

All of these variables scale linearly with $I_{abs}$, so the example plots below
all show how these outputs respond to variation in temperature, atmospheric
pressure, $\ce{CO2}$ concentration and vapour pressure deficit for $f_{APAR}$ = 0.91 
and PPFD = 834 $\text{mol}\,m^{-2}\,\text{month}^{-1}$.

### Gross primary productivity (``gpp``, GPP)


```{code-cell} python
:tags: [hide-input]
plot_fun('gpp', 'GPP')
```

### Dark respiration (``rd``)


```{code-cell} python
:tags: [hide-input]
plot_fun('rd', '$r_d$')
```

### Maximum rate of carboxylation (``vcmax``)


```{code-cell} python
:tags: [hide-input]
plot_fun('vcmax', '$v_{cmax}$')
```

### Maximum rate of carboxylation at standard temperature (``vcmax25``)

```{code-cell} python
:tags: [hide-input]
plot_fun('vcmax25', '$v_{cmax25}$')
```

### Maximum rate of electron transport. (``jmax``)

```{code-cell} python
:tags: [hide-input]
plot_fun('jmax', '$J_{max}$')
```

### Stomatal conductance (``gs``, $g_s$)

```{code-cell} python
:tags: [hide-input]
plot_fun('gs', '$g_s$')
```


