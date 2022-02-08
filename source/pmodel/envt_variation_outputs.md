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

    pyplot.show()

```

# Environmental variation in P Model outputs

This page shows how the main output variables from the P Model vary under
differing environmental conditions. The paired plots below show how C3 and C4
plants respond under a range of temperatures (-10°C to 60°C) and then pairs of
values for the other environmental variables:

* Atmospheric pressure: 101325 Pa and 80000 Pa
* Vapour pressure deficit: 500 Pa and 2000 Pa
* $\ce{CO2}$ concentration: 280 ppm and 410 ppm.

All of the pairwise plots share the same legend:

```{code-cell} python
:tags: [hide-input]

fig, ax = pyplot.subplots(1, 1, figsize=(6, 1.2))

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

ax.legend([blnk, blnk, blnk, rd, sld, circ, bl, dsh, trng ], 
            ['patm', 'co2', 'vpd', 
            f"{patm_1d[0]} Pa", f"{co2_1d[0]} ppm", f"{vpd_1d[0]} Pa",
            f"{patm_1d[1]} Pa", f"{co2_1d[1]} ppm", f"{vpd_1d[1]} Pa"
            ], 
        ncol=3, loc='upper center', frameon=False, prop={'size': 12})

ax.axis('off')

pyplot.show()
```

## Efficiency outputs

Two of the key outputs are measures of efficiency and are estimated simply by
creating a {class}`~pyrealm.pmodel.PModel` instance without needing to provide
estimates of absorbed irradiance.

### Light use efficiency (``lue``, LUE)


```{code-cell} python
:tags: [hide-input]
plot_fun('lue', 'LUE')
```

### Water use efficiency (``iwue``, IWUE)


```{code-cell} python
:tags: [hide-input]
plot_fun('iwue', 'IWUE')
```


(estimating-productivity)=
## Estimating productivity

The remaining key outputs are measures of photosynthetic productivity, such as
GPP, which are calculated by providing the P Model with estimates of the
fraction of absorbed photosynthetically active radiation (`fapar`) and the
photosynthetic photon flux density (`ppfd`). The product of these two variables
is an estimate of absorbed irradiance ($I_{abs}$).

The {meth}`~pyrealm.pmodel.PModelEnvironment.estimate_productivity` method is 
used to provide these estimates to the P Model instance. Once this has been run,
the following additional variables are populated:

* Gross primary productivity (``gpp``)
* Dark respiration (``rd``)
* Maximum rate of carboxylation (``vcmax``)
* Maximum rate of carboxylation at standard temperature (``vcmax25``)
* Maximum rate of electron transport. (``jmax``)
* Stomatal conductance (``gs``)

For the plots below, these values have been estimated using typical values for
tropical rainforest:

* $f_{APAR}$: 0.91 (unitless)
* PPFD: 834 $\text{mol}\,m^{-2}\,\text{month}^{-1}$

```{admonition} Units of PPFD
:class: warning

Note that the units of PPFD determine the units of these productivity measures. 
The example here uses PPFD expressed as $\text{mol}\,m^{-2}\,\text{month}^{-1}$: 
GPP is therefore $g\,C\,m^{-2}\text{month}^{-1}$. 
```

If required, productivity estimates per unit absorbed irradiance can be simply
calculated using ``fapar=1, ppfd=1``, which are the default values to
{meth}`~pyrealm.pmodel.PModelEnvironment.estimate_productivity`.

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


## Scaling with absorbed irradiance

All of the six productivity variables scale linearly with absorbed irradiance.
The plots below show how each variable changes, for a constant environment with
`tc` of 20°C, `patm` of 101325 Pa, `vpd` of 1000 Pa and $\ce{CO2}$ of 400 ppm,
when absorbed irradiance changes from 0 to 2000
$\text{mol}\,m^{-2}\,\text{month}^{-1}$.

```{code-cell} python
:tags: [hide-input]

# Calculate the photosynthetic environment 
pmodel_env = pmodel.PModelEnvironment(tc=20, patm=101325, vpd=1000, co2=400)  

# Run the P Models
pmodel_c3 = pmodel.PModel(pmodel_env)
pmodel_c4 = pmodel.PModel(pmodel_env, c4=True)

# Estimate productivity for tropical forest conditions (monthly, m2)
ppfd_vals = np.arange(2000)
pmodel_c3.estimate_productivity(fapar=1, ppfd=ppfd_vals)
pmodel_c4.estimate_productivity(fapar=1, ppfd=ppfd_vals)

def plot_iabs(ax, estvar, estvarlab):
    """Helper function to plot an estimated variable

    Args:
        estvar: String naming variable to be plotted
        estvarlab: String to be used in axis labels
    """

    # Loop over the envnt combinations for c3 and c4 models
    for this_mod, lfmt in zip((pmodel_c3, pmodel_c4), ('r-', 'b-')):

            plotvar = getattr(this_mod, estvar)
            ax.plot(ppfd_vals, plotvar, lfmt)
            
    # Set axis labels
    ax.set_xlabel('Absorbed irradiance (mol m2 month)')
    ax.set_ylabel(f'Estimated {estvarlab}')

fig, axs = pyplot.subplots(2, 3, figsize=(12, 5), sharex=True)

plot_iabs(axs[0,0], 'gpp', 'GPP')
plot_iabs(axs[0,1], 'rd', '$r_d$')
plot_iabs(axs[0,2], 'vcmax', '$v_{cmax}$')
plot_iabs(axs[1,0], 'vcmax25', '$v_{cmax25}$')
plot_iabs(axs[1,1], 'jmax', '$J_{max}$')
plot_iabs(axs[1,2], 'gs', '$g_s$')

axs[0,0].legend([Line2D([], [], linestyle='-', color='r'),
                 Line2D([], [], linestyle='-', color='b')],
                ['C3', 'C4'], loc='upper left', frameon=False)

pyplot.show()
```