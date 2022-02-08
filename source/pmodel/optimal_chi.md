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

# Leaf $\ce{CO2}$ and C3 / C4 photosynthesis 


The next step is to calculate the following factors:

- The optimal ratio of leaf internal $\ce{CO2}$ partial pressure ($c_i$, Pa) to
  ambient $\ce{CO2}$ partial pressure ($c_a$, Pa). This value $\chi = c_i/c_a$
  is a unitless ratio.
- The $\ce{CO2}$ limitation term of light use efficiency ($m_j$).
- The limitation term for $V_{cmax}$ ($m_{joc}$). 

The details of these calculations are in {class}`~pyrealm.pmodel.CalcOptimalChi`,
which supports two methods: `prentice14` and `c4`.

## Optimal $\chi$

Both methods (`prentice14` and `c4`) calculate $\chi$ following Equation 8 in 
({cite}`Prentice:2014bc`), but differ in the value used for the unit cost 
ratio parameter ($\beta$).

*  {cite}`Stocker:2020dh` estimated $\beta = 146$ for C3 plants, and this is 
defined as `beta_unit_cost_c3` in {class}`~pyrealm.param_classes.PModelParams`. 

*  Both {cite}`Lin:2015wh` and {cite}`DeKauwe:2015im` provide estimates for the $g_1$ 
parameter for C3 and C4 plants, with a ratio of C3/C4 values of around 3. The
$g_1$ parameter is equivalent to $\xi$ in the P model. Given that 
$\xi \propto \surd\beta$, a reasonable default for C4 plants is that 
$\beta = 146 /  9 \approx 16.222$, defined as  `beta_unit_cost_c4` in 
{class}`~pyrealm.param_classes.PModelParams`.


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

n_pts = 31

tc_1d = np.linspace(-10, 40, n_pts)
patm_1d = np.array([101325, 80000])
vpd_1d = np.array([500, 2000])
co2_1d = np.array([280, 410]) 

tc_4d, patm_4d, vpd_4d, co2_4d = np.meshgrid(tc_1d, patm_1d, vpd_1d, co2_1d)

# Calculate the photosynthetic environment 
pmodel_env = pmodel.PModelEnvironment(tc=tc_4d, patm=patm_4d,vpd=vpd_4d, co2=co2_4d)  

# Run the P Models
pmodel_c3 = pmodel.PModel(pmodel_env)
pmodel_c4 = pmodel.PModel(pmodel_env, c4=True)

# Create line plots of optimal chi

# Create a list of combinations and line formats 
# (line col: PATM, style: CO2, marker used for VPD)

idx_vals = {'vpd': zip([0, 1], vpd_1d), 
            'patm': zip([0, 1], patm_1d), 
            'co2': zip([0, 1], co2_1d)}

idx_combos = list(product(*idx_vals.values())) 
line_formats = ['r-','r--','b-', 'b--'] * 2

# Create side by side subplots
fig, (ax1, ax2) = pyplot.subplots(1, 2, figsize=(12, 5), sharey=True)

# Loop over the combinations for c3 and c4 models
for ((vdx, vvl), (pdx,pvl), (cdx, cvl)), lfmt in zip(idx_combos, line_formats):

    ax1.plot(tc_1d, pmodel_c3.optchi.chi[pdx, :, vdx, cdx], lfmt)
    ax2.plot(tc_1d, pmodel_c4.optchi.chi[pdx, :, vdx, cdx], lfmt)


# Add markers to note the two VPD inputs
for vdx, mrkr in zip([0, 1], ['o', '^']):
    for ax, mod in zip([ax1, ax2], [pmodel_c3, pmodel_c4]):
    
        mean_chi_at_low_end = mod.optchi.chi[:, 0, vdx, :].mean()
        ax.scatter(tc_1d[0] - 2, mean_chi_at_low_end, marker=mrkr, 
                   s=60, c='none', edgecolor='black') 
    

ax1.set_title('C3 variation in optimal $\chi$')
ax2.set_title('C4 variation in optimal $\chi$')

for this_ax in (ax1, ax2):
    this_ax.set_xlabel('Temperature °C')
    this_ax.set_ylabel('Optimal $\chi$')

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

ax2.legend([blnk, blnk, blnk, rd, sld, circ, bl, dsh, trng ], 
           ['patm', 'co2', 'vpd', 
            f"{patm_1d[0]} Pa", f"{co2_1d[0]} ppm", f"{vpd_1d[0]} Pa",
            f"{patm_1d[1]} Pa", f"{co2_1d[1]} ppm", f"{vpd_1d[1]} Pa"
            ], 
           ncol=3, loc='upper left', frameon=False)
pyplot.show()
```

## Limitation terms

The `prentice14` method estimates $m_j$ and $m_{joc}$ following {cite}`Prentice:2014bc`, 
but the `c4` method simply sets $m_j$ and $m_{joc}$ to 1, to reflect the lack 
of $\ce{CO2}$ limitation in C4 plants.


```{code-cell} python
:tags: [hide-input]

# Create side by side subplots for mj, mjoc
fig, (ax1, ax2) = pyplot.subplots(1, 2, figsize=(12, 5))

# Loop over the combinations for c3 and c4 models
for ((vdx, vvl), (pdx,pvl), (cdx, cvl)), lfmt in zip(idx_combos, line_formats):

    ax1.plot(tc_1d, pmodel_c3.optchi.mj[pdx, :, vdx, cdx], lfmt)
    ax2.plot(tc_1d, pmodel_c3.optchi.mjoc[pdx, :, vdx, cdx], lfmt)


# Add markers to note the two VPD inputs
for vdx, mrkr in zip([0, 1], ['o', '^']):
    for ax, mod in zip([ax1, ax2], [pmodel_c3.optchi.mj, pmodel_c3.optchi.mjoc]):
    
        mean_val_at_low_end = mod[:, 0, vdx, :].mean()
        ax.scatter(tc_1d[0] - 2, mean_val_at_low_end, marker=mrkr, 
                   s=60, c='none', edgecolor='black') 
    
for this_ax, var in zip((ax1, ax2), ('$m_{j}$', '$m_{joc}$')):
    this_ax.set_xlabel('Temperature °C')
    this_ax.set_ylabel(var)
    this_ax.set_title('C3 variation in ' + var)

# Add a legend showing the combinations

ax2.legend([blnk, blnk, blnk, rd, sld, circ, bl, dsh, trng ], 
           ['patm', 'co2', 'vpd', 
            f"{patm_1d[0]} Pa", f"{co2_1d[0]} ppm", f"{vpd_1d[0]} Pa",
            f"{patm_1d[1]} Pa", f"{co2_1d[1]} ppm", f"{vpd_1d[1]} Pa"
            ], 
           ncol=3, loc='upper left', frameon=False)

pyplot.show()

```






