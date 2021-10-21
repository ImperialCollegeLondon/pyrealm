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

# Calculation of optimal chi

Details: {class}`pyrealm.pmodel.CalcOptimalChi`

The next step is to calculate the following factors:

- The optimal ratio of leaf internal to ambient $\ce{CO2}$ partial
  pressure ($\chi = c_i/c_a$).
- The $\ce{CO2}$ limitation term of light use efficiency ($m_j$).
- The limitation term for $V_{cmax}$ ($m_{joc}$). 

The class supports two methods: `prentice14` and `c4`. 

## The `c4` method

This method simply sets all three variables ($\chi$, $m_j$ and $m_{joc}$) to 1,
to reflect the lack of $\ce{CO2}$ limitation in C4 plants.

## The `prentice14` method

This method calculates values for $\chi$ ({cite}`Prentice:2014bc`),  $m_j$
({cite}`Wang:2017go`) and $m_{joc}$ (???). The plots below show how these
parameters change with different environmental inputs.


```{code-cell} python
:tags: [hide-input]
from pyrealm import pmodel
from pyrealm.param_classes import PModelParams
import numpy as np
from matplotlib import pyplot

# Create inputs for a temperature curve at two atmospheric pressures
n_pts = 101
patm_1d = pmodel.calc_patm(np.array([0, 3000]))
tc_1d = np.linspace(0, 30, n_pts)
tc_2d = np.broadcast_to(tc_1d, (2, n_pts))
patm_2d = np.broadcast_to(patm_1d, (n_pts, 2)).transpose()

# Pass those through the intermediate steps to get inputs for CalcOptimalChi
pmodel_param = PModelParams()
gammastar = pmodel.calc_gammastar(tc_2d, patm=patm_2d)
kmm = pmodel.calc_kmm(tc_2d, patm=patm_2d)
viscosity = pmodel.calc_viscosity_h2o(tc_2d, patm=patm_2d)
viscosity_std = pmodel.calc_viscosity_h2o(pmodel_param.k_To, pmodel_param.k_Po)
ns_star = viscosity / viscosity_std

# Compare four scenarios of differing CO2 and VPD
ch = pmodel.calc_co2_to_ca(co2=410, patm=patm_2d)
cl = pmodel.calc_co2_to_ca(co2=280, patm=patm_2d)
optchi_ch_vh = pmodel.CalcOptimalChi(kmm=kmm, gammastar=gammastar, 
                                     ns_star=ns_star, ca=ch, vpd = 1)
optchi_ch_vl = pmodel.CalcOptimalChi(kmm=kmm, gammastar=gammastar, 
                                     ns_star=ns_star, ca=ch, vpd=0.5)
optchi_cl_vh = pmodel.CalcOptimalChi(kmm=kmm, gammastar=gammastar, 
                                     ns_star=ns_star, ca=cl, vpd = 1)
optchi_cl_vl = pmodel.CalcOptimalChi(kmm=kmm, gammastar=gammastar, 
                                     ns_star=ns_star, ca=cl, vpd=0.5)

# Create line plots of optimal chi
pyplot.plot(tc_1d, optchi_ch_vh.chi[0, ], label='0m, 410 ppm, VPD 1')
pyplot.plot(tc_1d, optchi_ch_vh.chi[1, ], label='3000m, 410 ppm, VPD 1')
pyplot.plot(tc_1d, optchi_ch_vl.chi[0, ], label='0m, 410 ppm, VPD 0.5')
pyplot.plot(tc_1d, optchi_ch_vl.chi[1, ], label='3000m, 410 ppm, VPD 0.5')
pyplot.title('Variation in optimal chi')
pyplot.xlabel('Temperature °C')
pyplot.ylabel('Optimal chi')
pyplot.legend()
pyplot.show()

# Create line plots of mj
pyplot.plot(tc_1d, optchi_ch_vh.mj[0, ], label='0m, 410 ppm, VPD 1')
pyplot.plot(tc_1d, optchi_cl_vh.mj[0, ], label='0m, 280 ppm, VPD 1')
pyplot.title('Variation in m_j')
pyplot.xlabel('Temperature °C')
pyplot.ylabel('m_j')
pyplot.legend()
pyplot.show()

# Create line plots of mj
pyplot.plot(tc_1d, optchi_ch_vh.mjoc[0, ], label='0m, 410 ppm, VPD 1')
pyplot.plot(tc_1d, optchi_ch_vh.mjoc[1, ], label='3000m, 410 ppm, VPD 1')
pyplot.plot(tc_1d, optchi_cl_vh.mjoc[0, ], label='0m, 280 ppm, VPD 1')
pyplot.plot(tc_1d, optchi_cl_vh.mjoc[1, ], label='3000m, 280 ppm, VPD 1')
pyplot.title('Variation in m_joc')
pyplot.xlabel('Temperature °C')
pyplot.ylabel('m_joc')
pyplot.legend()
pyplot.show()

```






