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

# The P-Model

This module is used to provide a Python implementation of the P-model and its
corollary predictions \citep{Prentice:2013bc,Wang:2017go}. It is based very 
heavily on the `rpmodel` implementation of the model \cite{Stocker:2020dh} 
but has some differences (see [here](rpmodel) for discussion).
 
## Overview

The P-model takes four core environmental variables and uses these to build a model of carbon capture under a range of scenarios. The four core variables are:

- temperature (`tc`),
- vapor pressure deficit (`vpd`),
- atmospheric CO2 concentration (`co2`)
- atmospheric pressure (`patm`)

If atmospheric pressure is not available then it can be calculated from elevation (`elv`). There are three broad steps in calculating the P-model, shown below with links to details of each step.

1. Environmental determination of key photosynthetic parameters ([Details](chi_inputs)).
2. Calculation of the optimal ratio of internal to ambient $\ce{CO2} partial pressure ([Details][optimal_chi]).
3. Calculation of light use efficiency and maximum carboxylation capacity ([Details][jmax_limitation]).

In addition to these core steps, the P-model can also include:

1. Temperature sensitivity of the instrinsic quantum yield of photosynthesis ($\phi_0$).
2. Inclusion of soil moisture stress, by providing relative soil moisture and local aridity ([Details][soil_moisture]).
3. Imposition of a maximum rate of rubisco regeneration limitation ($J_{max}$).
4. Calculation of absolute values of productivity, by providing the fraction of
   absorbed photosynthetically active radiation ($fA_{PAR}$) and the
   photosynthetic photon flux density (PPFD).

## Variable graph

The graph below shows the model inputs (blue) and internal variables (red) used
in the P-model. Optional inputs and internal variables are shown with a dashed
edge.

![pmodel.svg](pmodel.svg)






In addition to those variables, there are several modifications to the P-model
to allow for greater complexity and temperature dependency in the model
predictions. These cover the following core areas:

.. _iabs_scaling:

Absorbed photosynthetically active radiation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some calculations in the P-model are scaled to a relative measure of absorbed
photosynthetically active radiation. To calculate absolute values, for `gpp`,
`lue`, `rd` and `vcmax` values for both the fraction of absorbed
photosynthetically active radiation (`fapar`) and the photosynthetic photon
flux density (`ppfd`) must be provided to calculate the absolute irradiance
($I_{abs}$).

.. math::

    I_{abs} = \text{fapar} \cdot \text{ppfd}

Note that the units of `ppfd` determine the units of outputs: if `ppfd` is
in mol m-2 month-1, then respective output variables are returned as per unit
months.

Soil moisture stress
^^^^^^^^^^^^^^^^^^^^

The function :func:`calc_soilmstress` provides an empirical estimate of
a soil moisture stress factor based on soil moisture (`soilm`) and
average annual aridity (`meanalpha`). If **both** these values are provided
to :func:`pmodel` then this soil moisture stress factor is calculated and
applied to down-scale light use efficiency (and only light use efficiency),
otherwise it will be set to unity.

.. _kphio:

Temperature dependence of quantum yield efficiency
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The P-model uses a single variable to capture apparent quantum yield
efficiency  (`kphio`). By default, the P-model also incorporates
temperature  dependence of `kphio` using :func:`calc_ftemp_kphio`., but
this can be disabled by setting `do_ftemp_kphio=False`.

The default values of `kphio` vary with the model options, corresponding
to the empirically fitted values presented for three setups in Stocker
et al. (2019) Geosci. Model Dev.

- If `do_ftemp_kphio = False`, then `kphio = 0.049977` (ORG).
- If `do_ftemp_kphio = True` and
    - soil moisture stress is being used `kphio = 0.087182` (FULL),
    - soil moisture stress is not being used `kphio = 0.081785` (BRC).

Photosynthetic pathway
^^^^^^^^^^^^^^^^^^^^^^

The P-model can switch between the C3 or C4 photosynthetic pathways
using the argument `c4`. By default, the C3 pathway is used
(`c4=False`). If `c4=True`, the leaf-internal CO2 concentration is
assumed to be very large (:math:`m \to 1`,  `mj`). 

Limitation of :math:`J_{max}`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:math:`J_{max}` limitation is used to capture temperature dependency in the
maximum rate of RuBP regeneration. Four methods are implemented in
the class :class:`CalcLUEVcmax` and set using the `method_jmaxlim`
argument:

- `wang17` (default, Wang Han et al. 2017)
- `smith19` (Smith et al., 2019)
- `none` (removes :math:`J_{max}` limitation)
- `c4` (a c4 appropriate calculation)

.. TODO c4 may equal none with the appropriate optimal xi - not sure if
 wang17 or smith19 make sense with c4 optimal chi, in which case, don't need
 to have a c4 method at all, just c4 inputs. It looks, from the original text
 in rpmodel that it makes sense to feed c4 OptChi into wang17 (m' -> 0.669),
 in which case it can be retired. But this would be a breaking change, since 
 rpmodel forces c4 to use its own LUEVCmax 

Predictions of the P-model
^^^^^^^^^^^^^^^^^^^^^^^^^^

The P-model calculates the following quantities:

.. _ns_star:

Relative viscosity of water (:math:`\eta^{*}`)
""""""""""""""""""""""""""""""""""""""""""""""

The value :math:`\eta^{*}` (`ns_star`) is the viscosity of water in the
modelled environment relative to the viscosity at standard temperature and
pressure. 

.. math::

    \eta^{*} = \frac{\eta(T, p)}{\eta(T_0, p_0)}

This is used to scale the unit cost of transpiration and :math:`\eta(T, p)` is
calculated by  :func:`calc_density_h20` following Huber et al. (2009).

.. _opt_chi:

Optimal chi
"""""""""""


.. _lue:

Light use efficiency
""""""""""""""""""""

        - `iwue`: Intrinsic water use efficiency (iWUE, Pa), calculated as
                               \deqn{
                                     iWUE = ca (1-\chi)/(1.6)
                               }
        - `gs`: Stomatal conductance (gs, in mol C m-2 Pa-1), calculated as
                               \deqn{
                                    gs = A / (ca (1-\chi))
                               }
                               where \eqn{A} is \code{gpp}\eqn{/Mc}.
        - `vcmax`: Maximum carboxylation capacity \eqn{Vcmax} (mol C m-2) at growth temperature (argument
                              \code{tc}), calculated as
                              \deqn{
                                   Vcmax = \phi(T) \phi0 Iabs n
                              }
                              where \eqn{n} is given by \eqn{n=m'/mc}.
        - `vcmax25`: Maximum carboxylation capacity \eqn{Vcmax} (mol C m-2) normalised to 25 deg C
                             following a modified Arrhenius equation, calculated as \eqn{Vcmax25 = Vcmax / fv},
                             where \eqn{fv} is the instantaneous temperature response by Vcmax and is implemented
                             by function \link{calc_ftemp_inst_vcmax}.
        - `jmax`: The maximum rate of RuBP regeneration () at growth temperature (argument
                              \code{tc}), calculated using
                              \deqn{
                                   A_J = A_C
                              }
        - `rd`: Dark respiration \eqn{Rd} (mol C m-2), calculated as
                             \deqn{
                                 Rd = b0 Vcmax (fr / fv)
                             }
                             where \eqn{b0} is a constant and set to 0.015 (Atkin et al., 2015), \eqn{fv} is the
                             instantaneous temperature response by Vcmax and is implemented by function
                             \link{calc_ftemp_inst_vcmax}, and \eqn{fr} is the instantaneous temperature response
                             of dark respiration following Heskel et al. (2016) and is implemented by function
                             \link{calc_ftemp_inst_rd}.



The :func:`pmodel` function
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. Document the core function

.. automodule:: pyrealm.pmodel
    :members: pmodel


Ancillary functions
--------------------

The remaining functions within the :mod:`pyrealm.pmodel` module are ancillary
functions and classes to calculate various components of the P-model. They are
described below.

.. Document the remaining functions

.. automodule:: pyrealm.pmodel
    :members:
    :exclude-members: pmodel


Bibliography
------------

.. TODO get astrorefs working for less ugly style

.. bibliography:: refs.bib
    :style: unsrt 

.. Including this here to get pymodel.version included in the docs

.. automodule:: pyrealm.version
    :members:


