.. _pmodel:

The P-Model
===========

This module is used to provide a Python implementation of the P-model and its
corollary predictions \citep{Prentice:2013bc,Wang:2017go} .

The ``rpmodel`` package
-----------------------

This implementation is very heavily based on Benjamin Stocker's R implementation
of the P-model:

https://github.com/stineb/rpmodel/tree/master/R

The ``rpmodel`` package has been used to calculate the expected values used
in the module docstrings and which are used to provide :mod:`doctest` tests of
the basic operation of this implementation.

The main differences are:

- Many of hard-coded parameters within functions have been consolidated into
  a global parameter list (:const:`pypmodel.params.PARAM`, see
  :ref:`parameterisation`). This is used to provide values to all functions and
  is also editable, to allow users to explore model responses to parameter
  variation.
- The ``rpmodel`` package has suites of functions for calculating
  :math:`J_{max}` limitation and optimal :math:`\chi`. These have been combined
  into classes :class:`CalcLUEVcmax` and :class:`CalcOptimalChi` that share
  common parameters and outputs and which provide a ``method`` argument to
  switch between approaches.
- Some of the functions have shorter argument lists, either because the
  parameter in question is now defined in :const:`pmodel.PARAM` or because code
  reorganisation for Python has allowed for a simpler interface.

Description of the :func:`pmodel` 
---------------------------------

The core function is :func:`pmodel` and the core arguments to the model are:

- the temperature (``tc``),
- vapor pressure deficit (``vpd``),
- atmospheric CO2 concentration (``co2``)
- atmospheric pressure (``patm``)

.. TODO Vectorisation: if all of these are arrays with identical shape (or are
 scalars that can be broadcast to match an otherwise consistent array shape) then this could
 generate a time series or spatial surface using numpy. 

If atmospheric pressure is not available then elevation (``elv``) can be 
provided instead and  will be used to calculate `patm` via :func:`calc_patm`.

In addition to those variables, there are several modifications to the P-model
to allow for greater complexity and temperature dependency in the model
predictions. These cover the following core areas:

.. _iabs_scaling:

Absorbed photosynthetically active radiation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some calculations in the P-model are scaled to a relative measure of absorbed
photosynthetically active radiation. To calculate absolute values, for ``gpp``,
``lue``, ``rd`` and ``vcmax`` values for both the fraction of absorbed
photosynthetically active radiation (``fapar``) and the photosynthetic photon
flux density (``ppfd``) must be provided to calculate the absolute irradiance
(:math:`I_{abs}`).

.. math::

    I_{abs} = \text{fapar} \cdot \text{ppfd}

Note that the units of ``ppfd`` determine the units of outputs: if ``ppfd`` is
in mol m-2 month-1, then respective output variables are returned as per unit
months.

Soil moisture stress
^^^^^^^^^^^^^^^^^^^^

The function :func:`calc_soilmstress` provides an empirical estimate of
a soil moisture stress factor based on soil moisture (``soilm``) and
average annual aridity (``meanalpha``). If **both** these values are provided
to :func:`pmodel` then this soil moisture stress factor is calculated and
applied to down-scale light use efficiency (and only light use efficiency),
otherwise it will be set to unity.

.. _kphio:

Temperature dependence of quantum yield efficiency
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The P-model uses a single variable to capture apparent quantum yield
efficiency  (``kphio``). By default, the P-model also incorporates
temperature  dependence of ``kphio`` using :func:`calc_ftemp_kphio`., but
this can be disabled by setting ``do_ftemp_kphio=False``.

The default values of ``kphio`` vary with the model options, corresponding
to the empirically fitted values presented for three setups in Stocker
et al. (2019) Geosci. Model Dev.

- If ``do_ftemp_kphio = False``, then ``kphio = 0.049977`` (ORG).
- If ``do_ftemp_kphio = True`` and
    - soil moisture stress is being used ``kphio = 0.087182`` (FULL),
    - soil moisture stress is not being used ``kphio = 0.081785`` (BRC).

Photosynthetic pathway
^^^^^^^^^^^^^^^^^^^^^^

The P-model can switch between the C3 or C4 photosynthetic pathways
using the argument ``c4``. By default, the C3 pathway is used
(``c4=False``). If ``c4=True``, the leaf-internal CO2 concentration is
assumed to be very large (:math:`m \to 1`,  ``mj``). 

Limitation of :math:`J_{max}`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:math:`J_{max}` limitation is used to capture temperature dependency in the
maximum rate of RuBP regeneration. Four methods are implemented in
the class :class:`CalcLUEVcmax` and set using the ``method_jmaxlim``
argument:

- ``wang17`` (default, Wang Han et al. 2017)
- ``smith19`` (Smith et al., 2019)
- ``none`` (removes :math:`J_{max}` limitation)
- ``c4`` (a c4 appropriate calculation)

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

The value :math:`\eta^{*}` (``ns_star``) is the viscosity of water in the
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

        - ``iwue``: Intrinsic water use efficiency (iWUE, Pa), calculated as
                               \deqn{
                                     iWUE = ca (1-\chi)/(1.6)
                               }
        - ``gs``: Stomatal conductance (gs, in mol C m-2 Pa-1), calculated as
                               \deqn{
                                    gs = A / (ca (1-\chi))
                               }
                               where \eqn{A} is \code{gpp}\eqn{/Mc}.
        - ``vcmax``: Maximum carboxylation capacity \eqn{Vcmax} (mol C m-2) at growth temperature (argument
                              \code{tc}), calculated as
                              \deqn{
                                   Vcmax = \phi(T) \phi0 Iabs n
                              }
                              where \eqn{n} is given by \eqn{n=m'/mc}.
        - ``vcmax25``: Maximum carboxylation capacity \eqn{Vcmax} (mol C m-2) normalised to 25 deg C
                             following a modified Arrhenius equation, calculated as \eqn{Vcmax25 = Vcmax / fv},
                             where \eqn{fv} is the instantaneous temperature response by Vcmax and is implemented
                             by function \link{calc_ftemp_inst_vcmax}.
        - ``jmax``: The maximum rate of RuBP regeneration () at growth temperature (argument
                              \code{tc}), calculated using
                              \deqn{
                                   A_J = A_C
                              }
        - ``rd``: Dark respiration \eqn{Rd} (mol C m-2), calculated as
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

.. automodule:: pypmodel.pmodel
    :members: pmodel


Ancillary functions
--------------------

The remaining functions within the :mod:`pypmodel.pmodel` module are ancillary
functions and classes to calculate various components of the P-model. They are
described below.

.. Document the remaining functions

.. automodule:: pypmodel.pmodel
    :members:
    :exclude-members: pmodel


Bibliography
------------

.. TODO get astrorefs working for less ugly style

.. bibliography:: refs.bib
    :style: unsrt 

.. Including this here to get pymodel.version included in the docs

.. automodule:: pypmodel.version
    :members:


