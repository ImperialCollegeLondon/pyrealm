.. _pmodel:

The P-Model
===========


This module is used to provide a Python implementation of the P-model and its
corollary predictions (Prentice et al., 2014; Han et al., 2017).

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
- vapor presssure deficit (``vpd``),
- atmospheric CO2 concentration (``co2``)
- atmospheric pressure (``patm``)

If atmospheric pressure is not available then elevation (``elv``) can be 
provided instead and  will be used to calculate `patm` via :func:`calc_patm`.

In addition to those variables, there are several modifications to the P-model
to allow for greater complexity and temperature dependency in the model
predictions. These cover the following core areas:

Absorbed photosynthetically active radiation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some calculations in the P-model are scaled to a relative measure of absorbed
photosynthetically active radiation. To calculate absolute values, for ``gpp``,
``lue``, ``rd`` and ``vcmax`` values for both the fraction of absorbed
photosynthetically active radiation (``fapar``) and the photosynthetic photon
flux density (``ppfd``) must be provided to calculate the scaling factor
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
- If ``do_ftemp_kphio = True`` and:
    - soil moisture stress is being used ``kphio = 0.087182`` (FULL),
    - soil moisture stress is not being used ``kphio = 0.081785`` (BRC).

Photosynthetic pathway
^^^^^^^^^^^^^^^^^^^^^^

The P-model can switch between the C3 or C4 photosynthetic pathways
using the argument ``c4``. By default, the C3 pathway is used
(``c4=False``). If ``c4=True``, the leaf-internal CO2 concentration is
assumed to be very large, :math:`m \to 1` (returned variable ``mj``)
and :math:`m' \to 0.669` (with ``c = 0.41``).

.. TODO c = 0.41?

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
 wang17 or smith19 make sense with c4 optimal xi, in which case, don't need
 to have a c4 method at all, just c4 inputs.

Predictions of the P-model
^^^^^^^^^^^^^^^^^^^^^^^^^^

The P-model calculates the following quantities:

.. _ns_star:

Relative viscosity of water (:math:`\eta^{*}`)
"""""""""""""""""""""""""""""""""""""""""""""

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

Light use efficiency (LUE) is calculated as:

.. math::
    \text{LUE} = \phi(T) \cdot \phi0 \cdot m' \cdot M_C

Where:

- :math:`\phi(T)` is the temperature-dependent quantum yield efficiency modifier
  (see :ref:`kphio`). If ``do_ftemp_kphio=False`` then :math:`\phi(T) = 1`,
- :math:`\phi 0}` is given by argument ``kphio``,
- :math:`M_C` is the molecular mass of Carbon,
                                \eqn{m'=m} if \code{method_jmaxlim=="none"}, otherwise
                                \deqn{
                                       m' = m \sqrt( 1 - (c/m)^(2/3) )
                                }
                                with \eqn{c=0.41} (Wang et al., 2017) if \code{method_jmaxlim=="wang17"}. \eqn{Mc} is
                                the molecular mass of C (12.0107 g mol-1). \eqn{m} is given returned variable \code{mj}.
                                If \code{do_soilmstress==TRUE}, \eqn{LUE} is multiplied with a soil moisture stress factor,
                                calculated with \link{calc_soilmstress}.



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




.. Including this here to get pymodel.version included in the docs
.. automodule:: pypmodel.version
    :members:
