"""Arrhenius enzyme reaction models.

The reaction rates of enzyme systems of photosynthesis respond to temperature in a way
that is estimate using an Arrhenius relationship. There are alternative ways of
parameterizing this temperature scaling and this module defines functions implementing
different approaches to calculating Arrhenius factors for a given temperature, relative
to a reference temperature, given a set of coefficients for a specific enzyme system.

It also provides the :class:`pyrealm.pmodel.arrhenius.ArrheniusFactorABC` abstract base
class. This provides a framework that allows different scaling functions to be called
from the PModel and SubdailyPModel classes, and hence allows the models to use different
temperature relationships for enzyme rates.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from pyrealm.constants.core_const import CoreConst
from pyrealm.pmodel.pmodel_environment import PModelEnvironment

ARRHENIUS_METHOD_REGISTRY: dict[str, type[ArrheniusFactorABC]] = {}
"""A registry for Arrhenius factor calculation classes.

This dictionary is used as a registry for subclasses of the
:class:`~pyrealm.pmodel.arrhenius.ArrheniusFactorABC` abstract base class. The different
subclasses are keyed by a method name that can be used to retrieve a particular
implementation from this registry. For example: 

.. code:: python

    arrh_simple = ARRHENIUS_METHOD_REGISTRY['simple']
"""


def calculate_simple_arrhenius_factor(
    tk: NDArray[np.float64],
    tk_ref: float,
    ha: float,
    core_const: CoreConst = CoreConst(),
) -> NDArray[np.float64]:
    r"""Calculate an Arrhenius scaling factor using activation energy.

    Calculates the temperature-scaling factor :math:`f` for enzyme kinetics following
    a simple Arrhenius response governed solely by the activation energy for an enzyme
    (``ha``, :math:`H_a`). The rate is given for a temperature :math:`T` relative to a
    reference temperature :math:T_0`, both given in Kelvin.

    Arrhenius kinetics are described as:

    .. math::

        x(T) = \exp(c - H_a / (T R))

    The temperature-correction function :math:`f(T, H_a)` is:

      .. math::
        :nowrap:

        \[
            \begin{align*}
                f &= \frac{x(T)}{x(T_0)} \\
                  &= \exp \left( \frac{ H_a (T - T_0)}{T_0 R T}\right)
                        \text{, or equivalently}\\
                  &= \exp \left( \frac{ H_a}{R} \cdot
                        \left(\frac{1}{T_0} - \frac{1}{T}\right)\right)
            \end{align*}
        \]

    Args:
        tk: Temperature (K)
        tk_ref: The reference temperature for the reaction (K).
        ha: Activation energy (in :math:`J \text{mol}^{-1}`)
        core_const: Instance of :class:`~pyrealm.constants.core_const.CoreConst`.

    PModel Parameters:
        R: the universal gas constant (:math:`R`, ``k_R``)

    Returns:
        Estimated float values for :math:`f`

    Examples:
        >>> # Relative rate change from 25 to 10 degrees Celsius (percent change)
        >>> np.round((1.0-calc_ftemp_arrh( 283.15, 100000)) * 100, 4)
        array([88.1991])
    """

    return np.exp(ha * (tk - tk_ref) / (tk_ref * core_const.k_R * tk))


def calculate_kattge_knorr_arrhenius_factor(
    tk_leaf: NDArray[np.float64],
    tk_ref: float,
    tc_growth: NDArray[np.float64],
    ha: float,
    hd: float,
    entropy_intercept: float,
    entropy_slope: float,
    core_const: CoreConst = CoreConst(),
    mode: str = "M2002",
) -> NDArray[np.float64]:
    r"""Calculate an Arrhenius factor following :cite:t:`Kattge:2007db`.

    This implements a "peaked" version of the Arrhenius relationship, describing a
    decline in reaction rates at higher temperatures. In addition to the activation
    energy (see :meth:`~pyrealm.pmodel.arrhenius.calculate_simple_arrhenius_factor`),
    this implementation adds an entropy term and the deactivation energy of the enzyme
    system. The rate is given for a given instantaneous temperature :math:`T` relative
    to a reference temperature :math:T_0`, both given in Kelvin, but the entropy is
    calculated using a separate estimate of the growth temperature for a plant,
    expressed in °C.


    .. math::
        :nowrap:

        \[
            \begin{align*}

                f  &= \exp \left( \frac{ H_a (T - T_0)}{T_0 R T}\right)
                      \left(
                        \frac{1 + \exp \left( \frac{T_0 \Delta S - H_d }{ R T_0}\right)}
                             {1 + \exp \left( \frac{T \Delta S - H_d}{R T} \right)}
                      \right)
                      \left(\frac{T}{T_0}\right)
            \end{align*}

            \text{where,}

            \Delta S = a + b * t_g

        \]



    The function can operate in one of two modes (``M2002`` or ``J1942``) using
    alternative derivations of the modified Arrhenius relationship presented in
    :cite:t:`murphy:2021a`. The ``J1942`` includes an additional factor (tk/tk_ref) that
    is ommitted from the simpler ``M2002`` derivation.

    Args:
        tk_leaf: The instantaneous temperature in Kelvin (K) at which to calculate the
            factor (:math:`T`)
        tk_ref: The reference temperature in Kelvin for the process (:math:`T_0`)
        tc_growth: The growth temperature of the plants in °C (:math:`t_g`)
        ha: The activation energy of the enzyme (:math:`H_a`)
        hd: The deactivation energy of the enzyme (:math:`H_d`)
        entropy_intercept: The intercept of the entropy relationship (:math:`a`),
        entropy_slope: The slope of the entropy relationship (:math:`b`),
        core_const: Instance of :class:`~pyrealm.constants.core_const.CoreConst`.
        mode: The calculation mode.

    PModel Parameters:
        R: The universal gas constant (:math:`R`, ``k_R``)

    Returns:
        Values for :math:`f`

    Examples:
        >>> # Calculate the factor for the relative rate of V_cmax at 10 degrees
        >>> # compared to the rate at the reference temperature of 25°C.
        >>> from pyrealm.constants import PModelConst
        >>> pmodel_const = PModelConst()
        >>> # Get enzyme kinetics parameters
        >>> a, b, ha, hd = pmodel_const.kattge_knorr_kinetics
        >>> # Calculate entropy as a function of temperature _in °C_
        >>> deltaS = a + b * 10
        >>> # Calculate the arrhenius factor
        >>> val = calc_modified_arrhenius_factor(
        ...     tk= 10 + 273.15, Ha=ha, Hd=hd, deltaS=deltaS, tk_ref=25 +273.15
        ... )
        >>> np.round(val, 4)
        np.float64(0.261)
    """

    if mode not in ["M2002", "J1942"]:
        raise ValueError(
            f"Unknown mode option for calc_modified_arrhenius_factor: {mode}"
        )

    # Calculate entropy as a function of temperature _in °C_
    entropy = entropy_intercept + entropy_slope * tc_growth

    # Calculate Arrhenius components
    fva = calculate_simple_arrhenius_factor(tk=tk_leaf, ha=ha, tk_ref=tk_ref)

    fvb = (1 + np.exp((tk_ref * entropy - hd) / (core_const.k_R * tk_ref))) / (
        1 + np.exp((tk_leaf * entropy - hd) / (core_const.k_R * tk_leaf))
    )

    if mode == "M2002":
        # Medlyn et al. 2002 simplification
        return fva * fvb

    # Johnson et al 1942
    return fva * (tk_leaf / tk_ref) * fvb


class ArrheniusFactorABC(ABC):
    """Abstract base class for implementations of Arrhenius factors.

    This abstract base class provides a framework for implementing Arrhenius
    calculations within a PModel or SubdailyPModel. Individual subclasses are registered
    by a method name that can then be used with the ``arrhenius_method`` argument to
    those classes.

    The `__init__` method uses the PModelEnvironment to provide temperature and any
    other required variable to the calculation, along with the reference temperature to
    be used. The `calculate_arrhenius_factor` method provides validation to check that
    the coefficients required by a particular implementation are provided.

    Subclasses only need to implement the private abstract method `_calculate_values`,
    which should implement the actual calculation from the data and coefficients
    supplied to the instance, and should return an array of calculated factor values.
    """

    method: str
    """A short method name used to identify the class in
    :data:`~pyrealm.pmodel.arrhenius.ARRHENIUS_METHOD_REGISTRY`. This name is also used
    to select a matching dictionary of coefficients from inputs.
    """
    required_coefficients: set[str]
    """A set of the names of coefficients required to calculate factor values. These
    must be present in the coefficients dictionary matching the method name."""

    required_env_variables: list[str]
    """A list of names of optional attributes of
    :class:`~pyrealm.pmodel.pmodel_environment.PModelEnvironment` that must be populated
    to use a method.
    """

    def __init__(
        self,
        env: PModelEnvironment,
        reference_temperature: float,
        core_const: CoreConst = CoreConst(),
    ):
        self.env: PModelEnvironment = env
        """The PModelEnvironment containing the photosynthetic environment for the
        model."""

        self.core_const = core_const
        """The core constants to be used in the calculation"""

        self.tk_ref = reference_temperature + self.core_const.k_CtoK
        """The reference temperature in Kelvins, calculated internally."""

        self.tk = self.env.tc + self.core_const.k_CtoK
        """The temperature in Kelvings, calculated internally from the temperature in
        the `env` argument."""

        # Declare attributes populated by methods.
        self.arrhenius_factor: NDArray[np.float64]
        """The calculated Arrhenius factor."""

        # Run the calculation methods after checking for any required variables
        self._check_required_env_variables()

    @abstractmethod
    def _calculation_method(self, coefficients: dict) -> NDArray:
        pass

    def calculate_arrhenius_factor(self, coefficients: dict) -> NDArray:
        """Calculate the Arrhenius factor.

        Args:
            coefficients: A dictionary providing any required coefficients for a given
                enzyme system.

        Raises:
            ValueError: where the method name is not found in the coefficients
                dictionary or the required coefficents are not found in that matched
                dictionary.
        """

        # Check the coefficients dictionary provides an entry for this method
        if self.method not in coefficients:
            raise ValueError(
                f"The coefficients dict does not provide a parameterisation "
                f"for the {self.method} Arrhenius method."
            )

        # Check the required coefficients are found
        missing_coefficients = self.required_coefficients.difference(
            coefficients[self.method]
        )
        if missing_coefficients:
            raise ValueError(
                f"The coefficients for the {self.method} Arrhenius method do not "
                f"provide: {','.join(missing_coefficients)}"
            )

        return self._calculation_method(coefficients=coefficients[self.method])

    def _check_required_env_variables(self) -> None:
        """Check additional required variables are present."""

        for required_var in self.required_env_variables:
            if not hasattr(self.env, required_var):
                raise ValueError(
                    f"{self.__class__.__name__} (method {self.method}) requires "
                    f"{required_var} to be provided in the PModelEnvironment."
                )

    @classmethod
    def __init_subclass__(
        cls,
        method: str,
        required_coefficients: set[str],
        required_env_variables: list[str],
    ) -> None:
        """Initialise a subclass deriving from this ABC."""

        cls.method = method
        cls.required_coefficients = required_coefficients
        cls.required_env_variables = required_env_variables

        ARRHENIUS_METHOD_REGISTRY[cls.method] = cls


class SimpleArrhenius(
    ArrheniusFactorABC,
    method="simple",
    required_coefficients={"ha"},
    required_env_variables=[],
):
    """Class providing simple Arrhenius scaling."""

    def _calculation_method(self, coefficients: dict) -> NDArray:
        return calculate_simple_arrhenius_factor(
            tk=self.tk,
            tk_ref=self.tk_ref,
            ha=coefficients["ha"],
            core_const=self.core_const,
        )


class KattgeKnorrArrhenius(
    ArrheniusFactorABC,
    method="kattge_knorr",
    required_coefficients={"ha", "hd", "entropy_intercept", "entropy_slope"},
    required_env_variables=[],
):
    """Class providing Kattge Knorr Arrhenius scaling."""

    def _calculation_method(self, coefficients: dict) -> NDArray:
        return calculate_kattge_knorr_arrhenius_factor(
            tk_leaf=self.tk,
            tk_ref=self.tk_ref,
            tc_growth=self.env.tc,
            ha=coefficients["ha"],
            hd=coefficients["hd"],
            entropy_intercept=coefficients["entropy_intercept"],
            entropy_slope=coefficients["entropy_slope"],
            core_const=self.core_const,
        )
