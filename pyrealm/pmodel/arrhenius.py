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

from numpy.typing import NDArray

from pyrealm.core.experimental import warn_experimental
from pyrealm.pmodel.functions import (
    calculate_kattge_knorr_arrhenius_factor,
    calculate_simple_arrhenius_factor,
)
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


class ArrheniusFactorABC(ABC):
    """Abstract base class for implementations of Arrhenius factors.

    This abstract base class provides a framework for implementing Arrhenius
    calculations within a PModel or SubdailyPModel. Individual subclasses are registered
    by a method name that can then be used with the ``method_arrhenius`` argument to
    those classes.

    The ``__init__`` method uses the PModelEnvironment to provide temperature and any
    other required variables to the calculation, along with the reference temperature to
    be used. The ``calculate_arrhenius_factor`` method provides validation to check that
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
    ):
        self.env: PModelEnvironment = env
        """The PModelEnvironment containing the photosynthetic environment for the
        model."""

        # Run the calculation methods after checking for any required variables
        self._check_required_env_variables()

    @abstractmethod
    def _calculation_method(self, coefficients: dict) -> NDArray:
        pass

    def calculate_arrhenius_factor(self, coefficients: dict) -> NDArray:
        """Calculate the Arrhenius factor.

        This method calculates the Arrhenius factor for the model environment, given a
        dictionary providing the required enzyme coefficients.

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
                f"provide: {','.join(sorted(missing_coefficients))}"
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
    """Class providing simple Arrhenius scaling.

    This class provides an implementation of simple Arrhenius scaling for the data in a
    PModelEnvironment. It requires no variables other than the standard temperature and
    requires that a coefficient dictionary providing only ``ha`` (the activation energy
    constant, :math:`H_a`, J/mol).

    Examples:
        >>> import numpy as np
        >>> env = PModelEnvironment(
        ...     tc=np.array([20]),
        ...     patm=np.array([101325]),
        ...     co2=np.array([400]),
        ...     vpd=np.array([1000]),
        ... )
        >>> arrh = SimpleArrhenius(env=env)
        >>> # Simple Arrhenius scaling factor using V_cmax coefficients
        >>> arrh.calculate_arrhenius_factor(
        ...     coefficients={'simple': {'ha': 65330}}
        ... ).round(5)
        array([0.63795])
    """

    def _calculation_method(self, coefficients: dict) -> NDArray:
        return calculate_simple_arrhenius_factor(
            tk=self.env.tk,
            tk_ref=self.env.pmodel_const.tk_ref,
            ha=coefficients["ha"],
            k_R=self.env.core_const.k_R,
        )


class KattgeKnorrArrhenius(
    ArrheniusFactorABC,
    method="kattge_knorr",
    required_coefficients={"ha", "hd", "entropy_intercept", "entropy_slope"},
    required_env_variables=["mean_growth_temperature"],
):
    """Class providing Kattge Knorr Arrhenius scaling.

    This method implements the peaked Arrhenius scaling model of
    :cite:t:`Kattge:2007db`. It requires that the PModelEnvironment also provides values
    for the mean growth temperature of plants as ``mean_growth_temperature`` in °C. It
    also requires a coefficients dictionary providing:

    * the intercept (``entropy_intercept``) and slope (``entropy_slope``) of activation
      entropy as a function of the mean growth temperature in °C (J/mol/°C),
    * the deactivation energy constant  (``hd``, :math:`H_d`, J/mol) and
    * the activation energy constant (``ha``, :math:`H_a`, J/mol).

    Examples:
        >>> import numpy as np
        >>> env = PModelEnvironment(
        ...     tc=np.array([20]),
        ...     patm=np.array([101325]),
        ...     co2=np.array([400]),
        ...     vpd=np.array([1000]),
        ...     mean_growth_temperature=np.array([10]),
        ... )
        >>> arrh = KattgeKnorrArrhenius(env=env)
        >>> # Kattge and Knorr Arrhenius scaling factor using V_cmax coefficients
        >>> arrh.calculate_arrhenius_factor(
        ...     coefficients={"kattge_knorr":
        ...         {
        ...             'entropy_intercept': 668.39,
        ...             'entropy_slope': -1.07,
        ...             'ha': 71513,
        ...             'hd': 200000,
        ...         }
        ...     }
        ...  ).round(5)
        array([0.70109])
    """

    __experimental__ = True

    def _calculation_method(self, coefficients: dict) -> NDArray:
        warn_experimental("KattgeKnorrArrhenius")

        return calculate_kattge_knorr_arrhenius_factor(
            tk_leaf=self.env.tk,
            tk_ref=self.env.pmodel_const.tk_ref,
            tc_growth=getattr(self.env, "mean_growth_temperature"),
            coef=coefficients,
            k_R=self.env.core_const.k_R,
        )
