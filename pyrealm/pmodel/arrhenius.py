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
    required_env_variables=["growth_temperature"],
):
    """Class providing Kattge Knorr Arrhenius scaling."""

    def _calculation_method(self, coefficients: dict) -> NDArray:
        return calculate_kattge_knorr_arrhenius_factor(
            tk_leaf=self.tk,
            tk_ref=self.tk_ref,
            tc_growth=self.env.mean_growth_temperature,
            ha=coefficients["ha"],
            hd=coefficients["hd"],
            entropy_intercept=coefficients["entropy_intercept"],
            entropy_slope=coefficients["entropy_slope"],
            core_const=self.core_const,
        )
