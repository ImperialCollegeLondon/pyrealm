r"""The module :mod:`~pyrealm.pmodel.quantum_yield` provides
the abstract base class :class:`~pyrealm.pmodel.quantum_yield.QuantumYieldABC`,
which is used to support different implementations of the calculation of the intrinsic
quantum yield of photosynthesis.
"""  # noqa D210, D415

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from warnings import warn

import numpy as np
from numpy.typing import NDArray

from pyrealm import ExperimentalFeatureWarning
from pyrealm.constants import CoreConst, PModelConst
from pyrealm.core.utilities import (
    check_input_shapes,
    evaluate_horner_polynomial,
    summarize_attrs,
)
from pyrealm.pmodel.functions import calc_modified_arrhenius_factor
from pyrealm.pmodel.pmodel_environment import PModelEnvironment

QUANTUM_YIELD_CLASS_REGISTRY: dict[str, type[QuantumYieldABC]] = {}
r"""A registry for intrinsic quantum yield of photosynthesis calculation classes.

Different implementations of the calculation of the intrinsic quantum yield of
photosynthesis (:math:`\phi_{0}`) must all be subclasses of
:class:`~pyrealm.pmodel.quantum_yield.QuantumYieldABC` abstract base class. This
dictionary is used as a registry for defined subclasses and a method name is used to
retrieve a particular implementation from this registry. For example:

.. code:: python

    bianchi_phi_0 = QUANTUM_YIELD_CLASS_REGISTRY['bianchi']
"""


class QuantumYieldABC(ABC):
    r"""ABC for calculating the intrinsic quantum yield of photosynthesis.

    This provides an abstract base class for the implementation of alternative
    approaches to calculating the the intrinsic quantum yield of photosynthesis. All
    implementations estimate the :math:`\phi_{0} following values, which is then stored
    in the ``kphio`` attribute of the resulting class instance.

    The abstract base class requires that implementations of specific approaches defines
    the `calculate_kphio` method. The provides the approach specific calculation of
    ``kphio``  and is automatically called by the ``__init__`` method when a subclass
    instance is created.

    Args:
        env: An instance of
            :class:`~pyrealm.pmodel.pmodel_environment.PModelEnvironment`  providing the
            photosynthetic environment for the model.
        pmodel_const: An instance of
            :class:`~pyrealm.constants.pmodel_const.PModelConst`.

    Returns:
        Instances of the abstract base class should not be created - use instances of
        specific subclasses.
    """

    method: str
    """A short method name used to identify the class in
    :data:`~pyrealm.pmodel.quantum_yield.QUANTUM_YIELD_CLASS_REGISTRY`.
    """
    is_c4: bool
    """A flag indicating if the method captures the C4 photosynthetic pathway."""
    requires: list[str]
    """A list of names of optional attributes of
    :class:`~pyrealm.pmodel.pmodel_environment.PModelEnvironment` that must be populated
    to use a method.
    """

    def __init__(
        self,
        env: PModelEnvironment,
        pmodel_const: PModelConst = PModelConst(),
        core_const: CoreConst = CoreConst(),
    ):
        self.env: PModelEnvironment = env
        """The PModelEnvironment containing the photosynthetic environment for the
        model."""
        self.shape: tuple[int, ...] = env.shape
        """The shape of the input environment data."""
        self.pmodel_const: PModelConst = pmodel_const
        """The PModelConst used for calculating quantum yield"""
        self.core_const: CoreConst = core_const
        """The CoreConst used for calculating quantum yield"""

        # Declare attributes populated by methods. These are typed but not assigned a
        # default value as they must are populated by the subclass specific
        # calculate_kphio method, which is called below to populate the values.
        self.kphio: NDArray
        """The intrinsic quantum yield of photosynthesis."""

        # Run the calculation methods after checking for any required variables
        self._check_requires()
        self._calculate_kphio()

        # Validate that the subclass methods populate the attributes correctly.
        _ = check_input_shapes(env.ca, self.kphio)

    @abstractmethod
    def _calculate_kphio(self, **kwargs: Any) -> None:
        """Calculate the intrinsic quantum yield of photosynthesis."""

    def _check_requires(self) -> None:
        """Check additional required variables are present."""

        for required_var in self.requires:
            if getattr(self.env, required_var) is None:
                raise ValueError(
                    f"{self.__class__.__name__} (method {self.method}) requires "
                    f"{required_var} to be provided in the PModelEnvironment."
                )

    def __repr__(self) -> str:
        """Generates a string representation of an QuantumYield instance."""
        return f"{type(self).__name__}(shape={self.shape})"

    def summarize(self, dp: int = 2) -> None:
        """Print QuantumYield summary.

        Prints a summary of the variables calculated within an instance
        of QuantumYield including the mean, range and number of nan values.

        Args:
            dp: The number of decimal places used in rounding summary stats.
        """

        attrs = [("kphio", "-")]

        summarize_attrs(self, attrs, dp=dp)

    @classmethod
    def __init_subclass__(cls, method: str, is_c4: bool, requires: list[str]) -> None:
        """Initialise a subclass deriving from this ABC."""

        cls.method = method
        cls.is_c4 = is_c4
        cls.requires = requires
        QUANTUM_YIELD_CLASS_REGISTRY[cls.method] = cls


class QuantumYieldConstant(
    QuantumYieldABC,
    method="constant",
    is_c4=False,
    requires=[],
):
    """Constant kphio."""

    def _calculate_kphio(self, **kwargs: Any) -> None:
        """Constant kphio."""

        if "init_kphio" not in kwargs:
            raise ValueError("Missing definition of initial kphio.")

        self.kphio = np.array([kwargs["init_kphio"]])


class QuantumYieldBernacchiC3(
    QuantumYieldABC,
    method="bernacchi_c3",
    is_c4=False,
    requires=[],
):
    """Calculate kphio following Bernacchi for C3 plants."""

    def _calculate_kphio(self, **kwargs: Any) -> None:
        """Calculate kphio."""

        if "init_kphio" not in kwargs:
            raise ValueError("Missing definition of constant kphio.")

        ftemp = evaluate_horner_polynomial(self.env.tc, self.pmodel_const.kphio_C3)
        ftemp = np.clip(ftemp, 0.0, None)

        self.kphio = ftemp * kwargs["init_kphio"]


class QuantumYieldBernacchiC4(
    QuantumYieldABC,
    method="bernacchi_c4",
    is_c4=True,
    requires=[],
):
    """Calculate kphio following Bernacchi."""

    def _calculate_kphio(self, **kwargs: Any) -> None:
        """Calculate kphio."""

        if "init_kphio" not in kwargs:
            raise ValueError("Missing definition of constant kphio.")

        ftemp = evaluate_horner_polynomial(self.env.tc, self.pmodel_const.kphio_C4)
        ftemp = np.clip(ftemp, 0.0, None)

        self.kphio = ftemp * kwargs["init_kphio"]


class QuantumYieldSandoval(
    QuantumYieldABC,
    method="sandoval",
    is_c4=False,
    requires=["aridity_index", "mean_growth_temperature"],
):
    """Calculate kphio following Sandoval."""

    def _calculate_kphio(self, **kwargs: Any) -> None:
        """Constant kphio."""

        # Warn that this is an experimental feature.
        warn(
            "The sandoval method for calculating kphi0 is experimental, "
            "see the class documentation",
            ExperimentalFeatureWarning,
        )

        # Calculate activation entropy as a function of mean growth temperature, J/mol/K
        deltaS = 1558.853 - 50.223 * self.env.mean_growth_temperature
        # Calculate deaactivation energy J/mol
        Hd = 294.804 * deltaS
        # activation energy J/mol
        Ha = 75000.0

        # theoretical maximum phi0 and curvature parameters (Long, 1993;Sandoval et al.,
        # in.prep.)
        phi_o_theo = 0.111
        m = 6.8681
        n = 0.07956432

        # Calculate the optimal temperature to be used as the reference temperature in
        # the modified Arrhenius calculation
        Topt = Hd / (deltaS - self.core_const.k_R * np.log(Ha / (Hd - Ha)))

        # Calculate peak kphio given the aridity index
        kphio_peak = phi_o_theo / (1 + (self.env.aridity_index) ** m) ** n

        # Calculate the modified Arrhenius factor using the
        f_kphio = calc_modified_arrhenius_factor(
            tk=self.env.tc + self.core_const.k_CtoK,
            Ha=Ha,
            Hd=Hd,
            deltaS=deltaS,
            mode="J1942",
            tk_ref=Topt,
        )

        # Apply the factor and store it.
        self.kphio = kphio_peak * f_kphio
