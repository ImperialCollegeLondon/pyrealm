r"""The module :mod:`~pyrealm.pmodel.quantum_yield` provides
the abstract base class :class:`~pyrealm.pmodel.quantum_yield.QuantumYieldABC`, which is
used to support different implementations of the calculation of the intrinsic quantum
yield efficiency of photosynthesis (:math:`\phi_0`, unitless). Note that :math:`\phi_0`
is sometimes used to refer to the quantum yield of electron transfer, which is exactly
four times larger, so check definitions here.
"""  # noqa D210, D415

from __future__ import annotations

from abc import ABC, abstractmethod
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
    requires: list[str]
    """A list of names of optional attributes of
    :class:`~pyrealm.pmodel.pmodel_environment.PModelEnvironment` that must be populated
    to use a method.
    """
    default_reference_kphio: float
    """A default value for the reference kphio value for use with a given
    implementation."""

    def __init__(
        self,
        env: PModelEnvironment,
        reference_kphio: float | None = None,
        use_c4: bool = False,
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
        self.reference_kphio: float = reference_kphio or self.default_reference_kphio
        """The reference value for kphio for the method."""
        self.use_c4: bool = use_c4
        """Use a C4 parameterisation if available."""

        # Declare attributes populated by methods. These are typed but not assigned a
        # default value as they must are populated by the subclass specific
        # calculate_kphio method, which is called below to populate the values.
        self.kphio: NDArray
        """The calculated intrinsic quantum yield of photosynthesis."""

        # Run the calculation methods after checking for any required variables
        self._check_requires()
        self._calculate_kphio()

        # Validate that the subclass methods populate the attributes correctly.
        _ = check_input_shapes(env.ca, self.kphio)

    @abstractmethod
    def _calculate_kphio(self) -> None:
        """Calculate the intrinsic quantum yield of photosynthesis."""

    def _check_requires(self) -> None:
        """Check additional required variables are present."""

        for required_var in self.requires:
            if not hasattr(self.env, required_var):
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
    def __init_subclass__(
        cls,
        method: str,
        requires: list[str],
        default_reference_kphio: float,
    ) -> None:
        """Initialise a subclass deriving from this ABC."""

        cls.method = method
        cls.requires = requires
        cls.default_reference_kphio = default_reference_kphio
        QUANTUM_YIELD_CLASS_REGISTRY[cls.method] = cls


class QuantumYieldConstant(
    QuantumYieldABC,
    method="constant",
    requires=[],
    default_reference_kphio=0.049977,
):
    """Constant kphio."""

    def _calculate_kphio(self) -> None:
        """Constant kphio."""

        self.kphio = np.array([self.reference_kphio])


class QuantumYieldTemperature(
    QuantumYieldABC,
    method="temperature",
    requires=[],
    default_reference_kphio=0.081785,
):
    """Calculate temperature modulated kphio.

    This method follows  for C3 plants.
    """

    def _calculate_kphio(
        self,
    ) -> None:
        """Calculate kphio."""

        if self.use_c4:
            ftemp = evaluate_horner_polynomial(self.env.tc, self.pmodel_const.kphio_C4)
        else:
            ftemp = evaluate_horner_polynomial(self.env.tc, self.pmodel_const.kphio_C3)

        ftemp = np.clip(ftemp, 0.0, None)
        self.kphio = ftemp * self.reference_kphio


class QuantumYieldSandoval(
    QuantumYieldABC,
    method="sandoval",
    requires=["aridity_index", "mean_growth_temperature"],
    default_reference_kphio=1.0 / 9.0,
):
    """Calculate kphio following Sandoval.

    Reference kphio is the theoretical maximum quantum yield, defaulting to the ratio of
    1/9 in the absence of a Q cycle (Long, 1993).
    """

    def _calculate_kphio(self) -> None:
        """Calculate kphio."""

        # Warn that this is an experimental feature.
        warn(
            "The sandoval method for calculating kphio is experimental, "
            "see the class documentation",
            ExperimentalFeatureWarning,
        )

        # Calculate enzyme kinetics
        a_ent, b_ent, Hd_base, Ha = self.pmodel_const.sandoval_kinetics
        # Calculate activation entropy as a linear function of
        # mean growth temperature, J/mol/K
        deltaS = a_ent + b_ent * self.env.mean_growth_temperature
        # Calculate deaactivation energy J/mol
        Hd = Hd_base * deltaS

        # Calculate the optimal temperature to be used as the reference temperature in
        # the modified Arrhenius calculation
        Topt = Hd / (deltaS - self.core_const.k_R * np.log(Ha / (Hd - Ha)))

        # Calculate peak kphio given the aridity index
        m, n = self.pmodel_const.sandoval_peak_phio
        kphio_peak = self.reference_kphio / (1 + (self.env.aridity_index) ** m) ** n

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
        self.kphio = np.array([kphio_peak * f_kphio])
