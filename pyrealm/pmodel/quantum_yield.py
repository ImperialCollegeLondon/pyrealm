r"""The module :mod:`~pyrealm.pmodel.quantum_yield` provides the abstract base class
:class:`~pyrealm.pmodel.quantum_yield.QuantumYieldABC`, which is used to support
different implementations of the calculation of the intrinsic quantum yield efficiency
of photosynthesis (:math:`\phi_0`, unitless). The module then provides subclasses of the
ABC implementing different approaches.

Note that :math:`\phi_0` is sometimes used to refer to the quantum yield of electron
transfer, which is exactly four times larger, so check definitions here.
"""  # noqa D210, D415

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from pyrealm.core.experimental import warn_experimental
from pyrealm.core.utilities import (
    check_input_shapes,
    evaluate_horner_polynomial,
    summarize_attrs,
)
from pyrealm.pmodel.functions import calculate_kattge_knorr_arrhenius_factor
from pyrealm.pmodel.pmodel_environment import PModelEnvironment

QUANTUM_YIELD_CLASS_REGISTRY: dict[str, type[QuantumYieldABC]] = {}
r"""A registry for intrinsic quantum yield of photosynthesis calculation classes.

Different implementations of the calculation of the intrinsic quantum yield of
photosynthesis (:math:`\phi_{0}`) must all be subclasses of
:class:`~pyrealm.pmodel.quantum_yield.QuantumYieldABC` abstract base class. This
dictionary is used as a registry for defined subclasses and a method name is used to
retrieve a particular implementation from this registry. For example:

.. code:: python

    temperature_phio = QUANTUM_YIELD_CLASS_REGISTRY['temperature']
"""


class QuantumYieldABC(ABC):
    r"""ABC for calculating the intrinsic quantum yield of photosynthesis.

    This provides an abstract base class for the implementation of alternative
    approaches to calculating the the intrinsic quantum yield of photosynthesis. All
    implementations estimate the :math:`\phi_{0}` following values, which is then stored
    in the ``kphio`` attribute of the resulting class instance.

    The abstract base class requires that implementations of specific approaches defines
    the `calculate_kphio` method. The provides the approach specific calculation of
    ``kphio``  and is automatically called by the ``__init__`` method when a subclass
    instance is created.

    Subclasses must define several class attributes when created:

    .. code:: python

        class QuantumYieldFixed(
            QuantumYieldABC,
            method="method_name",
            requires=["an_environment_variable"],
            array_reference_kphio_ok=True,
        ):

    * The ``method`` argument sets the name of the method, which can then be used to
      select the implemented class from the
      :data:`~pyrealm.pmodel.quantum_yield.QUANTUM_YIELD_CLASS_REGISTRY`.
    * The `requires` argument sets a list of variables that must be present in the
      :class:`~pyrealm.pmodel.pmodel_environment.PModelEnvironment` to use this
      approach. The core ``tc``, ``vpd``, ``patm`` and ``co2`` variables do not need to
      be included in this list.
    * The ``array_reference_kphio_ok`` argument sets whether the method can accept an
      array of :math:`\phi_0` values or whether a single global reference value should
      be used.

    The definition of the ``_calculate_kphio`` method for subclasses can also provide C3
    and C4 implementations for calculate :math:`\phi_0` - or possibly raise an error for
    one pathway - using the ``use_c4`` attribute.

    Args:
        env: An instance of
            :class:`~pyrealm.pmodel.pmodel_environment.PModelEnvironment`  providing the
            photosynthetic environment for the model.
        reference_kphio: An optional value to be used instead of the global constant
            value . This is typically a single float but some approaches may support an
            array of values here.
        use_c4: Should the calculation use parameterisation for C4 photosynthesis rather
            than C3 photosynthesis.

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
    array_reference_kphio_ok: bool
    """Does the implementation handle arrays inputs to the reference_kphio __init__
    argument."""

    def __init__(
        self,
        env: PModelEnvironment,
        reference_kphio: float | NDArray | None = None,
        use_c4: bool = False,
    ):
        self.env: PModelEnvironment = env
        """The PModelEnvironment containing the photosynthetic environment for the
        model."""
        self.shape: tuple[int, ...] = env.shape
        """The shape of the input environment data."""

        # Set the reference kphio to the class default value if not provided and convert
        # the value to np.array if needed
        if reference_kphio is None:
            reference_kphio = self.env.pmodel_const.maximum_phi0
        if isinstance(reference_kphio, float | int):
            reference_kphio = np.array([reference_kphio])

        # Now check - if the reference_kphio value is a non-scalar array - that array
        # inputs are handled by the kphio method and that the shape matches the shape of
        # the environment.
        if isinstance(reference_kphio, np.ndarray) and reference_kphio.size > 1:
            if self.array_reference_kphio_ok:
                check_input_shapes(self.env.tc, reference_kphio)
            else:
                raise ValueError(
                    f"The {self.method} method for kphio does not support arrays "
                    "of reference kphio values"
                )

        self.reference_kphio: NDArray[np.float64] = reference_kphio
        """The kphio reference value for the method."""
        self.use_c4: bool = use_c4
        """Use a C4 parameterisation if available."""

        # Declare attributes populated by methods. These are typed but not assigned a
        # default value as they must are populated by the subclass specific
        # calculate_kphio method, which is called below to populate the values.
        self.kphio: NDArray[np.float64]
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

        attrs = (("kphio", "-"),)

        summarize_attrs(self, attrs, dp=dp)

    @classmethod
    def __init_subclass__(
        cls,
        method: str,
        requires: list[str],
        array_reference_kphio_ok: bool,
    ) -> None:
        """Initialise a subclass deriving from this ABC."""

        cls.method = method
        cls.requires = requires
        cls.array_reference_kphio_ok = array_reference_kphio_ok
        QUANTUM_YIELD_CLASS_REGISTRY[cls.method] = cls


class QuantumYieldFixed(
    QuantumYieldABC,
    method="fixed",
    requires=[],
    array_reference_kphio_ok=True,
):
    r"""Apply a fixed value for :math:`\phi_0`.

    This implementation applies a fixed value for the quantum yield without any
    environmental variation. It will accept an array of values to allow externally
    estimated values of `phi_0` to be passed to a P Model.
    """

    def _calculate_kphio(self) -> None:
        """Set fixed kphio."""

        self.kphio = self.reference_kphio


class QuantumYieldTemperature(
    QuantumYieldABC,
    method="temperature",
    requires=[],
    array_reference_kphio_ok=False,
):
    r"""Calculate temperature dependent of quantum yield efficiency.

    This implementation calculates temperature dependent quantum yield efficiency, as a
    quadratic function of temperature (:math:`T`).

    .. math::

        \phi(T) = a + b T - c T^2

    The values of :math:`a, b, c` are dependent on whether :math:`\phi_0` is being
    estimated for C3 or C4 photosynthesis. For C3 photosynthesis, the default values use
    the temperature dependence of the maximum quantum yield of photosystem II in
    light-adapted tobacco leaves determined by :cite:t:`Bernacchi:2003dc`. For C4
    photosynthesis, the default values are taken from :cite:t:`cai:2020a`.
    """

    def _calculate_kphio(
        self,
    ) -> None:
        """Calculate kphio."""

        if self.use_c4:
            ftemp = evaluate_horner_polynomial(
                self.env.tc, self.env.pmodel_const.kphio_C4
            )
        else:
            ftemp = evaluate_horner_polynomial(
                self.env.tc, self.env.pmodel_const.kphio_C3
            )

        ftemp = np.clip(ftemp, 0.0, None)
        self.kphio = ftemp * self.reference_kphio


class QuantumYieldSandoval(
    QuantumYieldABC,
    method="sandoval",
    requires=["aridity_index", "mean_growth_temperature"],
    array_reference_kphio_ok=False,
):
    r"""Calculate aridity and mean growth temperature effects on quantum yield.

    This experimental approach implements the method of :cite:t:`sandoval:in_prep`. This
    approach modifies the maximum possible :math:`\phi_0` as a function of the
    climatological aridity index. It then also adjusts the temperature at which the
    highest :math:`\phi_0` can be attained as a function of the mean growth temperature
    for an observation. It then calculates the expected :math:`\phi_0` as a function of
    temperature via a modified Arrhenius relationship.

    The reference kphio for this approach is the theoretical maximum quantum yield,
    defaulting to the ratio of 1/9 in the absence of a Q cycle :cite:`long:1993a`.
    """

    __experimental__: bool = True

    def peak_quantum_yield(
        self, aridity_index: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Calculate the peak quantum yield as a function of the aridity index.

        Args:
            aridity_index: An array of aridity index values.
        """

        # Calculate peak kphio given the aridity index
        m, n = self.env.pmodel_const.sandoval_peak_phio
        return self.reference_kphio / (1 + aridity_index**m) ** n

    def _calculate_kphio(self) -> None:
        """Calculate kphio."""

        # Warn that this is an experimental feature.
        warn_experimental("QuantumYieldSandoval")

        aridity_index = getattr(self.env, "aridity_index")
        mean_growth_temperature = getattr(self.env, "mean_growth_temperature")

        # Calculate enzyme kinetic. This needs to use a copy because the Hd value is
        # modified below.
        coef = self.env.pmodel_const.sandoval_kinetics.copy()

        # Calculate change in activation entropy as a linear function of
        # mean growth temperature, J/mol/K
        delta_entropy = (
            coef["entropy_intercept"] + coef["entropy_slope"] * mean_growth_temperature
        )
        # Calculate de-activation energy J/mol
        Hd = coef["hd"] * delta_entropy

        # Calculate the optimal temperature to be used as the reference temperature in
        # the modified Arrhenius calculation
        Topt = Hd / (
            delta_entropy
            - self.env.core_const.k_R * np.log(coef["ha"] / (Hd - coef["ha"]))
        )
        tk_leaf = self.env.tk
        # Calculate peak kphio given the aridity index
        kphio_peak = self.peak_quantum_yield(aridity_index=aridity_index)

        # Pass the modified Hd back into the calculation of the Arrhenius factor
        coef["hd"] = Hd

        # Calculate the modified Arrhenius factor using the
        f_kphio = calculate_kattge_knorr_arrhenius_factor(
            tk_leaf=tk_leaf,
            tk_ref=Topt,
            tc_growth=mean_growth_temperature,
            coef=coef,
            k_R=self.env.core_const.k_R,
        )

        # Apply the factor and store it.

        # The Sandoval implementation currently includes an additional factor on kphio
        # calculated as (tk_leaf / Topt). This might be the Murphy et al correction, so
        # might be removed, but at present this term is needed to match the R regression
        # test values

        self.kphio = kphio_peak * f_kphio * (tk_leaf / Topt)
