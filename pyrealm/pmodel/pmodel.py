"""The module :mod:`~pyrealm.pmodel.pmodel` provides the implementation of
the following  classes:

* :class:`~pyrealm.pmodel.pmodel.PModelABC`: An abstract base class providing some
  of the core functionality for initialising PModel subclasses.

* :class:`~pyrealm.pmodel.pmodel.PModel`: A subclass providing the standard
  implementation of the P Model.

* :class:`~pyrealm.pmodel.pmodel.SubdailyPModel`: A subclass providing the
  subdaily implementation of the P Model, which accounts for slow acclimation of core
  photosynthetic processes.


"""  # noqa D210, D415

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any
from warnings import warn

import numpy as np
from numpy.typing import NDArray

from pyrealm.constants import CoreConst, PModelConst
from pyrealm.core.utilities import summarize_attrs
from pyrealm.pmodel.acclimation import AcclimationModel
from pyrealm.pmodel.arrhenius import ARRHENIUS_METHOD_REGISTRY, ArrheniusFactorABC
from pyrealm.pmodel.jmax_limitation import (
    JMAX_LIMITATION_CLASS_REGISTRY,
    JmaxLimitationABC,
)
from pyrealm.pmodel.optimal_chi import OPTIMAL_CHI_CLASS_REGISTRY, OptimalChiABC
from pyrealm.pmodel.pmodel_environment import PModelEnvironment
from pyrealm.pmodel.quantum_yield import QUANTUM_YIELD_CLASS_REGISTRY, QuantumYieldABC


class PModelABC(ABC):
    r"""Abstract base class for the PModel and SubdailyPModel.

    The base class ``__init__`` implements the core arguments to the PModel subclasses:
    the forcing data to be used for the model and various methodological options for the
    calculation of the model parameters.

    Subclasses should define an ``__init__`` method that first calls
    ``super().__init__(...)`` to run the shared core functionality and then define any
    model specific attributes. The abstract base method ``_fit_model`` should then be
    defined and used to execute the model specific logic of the base class.

    Args:
        env: A :class:`~pyrealm.pmodel.pmodel_environment.PModelEnvironment` instance
        method_kphio: The method to use for calculating the quantum yield
            efficiency of photosynthesis (:math:`\phi_0`, unitless). The method name
            must be included in the
            :data:`~pyrealm.pmodel.quantum_yield.QUANTUM_YIELD_CLASS_REGISTRY`.
        method_optchi: (Optional, default=`prentice14`) Selects the method to be
            used for calculating optimal :math:`chi`. The choice of method also sets the
            choice of  C3 or C4 photosynthetic pathway (see
            :class:`~pyrealm.pmodel.optimal_chi.OptimalChiABC`).
        method_jmaxlim: (Optional, default=`wang17`) Method to use for
            :math:`J_{max}` limitation.
        method_arrhenius: (Optional, default=`simple`) Method to set the form of
            Arrhenius scaling used for `vcmax` and `jmax`.
        reference_kphio: An optional alternative reference value for the quantum yield
            efficiency of photosynthesis (:math:`\phi_0`, -) to be passed to the kphio
            calculation method.
    """

    _data_attributes: tuple[tuple[str, str], ...]
    """Data attributes included in the summarize table."""

    def __init__(
        self,
        env: PModelEnvironment,
        method_kphio: str = "temperature",
        method_optchi: str = "prentice14",
        method_jmaxlim: str = "wang17",
        method_arrhenius: str = "simple",
        reference_kphio: float | NDArray | None = None,
        **kwargs: dict[str, Any],
    ):
        self.shape: tuple = env.shape
        """Records the common numpy array shape of array inputs."""

        # Store a reference to the photosynthetic environment and a direct
        # reference to the parameterisation
        self.env: PModelEnvironment = env
        """The PModelEnvironment used to fit the P Model."""
        self.pmodel_const: PModelConst = env.pmodel_const
        """The PModelConst instance used to create the model environment."""
        self.core_const: CoreConst = env.core_const
        """The CoreConst instance used to create the model environment."""

        # -----------------------------------------------------------------------
        # Optimal Chi method setup
        # -----------------------------------------------------------------------

        self.method_optchi: str
        """The method used to calculate optimal chi."""
        self._optchi_class: type[OptimalChiABC]
        """The OptimalChiABC subclass to be used in the model."""
        self.optchi: OptimalChiABC
        r"""The optimal chi (:math:`\chi`) calculations for the fitted P Model."""

        self._method_setter(
            method_value_attr="method_optchi",
            method_class_attr="_optchi_class",
            method_value=method_optchi,
            method_registry=OPTIMAL_CHI_CLASS_REGISTRY,
        )

        # Set the C4 status
        self.c4: bool = self._optchi_class.is_c4
        """Boolean flag showing if the optimal chi method approximates a C3 or C4
        pathway."""

        # -----------------------------------------------------------------------
        # Cuantum yield of photosynthesis (kphio) setup
        # -----------------------------------------------------------------------

        self.method_kphio: str
        """The method used to calculate kphio."""
        self._kphio_class: type[QuantumYieldABC]
        """The QuantumYieldABC subclass to be used in the model."""
        self.kphio: QuantumYieldABC
        r"""The quantum yield (:math:`\phi_0`) calculations for the fitted P Model."""

        self._method_setter(
            method_value_attr="method_kphio",
            method_class_attr="_kphio_class",
            method_value=method_kphio,
            method_registry=QUANTUM_YIELD_CLASS_REGISTRY,
        )

        self.reference_kphio = reference_kphio
        """The value of the the reference kphio to be used in the model."""

        # -----------------------------------------------------------------------
        # Arrhenius scaling setup
        # -----------------------------------------------------------------------

        self.method_arrhenius = method_arrhenius
        """The method used to calculate Arrhenius factors."""

        self._arrhenius_class: type[ArrheniusFactorABC]
        """The ArrheniusFactorABC subclass to be used in the model."""

        self._method_setter(
            method_value_attr="method_arrhenius",
            method_class_attr="_arrhenius_class",
            method_value=method_arrhenius,
            method_registry=ARRHENIUS_METHOD_REGISTRY,
        )

        if self.method_arrhenius != "simple":
            warn(
                "We currently strongly recommend the use of the default `simple` "
                "method calculating Arrhenius factors."
            )

        # -----------------------------------------------------------------------
        # Jmax limitation term setup
        # -----------------------------------------------------------------------

        self.method_jmaxlim = method_jmaxlim
        """Records the method used to calculate Jmax limitation."""
        self._jmaxlim_class: type[JmaxLimitationABC]
        """The ArrheniusFactorABC subclass to be used in the model."""
        self.jmaxlim: JmaxLimitationABC
        """The Jmax limitation terms calculated for the model."""

        self._method_setter(
            method_value_attr="method_jmaxlim",
            method_class_attr="_jmaxlim_class",
            method_value=method_jmaxlim,
            method_registry=JMAX_LIMITATION_CLASS_REGISTRY,
        )

        # -----------------------------------------------------------------------
        # Define the other model attributes
        # -----------------------------------------------------------------------

        self.iwue: NDArray[np.float64]
        """Intrinsic water use efficiency (iWUE, µmol mol-1), calculated as:

        .. math::
        
            ( 5/8 * (c_a - c_i)) / P,
        
        where :math:`c_a` and :math:`c_i` are measured in Pa and :math:`P` is
        atmospheric pressure in megapascals.
        """

        self.lue: NDArray[np.float64]
        r"""Light use efficiency (LUE, g C mol-1), calculated as:

        .. math::

            \text{LUE} = \phi_0 \cdot m_j \cdot f_v \cdot M_C

        where :math:`f_v` is a limitation factor defined in
        :class:`~pyrealm.pmodel.jmax_limitation.JmaxLimitationABC` and :math:`M_C` is
        the molar mass of carbon.
        """

        self.vcmax: NDArray[np.float64]
        r"""Maximum rate of carboxylation at the growth temperature (µmol m-2 s-1),
        calculated as:

        .. math::

            V_{cmax} = \phi_{0} I_{abs} \frac{m}{m_c} f_{v} 

        where  :math:`f_v` is a limitation term calculated via the method selected in
        `method_jmaxlim`."""

        self.vcmax25: NDArray[np.float64]
        """Maximum rate of carboxylation at standard temperature (µmol m-2 s-1),
        estimated from :math:`V_{cmax}` using the selected method for Arrhenius scaling.
        """

        self.jmax: NDArray[np.float64]
        r"""Maximum rate of electron transport at the growth temperature (µmol m-2 s-1),
        calculated as:
        
        .. math::

            J_{max} = 4 \phi_{0} I_{abs} f_{j}

        where  :math:`f_j` is a limitation term calculated via the method selected in
        `method_jmaxlim`."""

        self.jmax25: NDArray[np.float64]
        """Maximum rate of electron transport at standard temperature (µmol m-2 s-1),
        estimated from :math:`J_{max}` using the selected method for Arrhenius scaling.
        """

        self.gpp: NDArray[np.float64]
        r"""Gross primary productivity (µg C m-2 s-1) calculated as:
        
        .. math::

            \text{GPP} = \text{LUE} \cdot I_{abs}
            
        where :math:`I_{abs}` is the absorbed photosynthetic radiation.
        """

        self.gs: NDArray[np.float64]
        r"""Stomatal conductance (µmol m-2 s-1), calculated as:

        .. math::

            g_s = \frac{LUE}{M_C}\frac{1}{c_a - c_i}

        When C4 photosynthesis is being used, the true partial pressure of CO2 in the
        substomatal cavities (:math:`c_i`) is used following the calculation of
        :math:`\chi` using
        :attr:`~pyrealm.constants.pmodel_const.PModelConst.beta_cost_ratio_c4`. Note
        that :math:`g_s \to \infty` as VPD :math:`\to 0` and hence :math:`(c_a - c_i)
        \to 0` and the reported values will be set to ``np.nan`` under these
        conditions."""

        self.A_c: NDArray[np.float64]
        """Maximum assimilation rate limited by carboxylation."""
        self.A_j: NDArray[np.float64]
        """Maximum assimilation rate limited by electron transport."""
        self.J: NDArray[np.float64]
        """Electron transfer rate."""

    @abstractmethod
    def _fit_model(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __repr__(self) -> str:
        """Generates a string representation of PModel instance."""

        return (
            f"{self.__class__.__name__}("
            f"shape={self.shape}, "
            f"method_optchi={self.method_optchi}, "
            f"method_arrhenius={self.method_arrhenius}, "
            f"method_jmaxlim={self.method_jmaxlim}, "
            f"method_kphio={self.method_kphio})"
        )

    def summarize(self, dp: int = 2) -> None:
        """Prints a summary of data attributes.

        Args:
            dp: The number of decimal places used in rounding summary stats.
        """

        summarize_attrs(self, self._data_attributes, dp=dp)

    def _method_setter(
        self,
        method_value_attr: str,
        method_class_attr: str,
        method_value: str,
        method_registry: dict,
    ) -> None:
        """Validate and set the method class attributes for a PModel method argument.

        This method checks that the provided method value is an option in the registry
        and then stores both the value and the appropriate method implementation class
        from the registry.

        Args:
            method_value_attr: The name of the attribute used to store the selected
                method name (e.g. ``method_optchi``)
            method_class_attr: The name of the attribute used to store the resulting
                selected class that implements the method
            method_value: The value passed to the method selection argument
            method_registry: A method registry, providing method class implementations
                keyed by method name values.
        """
        if method_value not in method_registry:
            raise ValueError(f"Unknown option for {method_value_attr}: {method_value}")

        setattr(self, method_value_attr, method_value)
        setattr(self, method_class_attr, method_registry[method_value])


class PModel(PModelABC):
    r"""Fit a standard P Model.

    This class fits the P Model to a given set of environmental and photosynthetic
    parameters. An extended description with typical use cases is given in
    :any:`pmodel_overview` but the basic flow of the model is:

    1. Estimate :math:`\ce{CO2}` limitation factors and optimal internal to ambient
       :math:`\ce{CO2}` partial pressure ratios (:math:`\chi`), using one of the
       methods based on :class:`~pyrealm.pmodel.optimal_chi.OptimalChiABC`.
    2. Estimate limitation factors to :math:`V_{cmax}` and :math:`J_{max}` using
       one of the methods implemented using
       :class:`~pyrealm.pmodel.jmax_limitation.JmaxLimitationABC`.

    Soil moisture effects:
        The `lavergne20_c3`, `lavergne20_c4`, ``prentice14_rootzonestress``,
        ``c4_rootzonestress`` and ``c4_no_gamma_rootzonestress`` options to
        ``method_optchi`` implement different approaches to soil moisture effects on
        photosynthesis. See also the alternative GPP penalty factors that can be applied
        after fitting the P Model
        (:func:`pyrealm.pmodel.functions.calc_soilmstress_stocker` and
        :func:`pyrealm.pmodel.functions.calc_soilmstress_mengoli`).

    Args:
        env: An instance of
           :class:`~pyrealm.pmodel.pmodel_environment.PModelEnvironment`
        method_kphio: The method to use for calculating the quantum yield
            efficiency of photosynthesis (:math:`\phi_0`, unitless). The method name
            must be included in the
            :data:`~pyrealm.pmodel.quantum_yield.QUANTUM_YIELD_CLASS_REGISTRY`.
        method_optchi: (Optional, default=`prentice14`) Selects the method to be
            used for calculating optimal :math:`chi`. The choice of method also sets the
            choice of  C3 or C4 photosynthetic pathway (see
            :class:`~pyrealm.pmodel.optimal_chi.OptimalChiABC`).
        method_jmaxlim: (Optional, default=`wang17`) Method to use for
            :math:`J_{max}` limitation.
        method_arrhenius: (Optional, default=`simple`) Method to set the form of
            Arrhenius scaling used for `vcmax` and `jmax`.
        reference_kphio: An optional alternative reference value for the quantum yield
            efficiency of photosynthesis (:math:`\phi_0`, -) to be passed to the kphio
            calculation method.
    """

    _data_attributes = (
        ("lue", "g C mol-1"),
        ("iwue", "µmol mol-1"),
        ("gpp", "µg C m-2 s-1"),
        ("vcmax", "µmol m-2 s-1"),
        ("vcmax25", "µmol m-2 s-1"),
        ("gs", "µmol m-2 s-1"),
        ("jmax", "µmol m-2 s-1"),
        ("jmax25", "µmol m-2 s-1"),
    )

    def __init__(
        self,
        env: PModelEnvironment,
        method_optchi: str = "prentice14",
        method_jmaxlim: str = "wang17",
        method_kphio: str = "temperature",
        method_arrhenius: str = "simple",
        reference_kphio: float | NDArray | None = None,
    ) -> None:
        # Initialise the superclass
        super().__init__(
            env=env,
            method_optchi=method_optchi,
            method_jmaxlim=method_jmaxlim,
            method_kphio=method_kphio,
            method_arrhenius=method_arrhenius,
            reference_kphio=reference_kphio,
        )
        # Fit the model
        self._fit_model()

    def _fit_model(self) -> None:
        """Fit the model.

        PPFD _must_ be provided in units of micromols per metre square per second (µmol
        m-2 s-1). This is required to ensure that values of :math:`J_{max}` and
        :math:`V_{cmax}` are also in µmol m-2 s-1.
        """

        warn(
            """
    Pyrealm 2.0.0 uses a new default for the quantum yield of photosynthesis (phi0=1/8).
    You may need to change settings to duplicate results from pyrealm 1.0.0.
            """,
            category=UserWarning,
        )

        # Calculate optimal chi
        self.optchi: OptimalChiABC = self._optchi_class(
            env=self.env,
            pmodel_const=self.env.pmodel_const,
        )
        self.c4 = self.optchi.is_c4

        # Calculate the quantum yield of photosynthesis (kphio)
        self.kphio = self._kphio_class(
            env=self.env,
            use_c4=self.c4,
            reference_kphio=self.reference_kphio,
        )

        # Calculation of Jmax limitation terms
        self.jmaxlim = self._jmaxlim_class(
            optchi=self.optchi,
            pmodel_const=self.env.pmodel_const,
        )

        # Get an Arrhenius instance for use in scaling rates
        arrhenius_factors = self._arrhenius_class(env=self.env)

        # Intrinsic water use efficiency (in µmol mol-1)
        self.iwue = (5 / 8 * (self.env.ca - self.optchi.ci)) / (1e-6 * self.env.patm)

        # Light use efficiency g Carbon per mol-1 of photons.
        self.lue = (
            self.kphio.kphio
            * self.optchi.mj
            * self.jmaxlim.f_v
            * self.core_const.k_c_molmass
        )

        # Calculate absorbed irradiance
        iabs = self.env.fapar * self.env.ppfd

        # GPP
        self.gpp = self.lue * iabs

        # Calculate V_cmax and J_max
        self.vcmax = self.kphio.kphio * iabs * self.optchi.mjoc * self.jmaxlim.f_v
        self.jmax = 4 * self.kphio.kphio * iabs * self.jmaxlim.f_j

        # - Calculate and apply the scaling factors.
        ftemp25_inst_vcmax = arrhenius_factors.calculate_arrhenius_factor(
            coefficients=self.pmodel_const.arrhenius_vcmax
        )
        self.vcmax25 = self.vcmax / ftemp25_inst_vcmax
        self.jmax25 = self.jmax / arrhenius_factors.calculate_arrhenius_factor(
            coefficients=self.pmodel_const.arrhenius_jmax
        )

        # AJ and AC
        self.A_j = self.kphio.kphio * iabs * self.optchi.mj * self.jmaxlim.f_v
        self.A_c = self.vcmax * self.optchi.mc

        if not np.allclose(self.A_j, self.A_c, equal_nan=True):
            raise RuntimeError(
                "Violation of coordination hypothesis: A_c is not equal to A_j"
            )

        # Stomatal conductance - do not estimate when VPD = 0 or when floating point
        # errors give rise to (ca - ci) < 0 and deliberately ignore the numpy divide by
        # zero warnings in those cases.
        ca_ci_diff = self.env.ca - self.optchi.ci
        with np.errstate(divide="ignore", invalid="ignore"):
            self.gs = np.where(
                np.logical_and(self.env.vpd > 0, ca_ci_diff > 0),
                self.A_c / ca_ci_diff,
                np.nan,
            )

    def to_subdaily(
        self,
        acclim_model: AcclimationModel,
        previous_realised: tuple[NDArray, NDArray, NDArray] | None = None,
    ) -> SubdailyPModel:
        r"""Convert a standard PModel to a subdaily P Model.

        This method converts a :class:`~pyrealm.pmodel.pmodel.PModel` instance to
        a to a :class:`~pyrealm.pmodel.pmodel.SubdailyPModel` instance with the
        same settings.

        Args:
            acclim_model: An AcclimationModel instance for the subdaily model.
            previous_realised: An optional set of arrays giving previous realised values
                for `xi`, `vcmax25` and `jmax25`.
        """

        return SubdailyPModel(
            env=self.env,
            method_optchi=self.method_optchi,
            method_arrhenius=self.method_arrhenius,
            method_jmaxlim=self.method_jmaxlim,
            method_kphio=self.method_kphio,
            reference_kphio=self.kphio.reference_kphio,
            acclim_model=acclim_model,
        )


class SubdailyPModel(PModelABC):
    r"""Fit a P Model incorporating acclimation in photosynthetic responses.

    The :class:`~pyrealm.pmodel.pmodel.PModel` implementation of the P Model
    assumes that plants instantaneously adopt optimal behaviour, which is reasonable
    where the data represents average conditions over longer timescales and the plants
    can be assumed to have acclimated to optimal behaviour. Over shorter timescales,
    this assumption is unwarranted and photosynthetic slow responses need to be
    included. This class implements the weighted-average approach of
    {cite:t}`mengoli:2022a`, but is extended to include the slow response of :math:`\xi`
    in addition to :math:`V_{cmax25}` and :math:`J_{max25}`.

    The workflow of the model:

    * The first dimension of the data arrays used to create the
      :class:`~pyrealm.pmodel.pmodel_environment.PModelEnvironment` instance must
      represent the time axis of the observations. The ``acclim_model`` argument is used
      to provide a :class:`~pyrealm.pmodel.acclimation.AcclimationModel` instance that
      sets the dates and time of those observations. One of the ``set_`` methods to that
      class must also be used to define a daily acclimation window that will be used to
      estimate the optimal daily behaviour of the plant.
    * The
      :meth:`AcclimationModel.get_daily_means<pyrealm.pmodel.acclimation.AcclimationModel.get_daily_means>`
      method is then used to extract daily average values for forcing variables from
      within the acclimation window, setting the conditions that the plant will optimise
      to.
    * A standard P Model is then run on those daily forcing values to generate predicted
      states for photosynthetic parameters that give rise to optimal productivity in
      that window.
    * The
      :meth:`AcclimationModel.apply_acclimation<pyrealm.pmodel.acclimation.AcclimationModel.apply_acclimation>`
      method is then used to calculate acclimating values for
      :math:`\xi`, :math:`V_{cmax25}` and :math:`J_{max25}`. These values are the actual
      realised values that will be used in calculating the P Model and which reflect the
      slow responses of those parameters to changing conditions. The speed of
      acclimation is controlled by the
      :attr:`AcclimationModel.alpha<pyrealm.pmodel.acclimation.AcclimationModel.alpha>`
      attribute, which sets the weight :math:`\alpha \in [0,1]` for an exponential
      moving average (see :meth:`pyrealm.core.utilities.exponential_moving_average`)
      Higher values of `alpha` give more rapid acclimation: :math:`\alpha=1`
      results in immediate acclimation and :math:`\alpha=0` results in no acclimation at
      all, with values pinned to the initial estimates.
    * By default, the initial values of the acclimated variables are taken to be the
      same as the initial optimal values. The `previous_realised` argument can be used
      to provide alternative initial realised values. This allows subdaily P models to
      be `restarted` using estimates of acclimated values.
    * The realised values are then filled back onto the original subdaily timescale,
      using the
      :meth:`AcclimationModel.fill_daily_to_subdaily<pyrealm.pmodel.acclimation.AcclimationModel.fill_daily_to_subdaily>`
      method. The subdaily values of :math:`V_{cmax}` and :math:`J_{max}` are calculated
      by using Arrhenius scaling to convert the acclimated values of :math:`V_{cmax25}`
      and :math:`J_{max25}` to the actual subdaily
      temperature observations. The value of :math:`c_i` is calculated using the
      acclimating values of :math:`\xi` but the actual subdaily values of temperature
      and vapour pressure deficit.
    * Predictions of GPP are then made as in the standard P Model.

    As with the :class:`~pyrealm.pmodel.pmodel.PModel`, the values of the `kphio`
    argument _can_ be provided as an array of values, potentially varying through time
    and space. The behaviour of the daily model that drives acclimation here is to take
    the daily mean `kphio` value for each time series within the acclimation window, as
    for the other variables. This is an experimental solution!

    Missing values:

        Missing data can arise in a number of ways: actual gaps in the forcing data, the
        observations starting part way through a day and missing some or all of the
        acclimation window for the day, or undefined values in P Model predictions. Some
        options include:

        * The ``allow_partial_data`` argument is passed on to the
          :meth:`~pyrealm.pmodel.acclimation.AcclimationModel.get_daily_means` method to
          allow daily optimum conditions to be calculated when the data in the
          acclimation window is incomplete. This does not fix problems when no data is
          present in the window or when the P Model predictions for a day are undefined.

        * The ``allow_holdover`` argument is passed on to the
          :meth:`~pyrealm.pmodel.acclimation.AcclimationModel.apply_acclimation` method
          to set whether missing values in the optimal predictions can be filled by
          holding over previous valid values.

    Args:
        env: An instance of
           :class:`~pyrealm.pmodel.pmodel_environment.PModelEnvironment`. The first
           dimension of the data must be time with an equal length to the acclimation
           model or a length of 1 if constant in time.
        acclim_model: An instance of
            :class:`~pyrealm.pmodel.acclimation.AcclimationModel`
        method_kphio: The method to use for calculating the quantum yield
            efficiency of photosynthesis (:math:`\phi_0`, unitless). The method name
            must be included in the
            :data:`~pyrealm.pmodel.quantum_yield.QUANTUM_YIELD_CLASS_REGISTRY`.
        method_optchi: (Optional, default=`prentice14`) Selects the method to be
            used for calculating optimal :math:`chi`. The choice of method also sets the
            choice of  C3 or C4 photosynthetic pathway (see
            :class:`~pyrealm.pmodel.optimal_chi.OptimalChiABC`).
        method_jmaxlim: (Optional, default=`wang17`) Method to use for
            :math:`J_{max}` limitation.
        method_arrhenius: (Optional, default=`simple`) Method to set the form of
            Arrhenius scaling used for `vcmax` and `jmax`.
        reference_kphio: An optional alternative reference value for the quantum yield
            efficiency of photosynthesis (:math:`\phi_0`, -) to be passed to the kphio
            calculation method.
        previous_realised: A tuple of previous realised values of three NumPy arrays
            (xi_real, vcmax25_real, jmax25_real).
    """

    _data_attributes = (
        ("iwue", "µmol mol-1"),
        ("gpp", "µg C m-2 s-1"),
        ("vcmax", "µmol m-2 s-1"),
        ("vcmax25", "µmol m-2 s-1"),
        ("gs", "µmol m-2 s-1"),
        ("jmax", "µmol m-2 s-1"),
        ("jmax25", "µmol m-2 s-1"),
    )

    def __init__(
        self,
        env: PModelEnvironment,
        acclim_model: AcclimationModel,
        method_optchi: str = "prentice14",
        method_jmaxlim: str = "wang17",
        method_kphio: str = "temperature",
        method_arrhenius: str = "simple",
        reference_kphio: float | NDArray | None = None,
        previous_realised: dict[str, NDArray] | None = None,
    ) -> None:
        # Initialise the superclass
        super().__init__(
            env=env,
            method_optchi=method_optchi,
            method_jmaxlim=method_jmaxlim,
            method_kphio=method_kphio,
            method_arrhenius=method_arrhenius,
            reference_kphio=reference_kphio,
        )

        # Subclass specific attributes
        self.acclim_model: AcclimationModel
        """The acclimation model used in the subdaily P Model."""
        self.previous_realised: Mapping[str, NDArray | None]
        """A dictionary of arrays of previous realised values for the acclimating
        variables 'xi', 'jmax25' and 'vcmax25'. If none were provided, the dictionary
        values are None."""

        # Other attributes
        self.pmodel_acclim: PModel
        r"""P Model predictions for the daily acclimation conditions.

        A :class:`~pyrealm.pmodel.pmodel.PModel` instance providing the
        predictions of the P Model for the daily acclimation conditions set for the
        SubdailyPModel. The model is used to obtain predictions of the instantaneous
        optimal estimates of :math:`V_{cmax}`, :math:`J_{max}` and :math:`\xi` during
        the acclimation window. These are then used to estimate realised values of those
        parameters given slow responses to acclimation.
        """

        # TODO - maybe encapsulate these in a dataclass?
        self.vcmax25_daily_optimal: NDArray[np.float64]
        r"""Daily optimal values in acclimation window for :math:`V_{cmax}`, scaled to
         standard temperature (:math:`V_{cmax25}`)."""
        self.vcmax25_daily_realised: NDArray[np.float64]
        r"""Realised daily responses in :math:`V_{cmax25}`"""
        self.jmax25_daily_optimal: NDArray[np.float64]
        r"""Daily optimal values in acclimation window for :math:`J_{max}`, scaled to
         standard temperature (:math:`J_{max25}`)."""
        self.jmax25_daily_realised: NDArray[np.float64]
        r"""Realised daily responses in :math:`J_{max25}`"""
        self.xi_daily_optimal: NDArray[np.float64]
        r"""Daily optimal values in acclimation window for :math:`\xi`"""
        self.xi_daily_realised: NDArray[np.float64]
        r"""Realised daily responses in :math:`\xi`"""

        # Fit the model
        self._fit_model(acclim_model=acclim_model, previous_realised=previous_realised)

    def _fit_model(
        self,
        acclim_model: AcclimationModel,
        previous_realised: dict[str, NDArray] | None,
    ) -> None:
        """Calculation logic of the subdaily P Model."""

        # Validate subdaily model specific arguments
        # * Check that the number of datetimes in the AcclimationModel is the same as
        #   the length of the first axis of the photosynthetic environment and that the
        #   one of the set_ methods has been run on the AcclimationModel.

        # Store the acclimation model
        self.acclim_model = acclim_model

        datetimes = self.acclim_model.datetimes
        if self.shape[0] not in (datetimes.shape[0], 1):
            raise ValueError(
                "The first dimension of the PModelEnvironment data is a different "
                "length to the number of AcclimationModel datetimes."
            )

        if not hasattr(self.acclim_model, "include"):
            raise ValueError(
                "The daily sampling window has not been set in the AcclimationModel"
            )
        # * Validate the previous realised values and standardise the internal
        #   representation as a dictionary.
        if previous_realised is None:
            self.previous_realised = {"xi": None, "jmax25": None, "vcmax25": None}
        else:
            # Is the fill method set to previous
            if self.acclim_model.fill_method != "previous":
                raise NotImplementedError(
                    "Using previous_realised is only implemented for "
                    "fill_method = 'previous'"
                )

            # Check it is a dictionary of numpy arrays for the three required variables
            if not (
                isinstance(previous_realised, dict)
                and (set(["xi", "jmax25", "vcmax25"]) == previous_realised.keys())
                and all(
                    [isinstance(val, np.ndarray) for val in previous_realised.values()]
                )
            ):
                raise ValueError(
                    "previous_realised must be a dictionary of arrays, with entries "
                    "for 'xi', 'jmax25' and 'vcmax25'."
                )

            # All variables should share the shape of a slice along the first axis of
            # the environmental forcings. Need to tell mypy to shut up - it does not
            # know that the values in previous_realised are confirmed to be arrays by
            # the code above

            try:
                for values in previous_realised.values():
                    _ = np.broadcast_shapes(self.shape, values.shape)
            except ValueError:
                raise ValueError(
                    "`previous_realised` arrays have wrong shape in SubdailyPModel"
                )

            self.previous_realised = previous_realised

        # 1) Generate a PModelEnvironment containing the average conditions within the
        #    daily acclimation window. This daily average environment also needs to also
        #    pass through any optional variables required by the optimal chi and kphio
        #    method set for the model, which can be accessed via the class requires
        #    attribute.

        # Get the list of variables for which to calculate daily acclimation conditions.
        daily_environment_vars = [
            "tc",
            "co2",
            "patm",
            "vpd",
            "fapar",
            "ppfd",
            *self._optchi_class.requires,
            *self._kphio_class.requires,
        ]

        # Construct a dictionary of daily acclimation variables, handling optional
        # choices which can be None.
        daily_environment: dict[str, NDArray] = {}
        for env_var_name in daily_environment_vars:
            env_var = getattr(self.env, env_var_name)
            if env_var is not None:
                # Broadcast the first dimension (if constant) since the acclimation
                # model requires this dimension to be equal to the number of times
                env_var_shape = (1,) * (len(self.shape) - env_var.ndim) + env_var.shape
                env_var_bcast = np.broadcast_to(
                    env_var,
                    [datetimes.shape[0], *env_var_shape[1:]],
                )
                daily_environment[env_var_name] = self.acclim_model.get_daily_means(
                    values=env_var_bcast,
                )

        # Calculate the acclimation environment passing on the constants definitions.
        pmodel_env_acclim: PModelEnvironment = PModelEnvironment(
            **daily_environment,
            bounds_checker=self.env._bounds_checker,
            pmodel_const=self.env.pmodel_const,
            core_const=self.env.core_const,
        )

        # Handle the kphio settings. First, calculate kphio at the subdaily scale.
        self.kphio: QuantumYieldABC = self._kphio_class(
            env=self.env,
            use_c4=self.c4,
            reference_kphio=self.reference_kphio,
        )

        # If the kphio method takes a single reference value then we can simply
        # recalculate the kphio using the same method for the daily acclimation
        # conditions but if the reference value is an array then the correct behaviour
        # is not obvious: currently, use the mean calculated kphio within the window to
        # calculate the daily acclimation value behaviour and set the kphio method to be
        # fixed to avoid altering the inputs.
        if self.kphio.reference_kphio.size > 1:
            daily_reference_kphio = self.acclim_model.get_daily_means(self.kphio.kphio)
            daily_method_kphio = "fixed"
        else:
            daily_reference_kphio = self.kphio.reference_kphio
            daily_method_kphio = self.method_kphio

        # 2) Fit a PModel to those environmental conditions, using the supplied settings
        #    for the original model.
        self.pmodel_acclim = PModel(
            env=pmodel_env_acclim,
            method_kphio=daily_method_kphio,
            method_optchi=self.method_optchi,
            method_jmaxlim=self.method_jmaxlim,
            method_arrhenius=self.method_arrhenius,
            reference_kphio=daily_reference_kphio,
        )
        self.pmodel_acclim._fit_model()

        # 4) Calculate the daily optimal values. Xi is simply the value from the optimal
        #   chi calculation but jmax and vcmax are scaled to values at 25°C using an
        #   instance of the requested Arrhenius scaling method .

        self.xi_daily_optimal = self.pmodel_acclim.optchi.xi

        arrhenius_daily = self._arrhenius_class(env=self.pmodel_acclim.env)

        self.vcmax25_daily_optimal = (
            self.pmodel_acclim.vcmax
            / arrhenius_daily.calculate_arrhenius_factor(
                coefficients=self.env.pmodel_const.arrhenius_vcmax
            )
        )
        self.jmax25_daily_optimal = (
            self.pmodel_acclim.jmax
            / arrhenius_daily.calculate_arrhenius_factor(
                coefficients=self.env.pmodel_const.arrhenius_jmax
            )
        )

        # 5) Calculate the realised daily values from the instantaneous optimal values

        self.xi_daily_realised = self.acclim_model.apply_acclimation(
            values=self.xi_daily_optimal,
            initial_values=self.previous_realised["xi"],
        )

        self.vcmax25_daily_realised = self.acclim_model.apply_acclimation(
            values=self.vcmax25_daily_optimal,
            initial_values=self.previous_realised["vcmax25"],
        )

        self.jmax25_daily_realised = self.acclim_model.apply_acclimation(
            values=self.jmax25_daily_optimal,
            initial_values=self.previous_realised["jmax25"],
        )

        # 6) Fill the realised xi, jmax25 and vcmax25 from daily values back to the
        # subdaily timescale.
        self.xi = self.acclim_model.fill_daily_to_subdaily(
            values=self.xi_daily_realised,
            previous_values=self.previous_realised["xi"],
        )
        self.vcmax25 = self.acclim_model.fill_daily_to_subdaily(
            self.vcmax25_daily_realised,
            previous_values=self.previous_realised["vcmax25"],
        )
        self.jmax25 = self.acclim_model.fill_daily_to_subdaily(
            self.jmax25_daily_realised,
            previous_values=self.previous_realised["jmax25"],
        )

        # 7) Adjust subdaily jmax25 and vcmax25 back to jmax and vcmax given the
        #    actual subdaily temperatures.
        arrhenius_subdaily = self._arrhenius_class(env=self.env)

        self.vcmax: NDArray[np.float64] = (
            self.vcmax25
            * arrhenius_subdaily.calculate_arrhenius_factor(
                coefficients=self.env.pmodel_const.arrhenius_vcmax
            )
        )

        self.jmax: NDArray[np.float64] = (
            self.jmax25
            * arrhenius_subdaily.calculate_arrhenius_factor(
                coefficients=self.env.pmodel_const.arrhenius_jmax
            )
        )

        # 8) Recalculate chi, but using the realised values of the xi parameter at the
        #    subdaily scale, not the instantaneous values of xi.
        self.optchi = self._optchi_class(
            env=self.env, pmodel_const=self.env.pmodel_const
        )
        self.optchi.estimate_chi(xi_values=self.xi)

        # Calculate Ac, J and Aj at subdaily scale to calculate assimilation
        self.A_c = self.vcmax * self.optchi.mc

        iabs = self.env.fapar * self.env.ppfd

        self.J = (4 * self.kphio.kphio * iabs) / np.sqrt(
            1 + ((4 * self.kphio.kphio * iabs) / self.jmax) ** 2
        )

        self.A_j = (self.J / 4) * self.optchi.mj

        # Calculate GPP and convert from mol to gC
        assimilation = np.minimum(self.A_j, self.A_c)
        self.gpp = assimilation * self.env.core_const.k_c_molmass

        # Stomatal conductance - do not estimate when VPD = 0 or when floating point
        # errors give rise to (ca - ci) < 0 and deliberately ignore the numpy divide by
        # zero warnings in those cases.
        ca_ci_diff = self.env.ca - self.optchi.ci
        with np.errstate(divide="ignore", invalid="ignore"):
            self.gs = np.where(
                np.logical_and(self.env.vpd > 0, ca_ci_diff > 0),
                self.A_c / ca_ci_diff,
                np.nan,
            )

        # Intrinsic water use efficiency (in µmol mol-1)
        self.iwue = (5 / 8 * (self.env.ca - self.optchi.ci)) / (1e-6 * self.env.patm)

    @property
    def lue(self) -> None:  # type: ignore[override]
        """The Subdaily P Model does not predict light use efficiency."""
        raise AttributeError(
            "The Subdaily P Model does not predict light use efficiency."
        )
